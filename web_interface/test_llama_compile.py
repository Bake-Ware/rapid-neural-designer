#!/usr/bin/env python3
"""
Test that the LLaMA Mini graph (llama_mini.json) compiles to valid Python
via the Rapid Neural Designer's handlebars-style template compiler.
"""

import json
import re
import subprocess
import sys
import os
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

GRAPH_PATH = os.path.join(SCRIPT_DIR, "models", "llama_mini.json")
COMPONENT_PATHS = {
    "molecular/embedding":         os.path.join(SCRIPT_DIR, "components", "embedding.json"),
    "molecular/transformer_block": os.path.join(SCRIPT_DIR, "components", "transformer_block.json"),
    "molecular/rmsnorm":           os.path.join(SCRIPT_DIR, "components", "rmsnorm.json"),
    "molecular/linear":            os.path.join(SCRIPT_DIR, "components", "linear.json"),
}
ATOMIC_PATHS = {
    "data": os.path.join(SCRIPT_DIR, "atomics", "data.json"),
    "init": os.path.join(SCRIPT_DIR, "atomics", "init.json"),
}
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "llama_compiled.py")


def load_graph(path):
    with open(path) as f:
        return json.load(f)


def load_component_defs():
    """Load all molecular component definitions (single-object JSON files)."""
    defs = {}
    for type_key, path in COMPONENT_PATHS.items():
        with open(path) as f:
            data = json.load(f)
        defs[type_key] = data
    return defs


def load_atomic_defs():
    """Load atomic definitions (array-of-objects JSON files), keyed by 'atomic/{category}/{id}'."""
    defs = {}
    for category, path in ATOMIC_PATHS.items():
        with open(path) as f:
            items = json.load(f)
        for item in items:
            type_key = f"atomic/{category}/{item['id']}"
            defs[type_key] = item
    return defs


def build_link_map(graph):
    """
    Build two maps from the links array.
    Each link is [link_id, src_node_id, src_slot, dst_node_id, dst_slot, type].
    Returns:
      - link_by_id: {link_id -> (src_node_id, src_slot, dst_node_id, dst_slot)}
      - node_map: {node_id -> node_dict}
    """
    link_by_id = {}
    for link in graph["links"]:
        lid, src_nid, src_slot, dst_nid, dst_slot, _typ = link
        link_by_id[lid] = (src_nid, src_slot, dst_nid, dst_slot)
    node_map = {n["id"]: n for n in graph["nodes"]}
    return link_by_id, node_map


def topological_sort(graph, link_by_id, node_map):
    """Sort nodes so every node's inputs are computed before it runs."""
    # Build adjacency: for each link, src_node -> dst_node
    in_degree = defaultdict(int)
    dependents = defaultdict(set)
    all_node_ids = [n["id"] for n in graph["nodes"]]
    for nid in all_node_ids:
        in_degree[nid] = 0  # ensure every node is present

    for _lid, (src_nid, _ss, dst_nid, _ds) in link_by_id.items():
        dependents[src_nid].add(dst_nid)
        in_degree[dst_nid] += 1

    # Kahn's algorithm
    queue = [nid for nid in all_node_ids if in_degree[nid] == 0]
    ordered = []
    while queue:
        queue.sort()  # deterministic order among equal-priority nodes
        nid = queue.pop(0)
        ordered.append(nid)
        for dep in sorted(dependents[nid]):
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                queue.append(dep)

    assert len(ordered) == len(all_node_ids), "Graph has a cycle!"
    return ordered


def make_var_name(def_id, node_id, output_name):
    """Generate a Python variable name for a node's output port."""
    return f"{def_id}_{node_id}_{output_name}"


def resolve_definition(node_type, comp_defs, atomic_defs):
    """Find the component/atomic definition for a given node type."""
    if node_type in comp_defs:
        return comp_defs[node_type]
    if node_type in atomic_defs:
        return atomic_defs[node_type]
    raise ValueError(f"No definition found for node type: {node_type}")


def python_value(val):
    """Convert a JSON value to its Python source representation."""
    if isinstance(val, bool):
        return "True" if val else "False"
    if isinstance(val, str):
        return val  # strings used as-is (they appear inline in templates)
    return str(val)


def substitute_template(code_template, substitutions):
    """
    Replace all {{key}} placeholders in the code template.
    Uses a single regex pass, falling back to the raw placeholder
    if no substitution is found (to help catch bugs).
    """
    def replacer(m):
        key = m.group(1)
        if key in substitutions:
            return substitutions[key]
        return m.group(0)  # leave unchanged if not found
    return re.sub(r"\{\{(\w+)\}\}", replacer, code_template)


def compile_graph(graph, comp_defs, atomic_defs):
    """Compile the LiteGraph JSON into a Python source string."""
    link_by_id, node_map = build_link_map(graph)
    ordered_ids = topological_sort(graph, link_by_id, node_map)

    all_imports = set()
    code_blocks = []

    for nid in ordered_ids:
        node = node_map[nid]
        defn = resolve_definition(node["type"], comp_defs, atomic_defs)
        def_id = defn["id"]

        # ---- Build substitution map ----
        subs = {"_id": str(nid)}

        # Properties
        for key, val in node.get("properties", {}).items():
            subs[key] = python_value(val)

        # Outputs -> variable names
        for idx, out in enumerate(node.get("outputs", [])):
            var = make_var_name(def_id, nid, out["name"])
            subs[out["name"]] = var

        # Inputs -> resolve via links to source variable names
        for idx, inp in enumerate(node.get("inputs", [])):
            link_id = inp.get("link")
            if link_id is None:
                subs[inp["name"]] = "None"
                continue
            src_nid, src_slot, _dst_nid, _dst_slot = link_by_id[link_id]
            src_node = node_map[src_nid]
            src_defn = resolve_definition(src_node["type"], comp_defs, atomic_defs)
            src_output = src_node["outputs"][src_slot]
            src_var = make_var_name(src_defn["id"], src_nid, src_output["name"])
            subs[inp["name"]] = src_var

        # ---- Substitute template ----
        code = substitute_template(defn["code"], subs)

        # Add comment header
        title = node.get("title", defn.get("name", def_id))
        code_blocks.append(f"# --- Node {nid}: {title} ---")
        code_blocks.append(code)
        code_blocks.append("")  # blank line

        # Collect imports
        for imp in defn.get("imports", []):
            all_imports.add(imp)

    # ---- Assemble final script ----
    header = '"""Auto-generated by Rapid Neural Designer compiler."""\n'
    import_block = "\n".join(sorted(all_imports))
    body = "\n".join(code_blocks)
    return f"{header}\n{import_block}\n\n{body}"


def main():
    print("Loading graph and definitions...")
    graph = load_graph(GRAPH_PATH)
    comp_defs = load_component_defs()
    atomic_defs = load_atomic_defs()

    print(f"Graph has {len(graph['nodes'])} nodes and {len(graph['links'])} links")

    print("Compiling graph...")
    source = compile_graph(graph, comp_defs, atomic_defs)

    print(f"Writing compiled code to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w") as f:
        f.write(source)

    # Print the generated source for inspection
    print("\n" + "=" * 60)
    print("GENERATED SOURCE CODE:")
    print("=" * 60)
    for i, line in enumerate(source.split("\n"), 1):
        print(f"  {i:3d} | {line}")
    print("=" * 60 + "\n")

    # Syntax check
    print("Checking Python syntax...")
    try:
        compile(source, OUTPUT_PATH, "exec")
        print("  Syntax OK")
    except SyntaxError as e:
        print(f"  SYNTAX ERROR: {e}")
        sys.exit(1)

    # Execute the compiled code
    print("\nExecuting compiled code...")
    result = subprocess.run(
        [sys.executable, OUTPUT_PATH],
        capture_output=True,
        text=True,
        timeout=120,
    )

    print(f"  Return code: {result.returncode}")
    if result.stdout.strip():
        print(f"  STDOUT:\n    {result.stdout.strip()}")
    if result.stderr.strip():
        print(f"  STDERR:\n    {result.stderr.strip()}")

    if result.returncode == 0:
        print("\nSUCCESS: Compiled LLaMA graph executed correctly!")
    else:
        print("\nFAILURE: Compiled code did not execute successfully.")
        sys.exit(1)


if __name__ == "__main__":
    main()
