#!/usr/bin/env python3
"""
Test that the 5 new model graphs (lstm, autoencoder, vae, seq2seq, bert)
compile to valid, runnable Python code via the Rapid Neural Designer's
handlebars-style template compiler.
"""

import json
import re
import subprocess
import sys
import os
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# All model graphs to test
MODEL_GRAPHS = {
    "lstm":        os.path.join(SCRIPT_DIR, "models", "lstm.json"),
    "autoencoder": os.path.join(SCRIPT_DIR, "models", "autoencoder.json"),
    "vae":         os.path.join(SCRIPT_DIR, "models", "vae.json"),
    "seq2seq":     os.path.join(SCRIPT_DIR, "models", "seq2seq.json"),
    "bert":        os.path.join(SCRIPT_DIR, "models", "bert.json"),
}

# Component definitions (molecular - single-object JSON files)
COMPONENT_PATHS = {
    "molecular/embedding":              os.path.join(SCRIPT_DIR, "components", "embedding.json"),
    "molecular/linear":                 os.path.join(SCRIPT_DIR, "components", "linear.json"),
    "molecular/lstm_layer":             os.path.join(SCRIPT_DIR, "components", "lstm_layer.json"),
    "molecular/rnn_layer":              os.path.join(SCRIPT_DIR, "components", "rnn_cell.json"),
    "molecular/rmsnorm":                os.path.join(SCRIPT_DIR, "components", "rmsnorm.json"),
    "molecular/layernorm":              os.path.join(SCRIPT_DIR, "components", "layernorm.json"),
    "molecular/multi_head_attention":   os.path.join(SCRIPT_DIR, "components", "multi_head_attention.json"),
    "molecular/bidirectional_attention": os.path.join(SCRIPT_DIR, "components", "bidirectional_attention.json"),
    "molecular/cross_attention":        os.path.join(SCRIPT_DIR, "components", "cross_attention.json"),
    "molecular/swiglu_ffn":             os.path.join(SCRIPT_DIR, "components", "swiglu_ffn.json"),
    "molecular/conv2d":                 os.path.join(SCRIPT_DIR, "components", "conv2d.json"),
    "molecular/maxpool2d":              os.path.join(SCRIPT_DIR, "components", "maxpool2d.json"),
    "molecular/flatten":                os.path.join(SCRIPT_DIR, "components", "flatten.json"),
    "molecular/gat_layer":              os.path.join(SCRIPT_DIR, "components", "gat_layer.json"),
}

# Atomic definitions (array-of-objects JSON files, keyed by category)
ATOMIC_PATHS = {
    "data":  os.path.join(SCRIPT_DIR, "atomics", "data.json"),
    "init":  os.path.join(SCRIPT_DIR, "atomics", "init.json"),
    "trig":  os.path.join(SCRIPT_DIR, "atomics", "trig.json"),
    "math":  os.path.join(SCRIPT_DIR, "atomics", "math.json"),
    "shape": os.path.join(SCRIPT_DIR, "atomics", "shape.json"),
}


# ---------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------

def load_graph(path):
    with open(path) as f:
        return json.load(f)


def load_component_defs():
    """Load all molecular component definitions (single-object JSON files)."""
    defs = {}
    for type_key, path in COMPONENT_PATHS.items():
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            defs[type_key] = data
    return defs


def load_atomic_defs():
    """Load atomic definitions (array-of-objects JSON files), keyed by 'atomic/{category}/{id}'."""
    defs = {}
    for category, path in ATOMIC_PATHS.items():
        if not os.path.exists(path):
            continue
        with open(path) as f:
            items = json.load(f)
        for item in items:
            type_key = f"atomic/{category}/{item['id']}"
            defs[type_key] = item
    return defs


def build_link_map(graph):
    """Build maps from the links array."""
    link_by_id = {}
    for link in graph["links"]:
        lid, src_nid, src_slot, dst_nid, dst_slot, _typ = link
        link_by_id[lid] = (src_nid, src_slot, dst_nid, dst_slot)
    node_map = {n["id"]: n for n in graph["nodes"]}
    return link_by_id, node_map


def topological_sort(graph, link_by_id, node_map):
    """Sort nodes so every node's inputs are computed before it runs."""
    in_degree = defaultdict(int)
    dependents = defaultdict(set)
    all_node_ids = [n["id"] for n in graph["nodes"]]
    for nid in all_node_ids:
        in_degree[nid] = 0

    for _lid, (src_nid, _ss, dst_nid, _ds) in link_by_id.items():
        dependents[src_nid].add(dst_nid)
        in_degree[dst_nid] += 1

    queue = [nid for nid in all_node_ids if in_degree[nid] == 0]
    ordered = []
    while queue:
        queue.sort()
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
        return val
    return str(val)


def substitute_template(code_template, substitutions):
    """Replace all {{key}} placeholders in the code template."""
    def replacer(m):
        key = m.group(1)
        if key in substitutions:
            return substitutions[key]
        return m.group(0)
    return re.sub(r"\{\{(\w+)\}\}", replacer, code_template)


def compile_graph(graph, comp_defs, atomic_defs):
    """Compile a LiteGraph JSON into a Python source string."""
    link_by_id, node_map = build_link_map(graph)
    ordered_ids = topological_sort(graph, link_by_id, node_map)

    all_imports = set()
    code_blocks = []

    for nid in ordered_ids:
        node = node_map[nid]
        defn = resolve_definition(node["type"], comp_defs, atomic_defs)
        def_id = defn.get("id", defn["name"].lower().replace(" ", "_"))

        # Build substitution map
        subs = {"_id": str(nid)}

        # Properties
        for key, val in node.get("properties", {}).items():
            subs[key] = python_value(val)

        # Outputs -> variable names
        for idx, out in enumerate(defn.get("outputs", [])):
            var = make_var_name(def_id, nid, out["name"])
            subs[out["name"]] = var

        # Inputs -> resolve via links to source variable names
        for idx, inp in enumerate(defn.get("inputs", [])):
            # Find the link id from the node's inputs array
            node_inputs = node.get("inputs", [])
            if idx < len(node_inputs):
                link_id = node_inputs[idx].get("link")
            else:
                link_id = None
            if link_id is None:
                subs[inp["name"]] = "None"
                continue
            src_nid, src_slot, _dst_nid, _dst_slot = link_by_id[link_id]
            src_node = node_map[src_nid]
            src_defn = resolve_definition(src_node["type"], comp_defs, atomic_defs)
            src_def_id = src_defn.get("id", src_defn["name"].lower().replace(" ", "_"))
            src_output = src_defn["outputs"][src_slot]
            src_var = make_var_name(src_def_id, src_nid, src_output["name"])
            subs[inp["name"]] = src_var

        # Substitute template
        code = substitute_template(defn["code"], subs)

        title = node.get("title", defn.get("name", def_id))
        code_blocks.append(f"# --- Node {nid}: {title} ---")
        code_blocks.append(code)
        code_blocks.append("")

        for imp in defn.get("imports", []):
            all_imports.add(imp)

    header = '"""Auto-generated by Rapid Neural Designer compiler."""\n'
    import_block = "\n".join(sorted(all_imports))
    body = "\n".join(code_blocks)
    return f"{header}\n{import_block}\n\n{body}"


# ---------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------

def test_model(name, graph_path, comp_defs, atomic_defs):
    """Compile and execute one model graph. Returns True on success."""
    print(f"\n{'=' * 60}")
    print(f"  MODEL: {name.upper()}")
    print(f"{'=' * 60}")

    graph = load_graph(graph_path)
    print(f"  Nodes: {len(graph['nodes'])}, Links: {len(graph['links'])}")

    # Compile
    try:
        source = compile_graph(graph, comp_defs, atomic_defs)
    except Exception as e:
        print(f"  COMPILE ERROR: {e}")
        return False

    # Print generated source
    print(f"\n  --- Generated Source ({name}) ---")
    for i, line in enumerate(source.split("\n"), 1):
        print(f"    {i:3d} | {line}")
    print(f"  --- End Source ---\n")

    # Syntax check
    try:
        compile(source, f"{name}.py", "exec")
        print(f"  Syntax: OK")
    except SyntaxError as e:
        print(f"  SYNTAX ERROR: {e}")
        return False

    # Write to temp file and execute
    output_path = os.path.join(SCRIPT_DIR, f"{name}_compiled.py")
    with open(output_path, "w") as f:
        f.write(source)

    try:
        result = subprocess.run(
            [sys.executable, output_path],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.stdout.strip():
            print(f"  STDOUT: {result.stdout.strip()}")
        if result.stderr.strip():
            print(f"  STDERR: {result.stderr.strip()}")

        if result.returncode == 0:
            print(f"  RESULT: SUCCESS")
            return True
        else:
            print(f"  RESULT: FAILURE (return code {result.returncode})")
            return False

    except subprocess.TimeoutExpired:
        print(f"  RESULT: TIMEOUT (>300s)")
        return False
    finally:
        # Clean up compiled file
        try:
            os.unlink(output_path)
        except OSError:
            pass


def main():
    print("Loading component definitions...")
    comp_defs = load_component_defs()
    atomic_defs = load_atomic_defs()
    print(f"  Loaded {len(comp_defs)} molecular components, {len(atomic_defs)} atomic components")

    results = {}
    for name, path in MODEL_GRAPHS.items():
        results[name] = test_model(name, path, comp_defs, atomic_defs)

    # Summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:12s}: {status}")
        if not passed:
            all_pass = False

    print(f"{'=' * 60}")
    if all_pass:
        print("  All 5 models compiled and executed successfully!")
    else:
        print("  Some models failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
