"""Pure graph manipulation functions for RND architectures.

All functions take a graph dict, perform an operation, and return the modified
graph. No side effects, no I/O — keeps MCP endpoint thin and logic testable.
"""

from __future__ import annotations

import copy
from typing import Any

from .component_catalog import ComponentCatalog


def create_blank_graph(info: str = "") -> dict:
    """Return a new empty graph dict."""
    return {
        "last_node_id": 0,
        "last_link_id": 0,
        "nodes": [],
        "links": [],
        "groups": [],
        "config": {},
        "extra": {"info": info},
        "version": 0.4,
    }


def add_node(graph: dict, node_type: str, catalog: ComponentCatalog,
             properties: dict | None = None, title: str = "",
             pos: list | None = None) -> tuple[dict, int]:
    """Add a node to the graph. Returns (graph, new_node_id).

    node_type: 'molecular/{id}' or 'atomic/{category}/{id}'
    """
    defn = catalog.get_node_definition(node_type)
    if not defn:
        raise ValueError(f"Unknown node type: {node_type}")

    graph["last_node_id"] += 1
    node_id = graph["last_node_id"]

    # Build input/output ports from definition
    inputs = [{"name": p["name"], "type": p["type"], "link": None}
              for p in defn.get("inputs", [])]
    outputs = [{"name": p["name"], "type": p["type"], "links": []}
               for p in defn.get("outputs", [])]

    # Build properties from defaults + overrides
    node_props = {}
    widget_values = []
    for prop_def in defn.get("properties", []):
        val = (properties or {}).get(prop_def["name"], prop_def.get("default"))
        node_props[prop_def["name"]] = val
        widget_values.append(val)

    # Auto-position if not specified
    if pos is None:
        if graph["nodes"]:
            max_y = max(n.get("pos", [0, 0])[1] for n in graph["nodes"])
            pos = [100, max_y + 180]
        else:
            pos = [100, 100]

    node = {
        "id": node_id,
        "type": node_type,
        "pos": pos,
        "size": [220, 130],
        "flags": {},
        "order": len(graph["nodes"]),
        "mode": 0,
        "inputs": inputs,
        "outputs": outputs,
        "title": title or defn.get("name", node_type),
        "properties": node_props,
        "widgets_values": widget_values,
    }
    graph["nodes"].append(node)
    return graph, node_id


def remove_node(graph: dict, node_id: int) -> dict:
    """Remove a node and all connected links."""
    # Find and remove links connected to this node
    links_to_remove = [l[0] for l in graph["links"]
                       if l[1] == node_id or l[3] == node_id]
    for link_id in links_to_remove:
        graph = remove_link(graph, link_id)

    graph["nodes"] = [n for n in graph["nodes"] if n["id"] != node_id]
    return graph


def update_node_properties(graph: dict, node_id: int,
                           properties: dict | None = None,
                           title: str = "") -> dict:
    """Update properties and/or title on an existing node."""
    node = _find_node(graph, node_id)
    if properties:
        node["properties"].update(properties)
        # Rebuild widgets_values from properties
        node["widgets_values"] = list(node["properties"].values())
    if title:
        node["title"] = title
    return graph


def add_link(graph: dict, src_node_id: int, src_slot: int,
             dst_node_id: int, dst_slot: int) -> tuple[dict, int]:
    """Wire an output slot to an input slot. Returns (graph, new_link_id)."""
    src_node = _find_node(graph, src_node_id)
    dst_node = _find_node(graph, dst_node_id)

    if src_slot >= len(src_node["outputs"]):
        raise ValueError(f"Node {src_node_id} has no output slot {src_slot} "
                         f"(has {len(src_node['outputs'])})")
    if dst_slot >= len(dst_node["inputs"]):
        raise ValueError(f"Node {dst_node_id} has no input slot {dst_slot} "
                         f"(has {len(dst_node['inputs'])})")

    src_output = src_node["outputs"][src_slot]
    dst_input = dst_node["inputs"][dst_slot]

    # Check if input slot is already connected
    if dst_input.get("link") is not None:
        raise ValueError(f"Input slot {dst_slot} on node {dst_node_id} is already connected "
                         f"(link {dst_input['link']}). Disconnect it first.")

    # Type check
    src_type = src_output.get("type", "tensor")
    dst_type = dst_input.get("type", "tensor")
    if src_type != dst_type:
        raise ValueError(f"Type mismatch: output is '{src_type}', input is '{dst_type}'")

    graph["last_link_id"] += 1
    link_id = graph["last_link_id"]

    # Add link tuple
    graph["links"].append([link_id, src_node_id, src_slot, dst_node_id, dst_slot, src_type])

    # Update port bookkeeping
    src_output["links"].append(link_id)
    dst_input["link"] = link_id

    return graph, link_id


def remove_link(graph: dict, link_id: int) -> dict:
    """Remove a link by ID and clean up port references."""
    link = None
    for l in graph["links"]:
        if l[0] == link_id:
            link = l
            break
    if not link:
        raise ValueError(f"Link {link_id} not found")

    _, src_node_id, src_slot, dst_node_id, dst_slot, _ = link

    # Clean up source output
    src_node = _find_node(graph, src_node_id)
    if src_slot < len(src_node["outputs"]):
        links_list = src_node["outputs"][src_slot].get("links", [])
        if link_id in links_list:
            links_list.remove(link_id)

    # Clean up destination input
    dst_node = _find_node(graph, dst_node_id)
    if dst_slot < len(dst_node["inputs"]):
        if dst_node["inputs"][dst_slot].get("link") == link_id:
            dst_node["inputs"][dst_slot]["link"] = None

    graph["links"] = [l for l in graph["links"] if l[0] != link_id]
    return graph


def validate_graph(graph: dict) -> list[dict]:
    """Return validation issues: disconnected inputs, type mismatches, etc."""
    issues = []
    node_ids = {n["id"] for n in graph["nodes"]}

    for node in graph["nodes"]:
        # Check for disconnected required inputs
        for i, inp in enumerate(node.get("inputs", [])):
            if inp.get("link") is None:
                issues.append({
                    "level": "warning",
                    "node_id": node["id"],
                    "message": f"Input '{inp['name']}' on '{node['title']}' is not connected",
                })

    # Check links reference valid nodes
    for link in graph["links"]:
        link_id, src_id, src_slot, dst_id, dst_slot, _ = link
        if src_id not in node_ids:
            issues.append({"level": "error", "link_id": link_id,
                           "message": f"Link {link_id} references missing source node {src_id}"})
        if dst_id not in node_ids:
            issues.append({"level": "error", "link_id": link_id,
                           "message": f"Link {link_id} references missing destination node {dst_id}"})

    if not graph["nodes"]:
        issues.append({"level": "info", "message": "Graph is empty"})

    return issues


def get_graph_summary(graph: dict) -> dict:
    """Return a concise summary of the graph structure."""
    return {
        "node_count": len(graph["nodes"]),
        "link_count": len(graph["links"]),
        "nodes": [
            {"id": n["id"], "type": n["type"], "title": n.get("title", ""),
             "inputs": len(n.get("inputs", [])), "outputs": len(n.get("outputs", [])),
             "properties": n.get("properties", {})}
            for n in graph["nodes"]
        ],
        "links": [
            {"id": l[0], "from": f"node {l[1]} slot {l[2]}",
             "to": f"node {l[3]} slot {l[4]}", "type": l[5]}
            for l in graph["links"]
        ],
        "info": graph.get("extra", {}).get("info", ""),
    }


# ---- Internal helpers ----

def _find_node(graph: dict, node_id: int) -> dict:
    for n in graph["nodes"]:
        if n["id"] == node_id:
            return n
    raise ValueError(f"Node {node_id} not found")
