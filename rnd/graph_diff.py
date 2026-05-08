"""Structural diff between two RND architecture graphs."""

from __future__ import annotations

from typing import Any


def diff_graphs(graph_a: dict, graph_b: dict) -> dict:
    """Compute a structural diff between two graph dicts.

    Returns:
        {
            "nodes_added": [...],       # nodes in B not in A
            "nodes_removed": [...],     # nodes in A not in B
            "nodes_modified": [...],    # nodes in both but different
            "links_added": [...],
            "links_removed": [...],
            "metadata_changes": {...},
            "summary": "..."
        }
    """
    nodes_a = {n["id"]: n for n in graph_a.get("nodes", [])}
    nodes_b = {n["id"]: n for n in graph_b.get("nodes", [])}

    ids_a = set(nodes_a.keys())
    ids_b = set(nodes_b.keys())

    nodes_added = [_node_summary(nodes_b[nid]) for nid in (ids_b - ids_a)]
    nodes_removed = [_node_summary(nodes_a[nid]) for nid in (ids_a - ids_b)]

    nodes_modified = []
    for nid in ids_a & ids_b:
        changes = _diff_node(nodes_a[nid], nodes_b[nid])
        if changes:
            nodes_modified.append({"id": nid, "title": nodes_b[nid].get("title", ""), **changes})

    # Links: compare as sets of (src, src_slot, dst, dst_slot)
    links_a = {(l[1], l[2], l[3], l[4]): l for l in graph_a.get("links", [])}
    links_b = {(l[1], l[2], l[3], l[4]): l for l in graph_b.get("links", [])}

    keys_a = set(links_a.keys())
    keys_b = set(links_b.keys())

    links_added = [_link_summary(links_b[k]) for k in (keys_b - keys_a)]
    links_removed = [_link_summary(links_a[k]) for k in (keys_a - keys_b)]

    # Metadata
    metadata_changes = {}
    info_a = graph_a.get("extra", {}).get("info", "")
    info_b = graph_b.get("extra", {}).get("info", "")
    if info_a != info_b:
        metadata_changes["info"] = {"from": info_a, "to": info_b}

    # Summary
    parts = []
    if nodes_added:
        parts.append(f"{len(nodes_added)} node(s) added")
    if nodes_removed:
        parts.append(f"{len(nodes_removed)} node(s) removed")
    if nodes_modified:
        parts.append(f"{len(nodes_modified)} node(s) modified")
    if links_added:
        parts.append(f"{len(links_added)} link(s) added")
    if links_removed:
        parts.append(f"{len(links_removed)} link(s) removed")
    summary = ", ".join(parts) if parts else "No structural changes"

    return {
        "nodes_added": nodes_added,
        "nodes_removed": nodes_removed,
        "nodes_modified": nodes_modified,
        "links_added": links_added,
        "links_removed": links_removed,
        "metadata_changes": metadata_changes,
        "summary": summary,
    }


def _node_summary(node: dict) -> dict:
    return {
        "id": node["id"],
        "type": node.get("type", ""),
        "title": node.get("title", ""),
        "properties": node.get("properties", {}),
    }


def _link_summary(link: list) -> dict:
    return {
        "id": link[0],
        "from_node": link[1], "from_slot": link[2],
        "to_node": link[3], "to_slot": link[4],
        "type": link[5] if len(link) > 5 else "tensor",
    }


def _diff_node(a: dict, b: dict) -> dict | None:
    """Compare two versions of the same node. Returns changes or None."""
    changes = {}

    if a.get("type") != b.get("type"):
        changes["type"] = {"from": a.get("type"), "to": b.get("type")}

    if a.get("title") != b.get("title"):
        changes["title"] = {"from": a.get("title"), "to": b.get("title")}

    props_a = a.get("properties", {})
    props_b = b.get("properties", {})
    if props_a != props_b:
        prop_changes = {}
        all_keys = set(list(props_a.keys()) + list(props_b.keys()))
        for k in all_keys:
            va = props_a.get(k)
            vb = props_b.get(k)
            if va != vb:
                prop_changes[k] = {"from": va, "to": vb}
        if prop_changes:
            changes["properties"] = prop_changes

    return changes if changes else None
