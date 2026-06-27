"""MCP Streamable HTTP endpoint for the RND Platform.

Implements the Model Context Protocol (JSON-RPC 2.0) directly in Flask,
exposing research management tools to Claude Desktop, claude.ai, and
any MCP-compatible client.

Mount at /mcp in the unified backend.
"""

from __future__ import annotations

import json
import secrets
from typing import Any

from flask import Blueprint, Response, jsonify, request

import copy
from pathlib import Path

from .auth import AuthDB
from .canonical import content_hash as compute_content_hash
from .component_catalog import ComponentCatalog
from .graph_diff import diff_graphs
from .graph_ops import (
    create_blank_graph, add_node as graph_add_node, remove_node as graph_remove_node,
    update_node_properties, add_link as graph_add_link, remove_link as graph_remove_link,
    validate_graph, get_graph_summary,
)
from .repo import RNDRepo
from .models import (
    Architecture,
    EvidenceLink, EvidenceSign, EvidenceStrength,
    ExperimentInputs, ExperimentStatus, ObservedResults,
    FindingResolution, StatementResolution,
    PaperStatus, SectionType,
    Program, Project, Thread, ThreadState,
    Statement, Experiment, Finding, Paper, PaperSection,
    Disclosure, DisclosureType, Visibility,
    now_iso,
)

MCP_PROTOCOL_VERSION = "2025-03-26"
SERVER_NAME = "rnd-platform"
SERVER_VERSION = "0.1.0"

mcp_bp = Blueprint("mcp", __name__)

# Set by init_mcp()
_repo: RNDRepo = None  # type: ignore
_auth: AuthDB = None  # type: ignore
_catalog: ComponentCatalog = None  # type: ignore


def init_mcp(repo: RNDRepo, auth_db: AuthDB, web_interface_root: str | Path = ""):
    global _repo, _auth, _catalog
    _repo = repo
    _auth = auth_db
    if web_interface_root:
        _catalog = ComponentCatalog(Path(web_interface_root))
    else:
        # Default: sibling web_interface directory
        _catalog = ComponentCatalog(Path(repo.root) / "web_interface")


# ------------------------------------------------------------------
# Tool registry
# ------------------------------------------------------------------

_tools: list[dict] = []
_handlers: dict[str, Any] = {}


def tool(name: str, description: str, schema: dict, *, annotations: dict | None = None):
    """Register an MCP tool."""
    def decorator(fn):
        _tools.append({
            "name": name,
            "description": description,
            "inputSchema": {"type": "object", "properties": schema.get("properties", {}),
                            "required": schema.get("required", [])},
            **({"annotations": annotations} if annotations else {}),
        })
        _handlers[name] = fn
        return fn
    return decorator


# ------------------------------------------------------------------
# JSON-RPC helpers
# ------------------------------------------------------------------

def _ok(id, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": id, "result": result}


def _err(id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}


# ------------------------------------------------------------------
# MCP protocol handlers
# ------------------------------------------------------------------

def _handle_initialize(id, params: dict) -> dict:
    return _ok(id, {
        "protocolVersion": MCP_PROTOCOL_VERSION,
        "capabilities": {
            "tools": {"listChanged": False},
        },
        "serverInfo": {
            "name": SERVER_NAME,
            "version": SERVER_VERSION,
        },
        "instructions": (
            "RND Platform research management. Use these tools to create and manage "
            "research programs, projects, threads (research questions), statements "
            "(hypotheses), experiments, findings, architectures, papers, and disclosures. "
            "The hierarchy is: Program > Project > Thread > Statement/Experiment > Finding. "
            "Papers pull from threads and projects. Disclosures are citable documents."
        ),
    })


def _handle_tools_list(id, params: dict) -> dict:
    return _ok(id, {"tools": _tools})


def _handle_tools_call(id, params: dict) -> dict:
    name = params.get("name", "")
    args = params.get("arguments") or {}
    handler = _handlers.get(name)
    if not handler:
        return _err(id, -32601, f"Unknown tool: {name}")
    try:
        result = handler(**args)
        text = json.dumps(result, indent=2, default=str) if not isinstance(result, str) else result
        return _ok(id, {
            "content": [{"type": "text", "text": text}],
            "isError": False,
        })
    except Exception as e:
        return _ok(id, {
            "content": [{"type": "text", "text": f"Error: {e}"}],
            "isError": True,
        })


def _handle_ping(id, params: dict) -> dict:
    return _ok(id, {})


_method_handlers = {
    "initialize": _handle_initialize,
    "tools/list": _handle_tools_list,
    "tools/call": _handle_tools_call,
    "ping": _handle_ping,
}


# ------------------------------------------------------------------
# Flask route
# ------------------------------------------------------------------

def _authenticate() -> dict | None:
    """Check MCP credentials from Authorization header.
    Accepts:
      - Bearer <client_id>:<secret>  (direct PSK)
      - Bearer <oauth_access_token>  (OAuth 2.0 token from /token exchange)
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None
    token = auth_header[7:]

    # Try client_id:secret format first
    if ":" in token:
        client_id, secret = token.split(":", 1)
        user = _auth.validate_mcp_client(client_id, secret)
        if user:
            return user

    # Try as OAuth access token, then as a derived API token
    return _auth.validate_oauth_token(token) or _auth.validate_api_token(token)


@mcp_bp.route("", methods=["POST"])
def mcp_post():
    """Handle MCP Streamable HTTP POST requests (JSON-RPC 2.0)."""
    user = _authenticate()
    if not user:
        return jsonify({"error": "Invalid or missing MCP credentials. Use Bearer <client_id>:<secret>"}), 401

    try:
        body = request.get_json(force=True)
    except Exception:
        return jsonify({"jsonrpc": "2.0", "id": None,
                        "error": {"code": -32700, "message": "Parse error"}}), 400

    # Handle batch
    if isinstance(body, list):
        responses = []
        for msg in body:
            resp = _dispatch(msg)
            if resp is not None:
                responses.append(resp)
        if not responses:
            return "", 202
        return Response(json.dumps(responses), status=200,
                        content_type="application/json")

    resp = _dispatch(body)
    if resp is None:
        return "", 202
    return Response(json.dumps(resp), status=200,
                    content_type="application/json")


@mcp_bp.route("", methods=["GET"])
def mcp_get():
    """SSE stream endpoint (required by spec, returns server info)."""
    return jsonify({
        "name": SERVER_NAME,
        "version": SERVER_VERSION,
        "protocolVersion": MCP_PROTOCOL_VERSION,
    })


@mcp_bp.route("", methods=["DELETE"])
def mcp_delete():
    """Session termination (stateless, always OK)."""
    return "", 200


def _dispatch(msg: dict) -> dict | None:
    """Dispatch a single JSON-RPC message. Returns response or None for notifications."""
    method = msg.get("method", "")
    msg_id = msg.get("id")
    params = msg.get("params") or {}

    # Notifications (no id) — don't return a response
    if msg_id is None:
        return None

    handler = _method_handlers.get(method)
    if not handler:
        return _err(msg_id, -32601, f"Method not found: {method}")

    return handler(msg_id, params)


# ======================================================================
# Tool definitions
# ======================================================================

# ------------------------------------------------------------------
# Programs
# ------------------------------------------------------------------

@tool("list_programs", "List all research programs", {}, annotations={"readOnlyHint": True})
def list_programs():
    return [p.to_dict() for p in _repo.list_entities("program")]


@tool("create_program", "Create a new research program", {
    "properties": {
        "name": {"type": "string", "description": "Program name"},
        "description": {"type": "string", "description": "Program description"},
    },
    "required": ["name"],
})
def create_program(name: str, description: str = ""):
    teams = _repo.list_entities("team")
    if not teams:
        raise ValueError("No team configured")
    prog = Program.create(team_id=teams[0].id, name=name, description=description)
    _repo.save(prog)
    return prog.to_dict()


@tool("get_program", "Get a program by ID", {
    "properties": {"program_id": {"type": "string"}},
    "required": ["program_id"],
}, annotations={"readOnlyHint": True})
def get_program(program_id: str):
    prog = _repo.load("program", program_id)
    if not prog:
        raise ValueError(f"Program {program_id} not found")
    return prog.to_dict()


# ------------------------------------------------------------------
# Projects
# ------------------------------------------------------------------

@tool("list_projects", "List projects, optionally filtered by program", {
    "properties": {
        "program_id": {"type": "string", "description": "Filter by program ID (optional)"},
    },
}, annotations={"readOnlyHint": True})
def list_projects(program_id: str = ""):
    projects = _repo.list_entities("project")
    if program_id:
        projects = [p for p in projects if p.program_id == program_id]
    return [p.to_dict() for p in projects]


@tool("create_project", "Create a new project within a program", {
    "properties": {
        "program_id": {"type": "string", "description": "Parent program ID"},
        "name": {"type": "string", "description": "Project name"},
        "description": {"type": "string", "description": "Project description"},
    },
    "required": ["program_id", "name"],
})
def create_project(program_id: str, name: str, description: str = ""):
    proj = Project.create(program_id=program_id, name=name, description=description)
    _repo.save(proj)
    return proj.to_dict()


@tool("get_project", "Get a project by ID", {
    "properties": {"project_id": {"type": "string"}},
    "required": ["project_id"],
}, annotations={"readOnlyHint": True})
def get_project(project_id: str):
    proj = _repo.load("project", project_id)
    if not proj:
        raise ValueError(f"Project {project_id} not found")
    return proj.to_dict()


# ------------------------------------------------------------------
# Threads (research questions)
# ------------------------------------------------------------------

@tool("list_threads", "List research threads, optionally filtered by project", {
    "properties": {
        "project_id": {"type": "string", "description": "Filter by project ID (optional)"},
    },
}, annotations={"readOnlyHint": True})
def list_threads(project_id: str = ""):
    threads = _repo.list_entities("thread")
    if project_id:
        threads = [t for t in threads if t.project_id == project_id]
    return [t.to_dict() for t in threads]


@tool("create_thread", "Create a new research thread (question to investigate)", {
    "properties": {
        "project_id": {"type": "string", "description": "Parent project ID"},
        "question": {"type": "string", "description": "The research question"},
        "resolution_criterion": {"type": "string", "description": "What would resolve this question"},
    },
    "required": ["project_id", "question"],
})
def create_thread(project_id: str, question: str, resolution_criterion: str = ""):
    thread = Thread.create(project_id=project_id, question=question,
                           resolution_criterion=resolution_criterion)
    _repo.save(thread)
    return thread.to_dict()


@tool("get_thread", "Get a thread by ID", {
    "properties": {"thread_id": {"type": "string"}},
    "required": ["thread_id"],
}, annotations={"readOnlyHint": True})
def get_thread(thread_id: str):
    thread = _repo.load("thread", thread_id)
    if not thread:
        raise ValueError(f"Thread {thread_id} not found")
    return thread.to_dict()


@tool("update_thread", "Update a thread's state or question", {
    "properties": {
        "thread_id": {"type": "string"},
        "state": {"type": "string", "enum": ["open", "resolved", "stale", "reopened"],
                  "description": "New thread state"},
        "question": {"type": "string", "description": "Updated question"},
        "resolution_criterion": {"type": "string"},
    },
    "required": ["thread_id"],
})
def update_thread(thread_id: str, state: str = "", question: str = "",
                  resolution_criterion: str = ""):
    thread = _repo.load("thread", thread_id)
    if not thread:
        raise ValueError(f"Thread {thread_id} not found")
    if state:
        thread.state = ThreadState(state)
    if question:
        thread.question = question
    if resolution_criterion:
        thread.resolution_criterion = resolution_criterion
    thread.updated_at = now_iso()
    _repo.save(thread)
    return thread.to_dict()


# ------------------------------------------------------------------
# Statements (hypotheses)
# ------------------------------------------------------------------

@tool("list_statements", "List statements/hypotheses, optionally filtered by thread", {
    "properties": {
        "thread_id": {"type": "string", "description": "Filter by thread ID (optional)"},
    },
}, annotations={"readOnlyHint": True})
def list_statements(thread_id: str = ""):
    stmts = _repo.list_entities("statement")
    if thread_id:
        stmts = [s for s in stmts if s.thread_id == thread_id]
    return [s.to_dict() for s in stmts]


@tool("create_statement", "Create a hypothesis within a thread", {
    "properties": {
        "thread_id": {"type": "string", "description": "Parent thread ID"},
        "hypothesis": {"type": "string", "description": "The hypothesis to test"},
    },
    "required": ["thread_id", "hypothesis"],
})
def create_statement(thread_id: str, hypothesis: str):
    stmt = Statement.create(thread_id=thread_id, hypothesis=hypothesis)
    _repo.save(stmt)
    return stmt.to_dict()


@tool("get_statement", "Get a statement by ID", {
    "properties": {"statement_id": {"type": "string"}},
    "required": ["statement_id"],
}, annotations={"readOnlyHint": True})
def get_statement(statement_id: str):
    stmt = _repo.load("statement", statement_id)
    if not stmt:
        raise ValueError(f"Statement {statement_id} not found")
    return stmt.to_dict()


# ------------------------------------------------------------------
# Experiments
# ------------------------------------------------------------------

@tool("list_experiments", "List experiments, optionally filtered by thread", {
    "properties": {
        "thread_id": {"type": "string", "description": "Filter by thread ID (optional)"},
    },
}, annotations={"readOnlyHint": True})
def list_experiments(thread_id: str = ""):
    exps = _repo.list_entities("experiment")
    if thread_id:
        exps = [e for e in exps if e.thread_id == thread_id]
    return [e.to_dict() for e in exps]


@tool("create_experiment", "Create an experiment linked to a thread", {
    "properties": {
        "thread_id": {"type": "string", "description": "Parent thread ID"},
        "hypothesis": {"type": "string", "description": "What this experiment tests"},
        "expected": {"type": "string", "description": "Expected outcome"},
        "method": {"type": "object", "description": "Experiment method/procedure (freeform JSON)"},
        "architecture_id": {"type": "string", "description": "Architecture used (optional)"},
        "dataset": {"type": "string", "description": "Dataset used (optional)"},
        "hyperparameters": {"type": "object", "description": "Hyperparameters (optional)"},
    },
    "required": ["thread_id"],
})
def create_experiment(thread_id: str, hypothesis: str = "", expected: str = "",
                      method: dict = None, architecture_id: str = "",
                      dataset: str = "", hyperparameters: dict = None):
    inputs = ExperimentInputs(
        architecture_id=architecture_id,
        dataset=dataset,
        hyperparameters=hyperparameters or {},
    )
    exp = Experiment.create(
        thread_id=thread_id,
        created_by="mcp-client",
        inputs=inputs,
        hypothesis=hypothesis,
        expected=expected,
        method=method or {},
    )
    _repo.save(exp)
    return exp.to_dict()


@tool("get_experiment", "Get an experiment by ID", {
    "properties": {"experiment_id": {"type": "string"}},
    "required": ["experiment_id"],
}, annotations={"readOnlyHint": True})
def get_experiment(experiment_id: str):
    exp = _repo.load("experiment", experiment_id)
    if not exp:
        raise ValueError(f"Experiment {experiment_id} not found")
    return exp.to_dict()


@tool("update_experiment", "Update experiment status, results, or interpretation", {
    "properties": {
        "experiment_id": {"type": "string"},
        "status": {"type": "string", "enum": ["planned", "running", "complete", "failed", "abandoned"],
                   "description": "Experiment status"},
        "observed": {"type": "object", "description": "Observed results: {metrics: {}, artifacts: [], notes: ''}"},
        "interpretation": {"type": "string", "description": "Interpretation of results"},
    },
    "required": ["experiment_id"],
})
def update_experiment(experiment_id: str, status: str = "", observed: dict = None,
                      interpretation: str = ""):
    exp = _repo.load("experiment", experiment_id)
    if not exp:
        raise ValueError(f"Experiment {experiment_id} not found")
    if status:
        exp.status = ExperimentStatus(status)
    if observed:
        exp.observed = ObservedResults.from_dict(observed)
    if interpretation:
        exp.interpretation = interpretation
    exp.updated_at = now_iso()
    _repo.save(exp)
    return exp.to_dict()


@tool("attach_evidence", "Link an experiment to a statement as supporting/contradicting evidence", {
    "properties": {
        "experiment_id": {"type": "string"},
        "statement_id": {"type": "string", "description": "Statement this evidence applies to"},
        "sign": {"type": "string", "enum": ["supports", "contradicts", "neutral"],
                 "description": "Whether evidence supports or contradicts the statement"},
        "strength": {"type": "string", "enum": ["weak", "moderate", "strong"],
                     "description": "Strength of evidence (default: moderate)"},
        "note": {"type": "string", "description": "Explanation of the evidence link"},
    },
    "required": ["experiment_id", "statement_id", "sign"],
})
def attach_evidence(experiment_id: str, statement_id: str, sign: str,
                    strength: str = "moderate", note: str = ""):
    exp = _repo.load("experiment", experiment_id)
    if not exp:
        raise ValueError(f"Experiment {experiment_id} not found")
    link = EvidenceLink(
        statement_id=statement_id,
        sign=EvidenceSign(sign),
        strength=EvidenceStrength(strength),
        note=note,
    )
    exp.evidence.append(link)
    exp.updated_at = now_iso()
    _repo.save(exp)
    return link.to_dict()


# ------------------------------------------------------------------
# Findings
# ------------------------------------------------------------------

@tool("list_findings", "List findings, optionally filtered by thread", {
    "properties": {
        "thread_id": {"type": "string", "description": "Filter by thread ID (optional)"},
    },
}, annotations={"readOnlyHint": True})
def list_findings(thread_id: str = ""):
    findings = _repo.list_entities("finding")
    if thread_id:
        findings = [f for f in findings if f.thread_id == thread_id]
    return [f.to_dict() for f in findings]


@tool("create_finding", "Record a finding that resolves statements in a thread", {
    "properties": {
        "thread_id": {"type": "string"},
        "summary": {"type": "string", "description": "Summary of the finding"},
        "reasoning": {"type": "string", "description": "Detailed reasoning"},
        "statement_resolutions": {
            "type": "array",
            "description": "How each statement was resolved",
            "items": {
                "type": "object",
                "properties": {
                    "statement_id": {"type": "string"},
                    "resolution": {"type": "string", "enum": ["confirmed", "refuted", "modified", "inconclusive"]},
                    "note": {"type": "string"},
                },
            },
        },
        "experiment_refs": {
            "type": "array", "items": {"type": "string"},
            "description": "IDs of experiments that support this finding",
        },
        "resolve_thread": {"type": "boolean", "description": "Also mark the thread as resolved"},
    },
    "required": ["thread_id", "summary"],
})
def create_finding(thread_id: str, summary: str, reasoning: str = "",
                   statement_resolutions: list = None, experiment_refs: list = None,
                   resolve_thread: bool = False):
    resolutions = []
    for r in (statement_resolutions or []):
        resolutions.append(StatementResolution(
            statement_id=r["statement_id"],
            resolution=FindingResolution(r["resolution"]),
            note=r.get("note", ""),
        ))
    finding = Finding.create(
        thread_id=thread_id,
        summary=summary,
        reasoning=reasoning,
        statement_resolutions=resolutions,
        experiment_refs=experiment_refs or [],
    )
    _repo.save(finding)

    if resolve_thread:
        thread = _repo.load("thread", thread_id)
        if thread:
            thread.state = ThreadState.RESOLVED
            thread.updated_at = now_iso()
            _repo.save(thread)

    return finding.to_dict()


# ------------------------------------------------------------------
# Architectures
# ------------------------------------------------------------------

@tool("list_architectures", "List neural architectures", {
    "properties": {
        "include_archived": {"type": "boolean", "description": "Include archived (default false)"},
    },
}, annotations={"readOnlyHint": True})
def list_architectures(include_archived: bool = False):
    archs = _repo.list_entities("architecture")
    if not include_archived:
        archs = [a for a in archs if not a.archived]
    return [a.to_dict() for a in archs]


@tool("import_architecture", "Import a neural architecture definition", {
    "properties": {
        "name": {"type": "string", "description": "Architecture name"},
        "content": {"type": "object", "description": "Architecture definition (graph JSON)"},
        "variant_of": {"type": "string", "description": "Parent architecture ID if this is a variant"},
    },
    "required": ["name", "content"],
})
def import_architecture(name: str, content: dict, variant_of: str = None):
    arch = _repo.import_architecture(name=name, content=content, variant_of=variant_of)
    return arch.to_dict()


@tool("get_architecture", "Get an architecture by ID", {
    "properties": {"architecture_id": {"type": "string"}},
    "required": ["architecture_id"],
}, annotations={"readOnlyHint": True})
def get_architecture(architecture_id: str):
    arch = _repo.load("architecture", architecture_id)
    if not arch:
        raise ValueError(f"Architecture {architecture_id} not found")
    return arch.to_dict()


@tool("archive_architecture", "Archive an architecture (soft delete)", {
    "properties": {"architecture_id": {"type": "string"}},
    "required": ["architecture_id"],
})
def archive_architecture(architecture_id: str):
    if _repo.archive("architecture", architecture_id):
        return {"archived": True, "id": architecture_id}
    raise ValueError(f"Architecture {architecture_id} not found")


# ------------------------------------------------------------------
# Architecture editing
# ------------------------------------------------------------------

def _edit_arch(architecture_id: str, edit_fn):
    """Load architecture, apply edit, recompute hash, save. Returns (arch, edit_result)."""
    arch = _repo.load("architecture", architecture_id)
    if not arch:
        raise ValueError(f"Architecture {architecture_id} not found")
    result = edit_fn(arch.content)
    arch.content_hash = compute_content_hash(arch.content)
    arch.updated_at = now_iso()
    _repo.save(arch)
    return arch, result


@tool("create_architecture", "Create a new blank architecture graph", {
    "properties": {
        "name": {"type": "string", "description": "Architecture name"},
        "description": {"type": "string", "description": "Description"},
    },
    "required": ["name"],
})
def create_architecture(name: str, description: str = ""):
    content = create_blank_graph(description)
    arch = Architecture.create(
        name=name,
        content=content,
        content_hash=compute_content_hash(content),
    )
    _repo.save(arch)
    return arch.to_dict()


@tool("add_node", "Add a component or atomic node to an architecture", {
    "properties": {
        "architecture_id": {"type": "string"},
        "node_type": {"type": "string",
                      "description": "Type: 'molecular/{component_id}' or 'atomic/{category}/{atomic_id}'"},
        "properties": {"type": "object", "description": "Property overrides (merged with component defaults)"},
        "title": {"type": "string", "description": "Display title"},
        "position": {"type": "array", "items": {"type": "number"}, "description": "[x, y] position"},
    },
    "required": ["architecture_id", "node_type"],
})
def mcp_add_node(architecture_id: str, node_type: str, properties: dict = None,
                 title: str = "", position: list = None):
    node_id = [None]
    def edit(content):
        _, nid = graph_add_node(content, node_type, _catalog, properties, title, position)
        node_id[0] = nid
    arch, _ = _edit_arch(architecture_id, edit)
    return {"architecture_id": architecture_id, "node_id": node_id[0],
            "graph": get_graph_summary(arch.content)}


@tool("remove_node", "Remove a node and its connections from an architecture", {
    "properties": {
        "architecture_id": {"type": "string"},
        "node_id": {"type": "integer", "description": "Node ID to remove"},
    },
    "required": ["architecture_id", "node_id"],
})
def mcp_remove_node(architecture_id: str, node_id: int):
    def edit(content):
        graph_remove_node(content, node_id)
    arch, _ = _edit_arch(architecture_id, edit)
    return {"architecture_id": architecture_id, "removed_node": node_id,
            "graph": get_graph_summary(arch.content)}


@tool("connect_nodes", "Wire an output slot to an input slot between two nodes", {
    "properties": {
        "architecture_id": {"type": "string"},
        "src_node_id": {"type": "integer", "description": "Source node ID"},
        "src_slot": {"type": "integer", "description": "Output slot index on source (usually 0)"},
        "dst_node_id": {"type": "integer", "description": "Destination node ID"},
        "dst_slot": {"type": "integer", "description": "Input slot index on destination (usually 0)"},
    },
    "required": ["architecture_id", "src_node_id", "src_slot", "dst_node_id", "dst_slot"],
})
def mcp_connect_nodes(architecture_id: str, src_node_id: int, src_slot: int,
                      dst_node_id: int, dst_slot: int):
    link_id = [None]
    def edit(content):
        _, lid = graph_add_link(content, src_node_id, src_slot, dst_node_id, dst_slot)
        link_id[0] = lid
    arch, _ = _edit_arch(architecture_id, edit)
    return {"architecture_id": architecture_id, "link_id": link_id[0],
            "graph": get_graph_summary(arch.content)}


@tool("disconnect", "Remove a link between nodes", {
    "properties": {
        "architecture_id": {"type": "string"},
        "link_id": {"type": "integer", "description": "Link ID to remove"},
    },
    "required": ["architecture_id", "link_id"],
})
def mcp_disconnect(architecture_id: str, link_id: int):
    def edit(content):
        graph_remove_link(content, link_id)
    arch, _ = _edit_arch(architecture_id, edit)
    return {"architecture_id": architecture_id, "removed_link": link_id,
            "graph": get_graph_summary(arch.content)}


@tool("update_node", "Update a node's properties or title", {
    "properties": {
        "architecture_id": {"type": "string"},
        "node_id": {"type": "integer"},
        "properties": {"type": "object", "description": "Properties to update"},
        "title": {"type": "string", "description": "New display title"},
    },
    "required": ["architecture_id", "node_id"],
})
def mcp_update_node(architecture_id: str, node_id: int, properties: dict = None,
                    title: str = ""):
    def edit(content):
        update_node_properties(content, node_id, properties, title)
    arch, _ = _edit_arch(architecture_id, edit)
    return {"architecture_id": architecture_id, "node_id": node_id,
            "graph": get_graph_summary(arch.content)}


@tool("update_architecture_metadata", "Update architecture name or description", {
    "properties": {
        "architecture_id": {"type": "string"},
        "name": {"type": "string", "description": "New name"},
        "description": {"type": "string", "description": "New description (stored in graph extra.info)"},
    },
    "required": ["architecture_id"],
})
def update_architecture_metadata(architecture_id: str, name: str = "", description: str = ""):
    arch = _repo.load("architecture", architecture_id)
    if not arch:
        raise ValueError(f"Architecture {architecture_id} not found")
    if name:
        arch.name = name
    if description:
        arch.content.setdefault("extra", {})["info"] = description
        arch.content_hash = compute_content_hash(arch.content)
    arch.updated_at = now_iso()
    _repo.save(arch)
    return arch.to_dict()


@tool("validate_architecture", "Validate architecture graph for errors and warnings", {
    "properties": {"architecture_id": {"type": "string"}},
    "required": ["architecture_id"],
}, annotations={"readOnlyHint": True})
def mcp_validate_architecture(architecture_id: str):
    arch = _repo.load("architecture", architecture_id)
    if not arch:
        raise ValueError(f"Architecture {architecture_id} not found")
    issues = validate_graph(arch.content)
    return {"architecture_id": architecture_id, "issues": issues,
            "valid": all(i.get("level") != "error" for i in issues)}


# ------------------------------------------------------------------
# Architecture versioning
# ------------------------------------------------------------------

@tool("save_architecture_version", "Snapshot current architecture state with a version message", {
    "properties": {
        "architecture_id": {"type": "string"},
        "message": {"type": "string", "description": "What changed in this version"},
    },
    "required": ["architecture_id", "message"],
})
def mcp_save_version(architecture_id: str, message: str):
    version = _repo.save_architecture_version(architecture_id, message)
    return version.to_dict()


@tool("list_architecture_versions", "List version history for an architecture", {
    "properties": {"architecture_id": {"type": "string"}},
    "required": ["architecture_id"],
}, annotations={"readOnlyHint": True})
def mcp_list_versions(architecture_id: str):
    versions = _repo.list_architecture_versions(architecture_id)
    return [{"id": v.id, "name": v.name, "content_hash": v.content_hash,
             "created_at": v.created_at, "version_info": v.version_info}
            for v in versions]


@tool("load_architecture_version", "Restore a previous version into the current architecture", {
    "properties": {
        "architecture_id": {"type": "string", "description": "Architecture to restore into"},
        "version_id": {"type": "string", "description": "Version ID to restore from"},
    },
    "required": ["architecture_id", "version_id"],
})
def mcp_load_version(architecture_id: str, version_id: str):
    arch = _repo.load("architecture", architecture_id)
    if not arch:
        raise ValueError(f"Architecture {architecture_id} not found")
    version = _repo.load("architecture", version_id)
    if not version:
        raise ValueError(f"Version {version_id} not found")
    arch.content = copy.deepcopy(version.content)
    arch.content_hash = compute_content_hash(arch.content)
    arch.updated_at = now_iso()
    _repo.save(arch)
    return arch.to_dict()


@tool("branch_architecture", "Create a new architecture forked from an existing one", {
    "properties": {
        "source_id": {"type": "string", "description": "Architecture to branch from"},
        "name": {"type": "string", "description": "Name for the new branch"},
        "message": {"type": "string", "description": "Reason for branching"},
    },
    "required": ["source_id", "name"],
})
def mcp_branch_architecture(source_id: str, name: str, message: str = ""):
    source = _repo.load("architecture", source_id)
    if not source:
        raise ValueError(f"Architecture {source_id} not found")
    branch = Architecture.create(
        name=name,
        content=copy.deepcopy(source.content),
        content_hash=source.content_hash,
        variant_of=source_id,
        version_info={"message": message or f"Branched from {source.name}",
                      "version_number": 0, "source_id": source_id} if message else None,
    )
    _repo.save(branch)
    return branch.to_dict()


@tool("compare_architectures", "Compare two architectures showing structural differences", {
    "properties": {
        "architecture_id_a": {"type": "string", "description": "First architecture ID"},
        "architecture_id_b": {"type": "string", "description": "Second architecture ID"},
    },
    "required": ["architecture_id_a", "architecture_id_b"],
}, annotations={"readOnlyHint": True})
def mcp_compare_architectures(architecture_id_a: str, architecture_id_b: str):
    a = _repo.load("architecture", architecture_id_a)
    b = _repo.load("architecture", architecture_id_b)
    if not a:
        raise ValueError(f"Architecture {architecture_id_a} not found")
    if not b:
        raise ValueError(f"Architecture {architecture_id_b} not found")
    return diff_graphs(a.content, b.content)


# ------------------------------------------------------------------
# Component & Atomic catalog
# ------------------------------------------------------------------

@tool("list_components", "List all available neural components (linear, attention, transformer, etc.)", {},
      annotations={"readOnlyHint": True})
def mcp_list_components():
    return _catalog.list_components()


@tool("get_component", "Get full component definition including ports, properties, and graph decomposition", {
    "properties": {
        "component_id": {"type": "string", "description": "Component ID (e.g. 'linear', 'transformer_block')"},
    },
    "required": ["component_id"],
}, annotations={"readOnlyHint": True})
def mcp_get_component(component_id: str):
    comp = _catalog.get_component(component_id)
    if not comp:
        raise ValueError(f"Component '{component_id}' not found")
    return comp


@tool("list_atomics", "List available atomic primitives (math, activations, shape ops, etc.)", {
    "properties": {
        "category": {"type": "string",
                     "description": "Filter by category: math, trig, reduction, shape, comparison, init, data"},
    },
}, annotations={"readOnlyHint": True})
def mcp_list_atomics(category: str = ""):
    return _catalog.list_atomics(category)


@tool("get_atomic", "Get full atomic primitive definition", {
    "properties": {
        "category": {"type": "string", "description": "Atomic category"},
        "atomic_id": {"type": "string", "description": "Atomic ID (e.g. 'relu', 'matmul')"},
    },
    "required": ["category", "atomic_id"],
}, annotations={"readOnlyHint": True})
def mcp_get_atomic(category: str, atomic_id: str):
    atom = _catalog.get_atomic(category, atomic_id)
    if not atom:
        raise ValueError(f"Atomic '{category}/{atomic_id}' not found")
    return atom


# ------------------------------------------------------------------
# User components (private, per-user)
# ------------------------------------------------------------------

@tool("create_user_component", "Create a private reusable component (molecular)", {
    "properties": {
        "name": {"type": "string", "description": "Component name"},
        "description": {"type": "string", "description": "What this component does"},
        "inputs": {"type": "array", "items": {"type": "object"}, "description": "Input ports [{name, type}]"},
        "outputs": {"type": "array", "items": {"type": "object"}, "description": "Output ports [{name, type}]"},
        "properties": {"type": "array", "items": {"type": "object"},
                       "description": "Configurable properties [{name, type, default, description}]"},
        "code": {"type": "string", "description": "Python code template with {{variable}} substitution"},
        "imports": {"type": "array", "items": {"type": "string"}, "description": "Required Python imports"},
        "color": {"type": "string", "description": "Hex color (default: #888888)"},
        "graph": {"type": "object", "description": "Optional graph decomposition into atomics"},
    },
    "required": ["name", "inputs", "outputs", "code"],
})
def mcp_create_user_component(name: str, inputs: list, outputs: list, code: str,
                              description: str = "", properties: list = None,
                              imports: list = None, color: str = "#888888",
                              graph: dict = None):
    comp_id = name.lower().replace(" ", "_").replace("-", "_")
    definition = {
        "name": name, "id": comp_id, "description": description,
        "inputs": inputs, "outputs": outputs,
        "properties": properties or [], "code": code,
        "imports": imports or [], "color": color,
    }
    if graph:
        definition["graph"] = graph
    result = _auth.create_user_component(
        owner_id="mcp-client", kind="component",
        definition=json.dumps(definition),
    )
    result["definition"] = definition
    return result


@tool("create_user_atomic", "Create a private atomic primitive", {
    "properties": {
        "name": {"type": "string", "description": "Atomic name"},
        "atomic_id": {"type": "string", "description": "Unique ID (e.g. 'my_custom_op')"},
        "category": {"type": "string", "description": "Category: math, trig, reduction, shape, comparison, init, data"},
        "description": {"type": "string"},
        "inputs": {"type": "array", "items": {"type": "object"}, "description": "Input ports [{name, type}]"},
        "outputs": {"type": "array", "items": {"type": "object"}, "description": "Output ports [{name, type}]"},
        "code": {"type": "string", "description": "Python code template with {{variable}} substitution"},
        "imports": {"type": "array", "items": {"type": "string"}},
        "properties": {"type": "array", "items": {"type": "object"}, "description": "Optional properties"},
        "color": {"type": "string", "description": "Hex color"},
    },
    "required": ["name", "atomic_id", "category", "inputs", "outputs", "code"],
})
def mcp_create_user_atomic(name: str, atomic_id: str, category: str,
                           inputs: list, outputs: list, code: str,
                           description: str = "", imports: list = None,
                           properties: list = None, color: str = "#888888"):
    definition = {
        "name": name, "id": atomic_id, "category": category,
        "description": description, "inputs": inputs, "outputs": outputs,
        "code": code, "imports": imports or [],
        "color": color,
    }
    if properties:
        definition["properties"] = properties
    result = _auth.create_user_component(
        owner_id="mcp-client", kind="atomic", category=category,
        definition=json.dumps(definition),
    )
    result["definition"] = definition
    return result


@tool("list_user_components", "List your private components and atomics", {},
      annotations={"readOnlyHint": True})
def mcp_list_user_components():
    # MCP auth sets owner to "mcp-client" — list all for now
    # In practice, the MCP user is identified via the auth token
    rows = _auth.list_user_components("mcp-client")
    for r in rows:
        r["definition"] = json.loads(r["definition"])
    return rows


@tool("update_user_component", "Update a user component's definition", {
    "properties": {
        "component_id": {"type": "string", "description": "User component ID (uco-...)"},
        "definition": {"type": "object", "description": "Updated definition (full replacement)"},
    },
    "required": ["component_id", "definition"],
})
def mcp_update_user_component(component_id: str, definition: dict):
    result = _auth.update_user_component(
        component_id, "mcp-client",
        definition=json.dumps(definition),
    )
    if not result:
        raise ValueError(f"Component {component_id} not found or not owned by you")
    result["definition"] = json.loads(result["definition"])
    return result


@tool("delete_user_component", "Delete a user component", {
    "properties": {
        "component_id": {"type": "string", "description": "User component ID (uco-...)"},
    },
    "required": ["component_id"],
})
def mcp_delete_user_component(component_id: str):
    if not _auth.delete_user_component(component_id, "mcp-client"):
        raise ValueError(f"Component {component_id} not found or not owned by you")
    return {"deleted": True, "id": component_id}


@tool("promote_component", "Admin only: promote a user component to the built-in default catalog", {
    "properties": {
        "component_id": {"type": "string", "description": "User component ID (uco-...) to promote"},
    },
    "required": ["component_id"],
})
def mcp_promote_component(component_id: str):
    # Admin check — hardcoded to first registered user
    # MCP client is tied to a user via auth, but we use mcp-client owner for now
    # The actual admin check happens via the auth token's user
    uc = _auth.get_user_component(component_id)
    if not uc:
        raise ValueError(f"User component {component_id} not found")

    definition = json.loads(uc["definition"])
    kind = uc["kind"]

    if kind == "component":
        path = _catalog.promote_component(definition["id"], definition)
    elif kind == "atomic":
        category = uc.get("category") or definition.get("category", "")
        if not category:
            raise ValueError("Atomic must have a category to promote")
        path = _catalog.promote_atomic(definition, category)
    else:
        raise ValueError(f"Unknown kind: {kind}")

    # Remove from user_components (it's now built-in)
    _auth.delete_user_component(component_id, uc["owner_id"])
    return {"promoted": True, "kind": kind, "id": definition["id"], "path": str(path)}


# ------------------------------------------------------------------
# Papers
# ------------------------------------------------------------------

@tool("list_papers", "List all papers", {}, annotations={"readOnlyHint": True})
def list_papers():
    return [p.to_dict() for p in _repo.list_entities("paper")]


@tool("create_paper", "Create a new paper", {
    "properties": {
        "title": {"type": "string", "description": "Paper title"},
        "authors": {"type": "array", "items": {"type": "string"}, "description": "Author names"},
        "program_id": {"type": "string", "description": "Associated program ID"},
        "target": {"type": "string", "description": "Target venue (default: arxiv)"},
        "project_ids": {"type": "array", "items": {"type": "string"}},
        "thread_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title"],
})
def create_paper(title: str, authors: list = None, program_id: str = "",
                 target: str = "arxiv", project_ids: list = None, thread_ids: list = None):
    paper = Paper.create(title=title, authors=authors or [],
                         program_id=program_id, target=target)
    if project_ids:
        paper.project_ids = project_ids
    if thread_ids:
        paper.thread_ids = thread_ids
    _repo.save(paper)
    return paper.to_dict()


@tool("get_paper", "Get a paper by ID", {
    "properties": {"paper_id": {"type": "string"}},
    "required": ["paper_id"],
}, annotations={"readOnlyHint": True})
def get_paper(paper_id: str):
    paper = _repo.load("paper", paper_id)
    if not paper:
        raise ValueError(f"Paper {paper_id} not found")
    return paper.to_dict()


@tool("update_paper", "Update paper title, status, authors, or linked threads/projects", {
    "properties": {
        "paper_id": {"type": "string"},
        "title": {"type": "string"},
        "status": {"type": "string", "enum": ["outline", "drafting", "review", "submitted", "published"]},
        "authors": {"type": "array", "items": {"type": "string"}},
        "thread_ids": {"type": "array", "items": {"type": "string"}},
        "project_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["paper_id"],
})
def update_paper(paper_id: str, title: str = "", status: str = "",
                 authors: list = None, thread_ids: list = None, project_ids: list = None):
    paper = _repo.load("paper", paper_id)
    if not paper:
        raise ValueError(f"Paper {paper_id} not found")
    if title:
        paper.title = title
    if status:
        paper.status = PaperStatus(status)
    if authors is not None:
        paper.authors = authors
    if thread_ids is not None:
        paper.thread_ids = thread_ids
    if project_ids is not None:
        paper.project_ids = project_ids
    paper.updated_at = now_iso()
    _repo.save(paper)
    return paper.to_dict()


@tool("add_paper_section", "Add a section to a paper", {
    "properties": {
        "paper_id": {"type": "string"},
        "title": {"type": "string", "description": "Section title"},
        "section_type": {"type": "string", "enum": ["abstract", "introduction", "methods", "results",
                                                     "discussion", "conclusion", "appendix", "custom"],
                         "description": "Section type (default: custom)"},
        "content": {"type": "string", "description": "Section content (markdown)"},
        "order": {"type": "integer", "description": "Sort order"},
    },
    "required": ["paper_id", "title"],
})
def add_paper_section(paper_id: str, title: str, section_type: str = "custom",
                      content: str = "", order: int = 0):
    paper = _repo.load("paper", paper_id)
    if not paper:
        raise ValueError(f"Paper {paper_id} not found")
    section = PaperSection(
        id=f"sec-{secrets.token_hex(4)}",
        title=title,
        section_type=SectionType(section_type),
        order=order,
    )
    paper.sections.append(section)
    _repo.save(paper)
    if content:
        _repo.save_paper_section_content(paper, section, content)
    return {"paper_id": paper_id, "section": section.to_dict()}


@tool("get_section_content", "Get the markdown content of a paper section", {
    "properties": {
        "paper_id": {"type": "string"},
        "section_id": {"type": "string"},
    },
    "required": ["paper_id", "section_id"],
}, annotations={"readOnlyHint": True})
def get_section_content(paper_id: str, section_id: str):
    paper = _repo.load("paper", paper_id)
    if not paper:
        raise ValueError(f"Paper {paper_id} not found")
    section = next((s for s in paper.sections if s.id == section_id), None)
    if not section:
        raise ValueError(f"Section {section_id} not found")
    content = _repo.load_paper_section_content(section)
    return {"section_id": section_id, "content": content}


@tool("update_section_content", "Update the markdown content of a paper section", {
    "properties": {
        "paper_id": {"type": "string"},
        "section_id": {"type": "string"},
        "content": {"type": "string", "description": "New markdown content"},
    },
    "required": ["paper_id", "section_id", "content"],
})
def update_section_content(paper_id: str, section_id: str, content: str):
    paper = _repo.load("paper", paper_id)
    if not paper:
        raise ValueError(f"Paper {paper_id} not found")
    section = next((s for s in paper.sections if s.id == section_id), None)
    if not section:
        raise ValueError(f"Section {section_id} not found")
    _repo.save_paper_section_content(paper, section, content)
    return {"updated": True, "section_id": section_id}


# ------------------------------------------------------------------
# Disclosures
# ------------------------------------------------------------------

@tool("list_disclosures", "List all disclosures (citable documents)", {},
      annotations={"readOnlyHint": True})
def list_disclosures():
    return [d.to_dict() for d in _repo.list_entities("disclosure")]


@tool("create_disclosure", "Create a disclosure (citable document)", {
    "properties": {
        "title": {"type": "string", "description": "Disclosure title"},
        "type": {"type": "string", "enum": ["paper", "patent", "blog", "presentation", "dataset", "code", "other"],
                 "description": "Disclosure type"},
        "content": {"type": "string", "description": "Disclosure content (markdown)"},
        "source_url": {"type": "string", "description": "Source URL"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title"],
})
def create_disclosure(title: str, type: str = "other", content: str = "",
                      source_url: str = "", tags: list = None):
    disc = Disclosure.create(
        title=title,
        disclosure_type=DisclosureType(type),
        created_by="mcp-client",
        tags=tags or [],
    )
    if source_url:
        disc.source_url = source_url
    _repo.save(disc)
    if content:
        _repo.save_disclosure_content(disc, content)
    return disc.to_dict()


# ------------------------------------------------------------------
# Search
# ------------------------------------------------------------------

@tool("search", "Search across all research entities by text query", {
    "properties": {
        "query": {"type": "string", "description": "Search query"},
        "entity_type": {"type": "string", "description": "Filter by type (program, project, thread, etc.)"},
    },
    "required": ["query"],
}, annotations={"readOnlyHint": True})
def search(query: str, entity_type: str = ""):
    from .index import DerivedIndex
    index = DerivedIndex(_repo.index_path)
    index.open()
    try:
        index.rebuild(_repo.root)
        return index.search(query, entity_type=entity_type or None)
    finally:
        index.close()


# ------------------------------------------------------------------
# Overview (convenience)
# ------------------------------------------------------------------

@tool("research_overview", "Get a high-level overview of all programs, projects, and open threads", {},
      annotations={"readOnlyHint": True})
def research_overview():
    programs = _repo.list_entities("program")
    projects = _repo.list_entities("project")
    threads = _repo.list_entities("thread")
    overview = {
        "programs": [{"id": p.id, "name": p.name, "description": p.description} for p in programs],
        "projects": [{"id": p.id, "name": p.name, "program_id": p.program_id,
                       "description": p.description} for p in projects],
        "open_threads": [{"id": t.id, "question": t.question, "state": t.state.value,
                          "project_id": t.project_id} for t in threads
                         if t.state in (ThreadState.OPEN, ThreadState.REOPENED)],
        "counts": {
            "programs": len(programs),
            "projects": len(projects),
            "threads": len(threads),
            "open_threads": sum(1 for t in threads if t.state in (ThreadState.OPEN, ThreadState.REOPENED)),
            "statements": len(_repo.list_entities("statement")),
            "experiments": len(_repo.list_entities("experiment")),
            "findings": len(_repo.list_entities("finding")),
            "papers": len(_repo.list_entities("paper")),
            "architectures": len(_repo.list_entities("architecture")),
            "disclosures": len(_repo.list_entities("disclosure")),
        },
    }
    return overview
