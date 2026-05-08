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

from .repo import RNDRepo
from .models import (
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


def init_mcp(repo: RNDRepo):
    global _repo
    _repo = repo


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

@mcp_bp.route("", methods=["POST"])
def mcp_post():
    """Handle MCP Streamable HTTP POST requests (JSON-RPC 2.0)."""
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
