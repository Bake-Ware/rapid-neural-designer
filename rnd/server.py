"""REST API server for RND Platform (SDD §9.1).

Resource-oriented, standard HTTP verbs, JSON request/response.
Runs as a separate process from the editor backend.

Usage: python -m rnd.server [--repo .] [--port 5001]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from flask import Flask, Request, jsonify, request
from flask_cors import CORS

from .canonical import canonicalize, content_hash
from .index import DerivedIndex
from .models import (
    Architecture,
    Artifact,
    ArtifactCategory,
    Disclosure,
    DisclosureType,
    EvidenceLink,
    EvidenceSign,
    EvidenceStrength,
    Experiment,
    ExperimentInputs,
    ExperimentStatus,
    Finding,
    FindingResolution,
    Paper,
    PaperStatus,
    Program,
    Project,
    Statement,
    StatementResolution,
    Team,
    Thread,
    ThreadState,
    Visibility,
)
from .repo import RNDRepo

app = Flask(__name__)
CORS(app)

# These are set at startup
repo: RNDRepo = None  # type: ignore
index: DerivedIndex = None  # type: ignore


def error(msg: str, status: int = 400):
    return jsonify({"error": msg}), status


def ensure_index():
    """Reopen and rebuild index if needed."""
    if index.conn is None:
        index.open()


# ------------------------------------------------------------------
# Health
# ------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": "0.1.0"})


# ------------------------------------------------------------------
# Team
# ------------------------------------------------------------------

@app.route("/team", methods=["GET"])
def get_team():
    teams = repo.list_entities("team")
    if not teams:
        return error("No team configured", 404)
    return jsonify(teams[0].to_dict())


# ------------------------------------------------------------------
# Programs
# ------------------------------------------------------------------

@app.route("/programs", methods=["GET"])
def list_programs():
    programs = repo.list_entities("program")
    return jsonify([p.to_dict() for p in programs])


@app.route("/programs", methods=["POST"])
def create_program():
    data = request.get_json()
    if not data or "name" not in data:
        return error("'name' required")
    teams = repo.list_entities("team")
    if not teams:
        return error("No team configured", 500)
    prog = Program.create(
        team_id=teams[0].id,
        name=data["name"],
        description=data.get("description", ""),
    )
    repo.save(prog)
    return jsonify(prog.to_dict()), 201


@app.route("/programs/<program_id>", methods=["GET"])
def get_program(program_id):
    prog = repo.load("program", program_id)
    if not prog:
        return error("Program not found", 404)
    return jsonify(prog.to_dict())


# ------------------------------------------------------------------
# Projects
# ------------------------------------------------------------------

@app.route("/projects", methods=["GET"])
def list_projects():
    program_id = request.args.get("program_id")
    projects = repo.list_entities("project")
    if program_id:
        projects = [p for p in projects if p.program_id == program_id]
    return jsonify([p.to_dict() for p in projects])


@app.route("/projects", methods=["POST"])
def create_project():
    data = request.get_json()
    if not data or "name" not in data or "program_id" not in data:
        return error("'name' and 'program_id' required")
    proj = Project.create(
        program_id=data["program_id"],
        name=data["name"],
        description=data.get("description", ""),
    )
    repo.save(proj)
    return jsonify(proj.to_dict()), 201


@app.route("/projects/<project_id>", methods=["GET"])
def get_project(project_id):
    proj = repo.load("project", project_id)
    if not proj:
        return error("Project not found", 404)
    return jsonify(proj.to_dict())


# ------------------------------------------------------------------
# Threads
# ------------------------------------------------------------------

@app.route("/threads", methods=["GET"])
def list_threads():
    project_id = request.args.get("project_id")
    threads = repo.list_entities("thread")
    if project_id:
        threads = [t for t in threads if t.project_id == project_id]
    return jsonify([t.to_dict() for t in threads])


@app.route("/threads", methods=["POST"])
def create_thread():
    data = request.get_json()
    if not data or "question" not in data or "project_id" not in data:
        return error("'question' and 'project_id' required")
    thread = Thread.create(
        project_id=data["project_id"],
        question=data["question"],
        resolution_criterion=data.get("resolution_criterion", ""),
    )
    repo.save(thread)
    return jsonify(thread.to_dict()), 201


@app.route("/threads/<thread_id>", methods=["GET"])
def get_thread(thread_id):
    thread = repo.load("thread", thread_id)
    if not thread:
        return error("Thread not found", 404)
    return jsonify(thread.to_dict())


@app.route("/threads/<thread_id>", methods=["PATCH"])
def update_thread(thread_id):
    thread = repo.load("thread", thread_id)
    if not thread:
        return error("Thread not found", 404)
    data = request.get_json()
    if "state" in data:
        thread.state = ThreadState(data["state"])
    if "question" in data:
        thread.question = data["question"]
    if "resolution_criterion" in data:
        thread.resolution_criterion = data["resolution_criterion"]
    thread.updated_at = __import__("rnd.models", fromlist=["now_iso"]).now_iso()
    repo.save(thread)
    return jsonify(thread.to_dict())


# ------------------------------------------------------------------
# Statements
# ------------------------------------------------------------------

@app.route("/statements", methods=["GET"])
def list_statements():
    thread_id = request.args.get("thread_id")
    stmts = repo.list_entities("statement")
    if thread_id:
        stmts = [s for s in stmts if s.thread_id == thread_id]
    return jsonify([s.to_dict() for s in stmts])


@app.route("/statements", methods=["POST"])
def create_statement():
    data = request.get_json()
    if not data or "hypothesis" not in data or "thread_id" not in data:
        return error("'hypothesis' and 'thread_id' required")
    stmt = Statement.create(
        thread_id=data["thread_id"],
        hypothesis=data["hypothesis"],
    )
    repo.save(stmt)
    return jsonify(stmt.to_dict()), 201


@app.route("/statements/<statement_id>", methods=["GET"])
def get_statement(statement_id):
    stmt = repo.load("statement", statement_id)
    if not stmt:
        return error("Statement not found", 404)
    return jsonify(stmt.to_dict())


# ------------------------------------------------------------------
# Experiments
# ------------------------------------------------------------------

@app.route("/experiments", methods=["GET"])
def list_experiments():
    thread_id = request.args.get("thread_id")
    exps = repo.list_entities("experiment")
    if thread_id:
        exps = [e for e in exps if e.thread_id == thread_id]
    return jsonify([e.to_dict() for e in exps])


@app.route("/experiments", methods=["POST"])
def create_experiment():
    data = request.get_json()
    if not data or "thread_id" not in data or "inputs" not in data:
        return error("'thread_id' and 'inputs' required")
    inputs = ExperimentInputs.from_dict(data["inputs"])
    exp = Experiment.create(
        thread_id=data["thread_id"],
        created_by=data.get("created_by", "user-default"),
        inputs=inputs,
        hypothesis=data.get("hypothesis", ""),
        expected=data.get("expected", ""),
        method=data.get("method", {}),
        imported=data.get("imported", False),
    )
    if data.get("status"):
        exp.status = ExperimentStatus(data["status"])
    if data.get("observed"):
        from .models import ObservedResults
        exp.observed = ObservedResults.from_dict(data["observed"])
    if data.get("interpretation"):
        exp.interpretation = data["interpretation"]
    repo.save(exp)
    return jsonify(exp.to_dict()), 201


@app.route("/experiments/<experiment_id>", methods=["GET"])
def get_experiment(experiment_id):
    exp = repo.load("experiment", experiment_id)
    if not exp:
        return error("Experiment not found", 404)
    return jsonify(exp.to_dict())


@app.route("/experiments/<experiment_id>", methods=["PATCH"])
def update_experiment(experiment_id):
    exp = repo.load("experiment", experiment_id)
    if not exp:
        return error("Experiment not found", 404)
    data = request.get_json()
    if "status" in data:
        exp.status = ExperimentStatus(data["status"])
    if "observed" in data:
        from .models import ObservedResults
        exp.observed = ObservedResults.from_dict(data["observed"])
    if "interpretation" in data:
        exp.interpretation = data["interpretation"]
    if "evidence" in data:
        exp.evidence = [EvidenceLink.from_dict(e) for e in data["evidence"]]
    exp.updated_at = __import__("rnd.models", fromlist=["now_iso"]).now_iso()
    repo.save(exp)
    return jsonify(exp.to_dict())


@app.route("/experiments/<experiment_id>/evidence", methods=["POST"])
def attach_evidence(experiment_id):
    """Attach evidence linking this experiment to a statement."""
    exp = repo.load("experiment", experiment_id)
    if not exp:
        return error("Experiment not found", 404)
    data = request.get_json()
    if not data or "statement_id" not in data or "sign" not in data:
        return error("'statement_id' and 'sign' required")
    link = EvidenceLink(
        statement_id=data["statement_id"],
        sign=EvidenceSign(data["sign"]),
        strength=EvidenceStrength(data.get("strength", "moderate")),
        note=data.get("note", ""),
    )
    exp.evidence.append(link)
    exp.updated_at = __import__("rnd.models", fromlist=["now_iso"]).now_iso()
    repo.save(exp)
    return jsonify(link.to_dict()), 201


# ------------------------------------------------------------------
# Findings
# ------------------------------------------------------------------

@app.route("/findings", methods=["GET"])
def list_findings():
    thread_id = request.args.get("thread_id")
    findings = repo.list_entities("finding")
    if thread_id:
        findings = [f for f in findings if f.thread_id == thread_id]
    return jsonify([f.to_dict() for f in findings])


@app.route("/findings", methods=["POST"])
def create_finding():
    data = request.get_json()
    if not data or "thread_id" not in data or "summary" not in data:
        return error("'thread_id' and 'summary' required")
    resolutions = []
    for r in data.get("statement_resolutions", []):
        resolutions.append(StatementResolution(
            statement_id=r["statement_id"],
            resolution=FindingResolution(r["resolution"]),
            note=r.get("note", ""),
        ))
    finding = Finding.create(
        thread_id=data["thread_id"],
        summary=data["summary"],
        reasoning=data.get("reasoning", ""),
        statement_resolutions=resolutions,
        experiment_refs=data.get("experiment_refs", []),
    )
    repo.save(finding)

    # Optionally resolve the thread
    if data.get("resolve_thread", False):
        thread = repo.load("thread", data["thread_id"])
        if thread:
            thread.state = ThreadState.RESOLVED
            thread.updated_at = __import__("rnd.models", fromlist=["now_iso"]).now_iso()
            repo.save(thread)

    return jsonify(finding.to_dict()), 201


@app.route("/findings/<finding_id>", methods=["GET"])
def get_finding(finding_id):
    finding = repo.load("finding", finding_id)
    if not finding:
        return error("Finding not found", 404)
    return jsonify(finding.to_dict())


# ------------------------------------------------------------------
# Architectures
# ------------------------------------------------------------------

@app.route("/architectures", methods=["GET"])
def list_architectures():
    archs = repo.list_entities("architecture")
    include_archived = request.args.get("include_archived", "false").lower() == "true"
    if not include_archived:
        archs = [a for a in archs if not a.archived]
    return jsonify([a.to_dict() for a in archs])


@app.route("/architectures", methods=["POST"])
def import_architecture():
    data = request.get_json()
    if not data or "content" not in data:
        return error("'content' required")
    arch = repo.import_architecture(
        name=data.get("name", "unnamed"),
        content=data["content"],
        variant_of=data.get("variant_of"),
    )
    return jsonify(arch.to_dict()), 201


@app.route("/architectures/<architecture_id>", methods=["GET"])
def get_architecture(architecture_id):
    arch = repo.load("architecture", architecture_id)
    if not arch:
        return error("Architecture not found", 404)
    return jsonify(arch.to_dict())


@app.route("/architectures/<architecture_id>/archive", methods=["POST"])
def archive_architecture(architecture_id):
    if repo.archive("architecture", architecture_id):
        return jsonify({"archived": True})
    return error("Architecture not found", 404)


@app.route("/architectures/by-hash/<path:hash_value>", methods=["GET"])
def get_architecture_by_hash(hash_value):
    arch = repo.find_architecture_by_hash(hash_value)
    if not arch:
        return error("Architecture not found for hash", 404)
    return jsonify(arch.to_dict())


# ------------------------------------------------------------------
# Papers
# ------------------------------------------------------------------

@app.route("/papers", methods=["GET"])
def list_papers():
    papers = repo.list_entities("paper")
    return jsonify([p.to_dict() for p in papers])


@app.route("/papers", methods=["POST"])
def create_paper():
    data = request.get_json()
    if not data or "title" not in data:
        return error("'title' required")
    paper = Paper.create(
        title=data["title"],
        authors=data.get("authors", []),
        program_id=data.get("program_id", ""),
        target=data.get("target", "arxiv"),
    )
    if data.get("project_ids"):
        paper.project_ids = data["project_ids"]
    if data.get("thread_ids"):
        paper.thread_ids = data["thread_ids"]
    repo.save(paper)
    return jsonify(paper.to_dict()), 201


@app.route("/papers/<paper_id>", methods=["GET"])
def get_paper(paper_id):
    paper = repo.load("paper", paper_id)
    if not paper:
        return error("Paper not found", 404)
    return jsonify(paper.to_dict())


@app.route("/papers/<paper_id>", methods=["PATCH"])
def update_paper(paper_id):
    paper = repo.load("paper", paper_id)
    if not paper:
        return error("Paper not found", 404)
    data = request.get_json()
    if "title" in data:
        paper.title = data["title"]
    if "status" in data:
        paper.status = PaperStatus(data["status"])
    if "authors" in data:
        paper.authors = data["authors"]
    if "thread_ids" in data:
        paper.thread_ids = data["thread_ids"]
    if "project_ids" in data:
        paper.project_ids = data["project_ids"]
    if "metadata" in data:
        paper.metadata.update(data["metadata"])
    paper.updated_at = __import__("rnd.models", fromlist=["now_iso"]).now_iso()
    repo.save(paper)
    return jsonify(paper.to_dict())


# ------------------------------------------------------------------
# Disclosures
# ------------------------------------------------------------------

@app.route("/disclosures", methods=["GET"])
def list_disclosures():
    discs = repo.list_entities("disclosure")
    return jsonify([d.to_dict() for d in discs])


@app.route("/disclosures", methods=["POST"])
def create_disclosure():
    data = request.get_json()
    if not data or "title" not in data:
        return error("'title' required")
    disc = Disclosure.create(
        title=data["title"],
        disclosure_type=DisclosureType(data.get("type", "other")),
        created_by=data.get("created_by", "user-default"),
        tags=data.get("tags", []),
    )
    if data.get("source_url"):
        disc.source_url = data["source_url"]
    if data.get("visibility"):
        disc.visibility = Visibility(data["visibility"])
    repo.save(disc)

    # Save content if provided inline
    if data.get("content"):
        repo.save_disclosure_content(disc, data["content"])

    return jsonify(disc.to_dict()), 201


@app.route("/disclosures/<disclosure_id>", methods=["GET"])
def get_disclosure(disclosure_id):
    disc = repo.load("disclosure", disclosure_id)
    if not disc:
        return error("Disclosure not found", 404)
    result = disc.to_dict()
    if request.args.get("include_content", "false").lower() == "true":
        result["content"] = repo.load_disclosure_content(disc)
    return jsonify(result)


@app.route("/disclosures/<disclosure_id>/content", methods=["GET"])
def get_disclosure_content(disclosure_id):
    disc = repo.load("disclosure", disclosure_id)
    if not disc:
        return error("Disclosure not found", 404)
    content = repo.load_disclosure_content(disc)
    return content, 200, {"Content-Type": "text/markdown; charset=utf-8"}


@app.route("/disclosures/<disclosure_id>/content", methods=["PUT"])
def update_disclosure_content(disclosure_id):
    disc = repo.load("disclosure", disclosure_id)
    if not disc:
        return error("Disclosure not found", 404)
    content = request.get_data(as_text=True)
    repo.save_disclosure_content(disc, content)
    return jsonify({"updated": True})


# ------------------------------------------------------------------
# Artifacts
# ------------------------------------------------------------------

@app.route("/artifacts", methods=["GET"])
def list_artifacts():
    artifacts = repo.list_entities("artifact")
    return jsonify([a.to_dict() for a in artifacts])


@app.route("/artifacts/<artifact_id>", methods=["GET"])
def get_artifact(artifact_id):
    art = repo.load("artifact", artifact_id)
    if not art:
        return error("Artifact not found", 404)
    return jsonify(art.to_dict())


# ------------------------------------------------------------------
# Search (unified)
# ------------------------------------------------------------------

@app.route("/search", methods=["GET"])
def search_entities():
    q = request.args.get("q", "")
    entity_type = request.args.get("type")
    if not q:
        return error("'q' query parameter required")
    ensure_index()
    results = index.search(q, entity_type=entity_type)
    return jsonify(results)


# ------------------------------------------------------------------
# Graph queries
# ------------------------------------------------------------------

@app.route("/graph/children/<entity_id>", methods=["GET"])
def graph_children(entity_id):
    ensure_index()
    children = index.find_children(entity_id)
    return jsonify(children)


@app.route("/graph/evidence/<statement_id>", methods=["GET"])
def graph_evidence(statement_id):
    ensure_index()
    evidence = index.get_evidence_for_statement(statement_id)
    return jsonify(evidence)


@app.route("/graph/citations/<entity_id>", methods=["GET"])
def graph_citations(entity_id):
    ensure_index()
    citations = index.get_citations_for_entity(entity_id)
    return jsonify(citations)


@app.route("/graph/backlinks/<disclosure_id>", methods=["GET"])
def graph_backlinks(disclosure_id):
    ensure_index()
    backlinks = index.get_disclosure_backlinks(disclosure_id)
    return jsonify(backlinks)


@app.route("/graph/variants/<architecture_id>", methods=["GET"])
def graph_variants(architecture_id):
    ensure_index()
    variants = index.get_architecture_variants(architecture_id)
    return jsonify(variants)


# ------------------------------------------------------------------
# Index management
# ------------------------------------------------------------------

@app.route("/index/rebuild", methods=["POST"])
def rebuild_index():
    ensure_index()
    stats = index.rebuild(repo.root)
    return jsonify(stats)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def create_app(repo_path: str = ".") -> Flask:
    """Create and configure the Flask app with a specific repo."""
    global repo, index
    repo = RNDRepo(repo_path)
    if not repo.is_initialized():
        print(f"Error: {repo_path} is not an initialized RND repository.")
        print("Run 'python -m rnd init' first.")
        sys.exit(1)
    index = DerivedIndex(repo.index_path)
    index.open()
    index.rebuild(repo.root)
    return app


def main():
    parser = argparse.ArgumentParser(description="RND Platform API Server")
    parser.add_argument("--repo", default=".", help="Repository root (default: .)")
    parser.add_argument("--port", type=int, default=5001, help="Port (default: 5001)")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    args = parser.parse_args()

    create_app(args.repo)

    print("=" * 60)
    print("RND Platform API Server")
    print("=" * 60)
    print(f"  Repo:  {repo.root}")
    print(f"  URL:   http://{args.host}:{args.port}")
    print("=" * 60)

    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
