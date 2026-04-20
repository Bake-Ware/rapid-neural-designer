"""CLI entry point for RND Platform.

Usage: python -m rnd <command> [options]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .canonical import canonicalize, content_hash
from .index import DerivedIndex
from .models import (
    Architecture,
    Disclosure,
    DisclosureType,
    Experiment,
    ExperimentInputs,
    ExperimentStatus,
    Finding,
    Program,
    Project,
    Statement,
    Thread,
)
from .repo import RNDRepo


def get_repo(args) -> RNDRepo:
    root = getattr(args, "repo", None) or "."
    repo = RNDRepo(root)
    if not repo.is_initialized():
        print(f"Error: {root} is not an initialized RND repository.", file=sys.stderr)
        print("Run 'python -m rnd init' first.", file=sys.stderr)
        sys.exit(1)
    return repo


# ------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------

def cmd_init(args):
    """Initialize a new RND research repository."""
    repo = RNDRepo(args.repo or ".")
    repo.init(team_name=args.team or "", user_id=args.user or "")
    print(f"Initialized RND repository at {repo.root}")
    if args.team:
        print(f"  Team: {args.team}")


def cmd_import_architecture(args):
    """Import an architecture JSON file."""
    repo = get_repo(args)
    path = Path(args.file)
    if not path.exists():
        print(f"Error: {path} not found.", file=sys.stderr)
        sys.exit(1)

    content = json.loads(path.read_text(encoding="utf-8"))
    name = args.name or path.stem
    arch = repo.import_architecture(name=name, content=content,
                                     variant_of=args.variant_of)
    print(f"Imported architecture: {arch.id}")
    print(f"  Name: {arch.name}")
    print(f"  Hash: {arch.content_hash}")


def cmd_create_program(args):
    """Create a new program."""
    repo = get_repo(args)
    # Load team to get team_id
    teams = repo.list_entities("team")
    if not teams:
        print("Error: no team found. Re-init with --team.", file=sys.stderr)
        sys.exit(1)
    team = teams[0] if isinstance(teams[0], dict) else teams[0]
    team_id = team.id if hasattr(team, "id") else team["id"]

    prog = Program.create(team_id=team_id, name=args.name,
                          description=args.description or "")
    repo.save(prog)
    print(f"Created program: {prog.id}")
    print(f"  Name: {prog.name}")


def cmd_create_project(args):
    """Create a new project under a program."""
    repo = get_repo(args)
    proj = Project.create(program_id=args.program_id, name=args.name,
                          description=args.description or "")
    repo.save(proj)
    print(f"Created project: {proj.id}")
    print(f"  Name: {proj.name}")


def cmd_create_thread(args):
    """Create a new thread (research question) under a project."""
    repo = get_repo(args)
    thread = Thread.create(project_id=args.project_id, question=args.question,
                           resolution_criterion=args.criterion or "")
    repo.save(thread)
    print(f"Created thread: {thread.id}")
    print(f"  Question: {thread.question}")


def cmd_create_statement(args):
    """Create a statement (hypothesis) in a thread."""
    repo = get_repo(args)
    stmt = Statement.create(thread_id=args.thread_id, hypothesis=args.hypothesis)
    repo.save(stmt)
    print(f"Created statement: {stmt.id}")


def cmd_create_disclosure(args):
    """Create a disclosure from a markdown file or stdin."""
    repo = get_repo(args)
    dtype = DisclosureType(args.type)

    if args.file:
        content = Path(args.file).read_text(encoding="utf-8")
    else:
        content = sys.stdin.read()

    disc = Disclosure.create(
        title=args.title,
        disclosure_type=dtype,
        created_by=args.user or "user-default",
        tags=args.tags.split(",") if args.tags else [],
    )
    repo.save(disc)
    repo.save_disclosure_content(disc, content)
    print(f"Created disclosure: {disc.id}")
    print(f"  Title: {disc.title}")


def cmd_list(args):
    """List entities of a given type."""
    repo = get_repo(args)
    entities = repo.list_entities(args.type)
    if not entities:
        print(f"No {args.type} entities found.")
        return
    for entity in entities:
        eid = entity.id
        name = (getattr(entity, "name", None)
                or getattr(entity, "title", None)
                or getattr(entity, "question", None)
                or getattr(entity, "hypothesis", None)
                or getattr(entity, "summary", None)
                or "")
        state = getattr(entity, "state", None)
        status = getattr(entity, "status", None)
        extra = ""
        if state:
            extra = f" [{state.value if hasattr(state, 'value') else state}]"
        elif status:
            extra = f" [{status.value if hasattr(status, 'value') else status}]"
        print(f"  {eid}  {name}{extra}")


def cmd_index_rebuild(args):
    """Rebuild the derived SQLite index from files."""
    repo = get_repo(args)
    with DerivedIndex(repo.index_path) as idx:
        stats = idx.rebuild(repo.root)
    print("Index rebuilt:")
    for key, count in stats.items():
        print(f"  {key}: {count}")


def cmd_show(args):
    """Show full details of an entity."""
    repo = get_repo(args)
    # Infer type from ID prefix
    prefix_map = {
        "team": "team", "prog": "program", "proj": "project",
        "thread": "thread", "stmt": "statement", "exp": "experiment",
        "find": "finding", "arch": "architecture", "paper": "paper",
        "disc": "disclosure", "art": "artifact",
    }
    prefix = args.id.split("-")[0] if "-" in args.id else ""
    entity_type = prefix_map.get(prefix)
    if not entity_type:
        print(f"Error: cannot infer type from ID '{args.id}'", file=sys.stderr)
        sys.exit(1)

    entity = repo.load(entity_type, args.id)
    if entity is None:
        print(f"Entity {args.id} not found.", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(entity.to_dict(), indent=2))


# ------------------------------------------------------------------
# Parser
# ------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rnd", description="RND Platform CLI")
    parser.add_argument("--repo", default=".", help="Repository root (default: .)")
    sub = parser.add_subparsers(dest="command")

    # init
    p = sub.add_parser("init", help="Initialize RND repository")
    p.add_argument("--team", help="Team name")
    p.add_argument("--user", help="User ID for initial team member")

    # import-architecture
    p = sub.add_parser("import-architecture", help="Import architecture JSON")
    p.add_argument("file", help="Path to architecture JSON file")
    p.add_argument("--name", help="Architecture name (default: filename)")
    p.add_argument("--variant-of", help="Parent architecture ID")

    # create-program
    p = sub.add_parser("create-program", help="Create a program")
    p.add_argument("name", help="Program name")
    p.add_argument("--description", help="Description")

    # create-project
    p = sub.add_parser("create-project", help="Create a project")
    p.add_argument("name", help="Project name")
    p.add_argument("--program-id", required=True, help="Parent program ID")
    p.add_argument("--description", help="Description")

    # create-thread
    p = sub.add_parser("create-thread", help="Create a thread")
    p.add_argument("question", help="Research question")
    p.add_argument("--project-id", required=True, help="Parent project ID")
    p.add_argument("--criterion", help="Resolution criterion")

    # create-statement
    p = sub.add_parser("create-statement", help="Create a statement")
    p.add_argument("hypothesis", help="Hypothesis text")
    p.add_argument("--thread-id", required=True, help="Parent thread ID")

    # create-disclosure
    p = sub.add_parser("create-disclosure", help="Create a disclosure")
    p.add_argument("title", help="Disclosure title")
    p.add_argument("--type", default="other", help="Disclosure type")
    p.add_argument("--file", help="Markdown content file (or stdin)")
    p.add_argument("--user", help="Creator user ID")
    p.add_argument("--tags", help="Comma-separated tags")

    # list
    p = sub.add_parser("list", help="List entities")
    p.add_argument("type", help="Entity type (program, project, thread, etc.)")

    # show
    p = sub.add_parser("show", help="Show entity details")
    p.add_argument("id", help="Entity ID")

    # index
    p = sub.add_parser("index", help="Index operations")
    p.add_argument("action", choices=["rebuild"], help="Index action")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    commands = {
        "init": cmd_init,
        "import-architecture": cmd_import_architecture,
        "create-program": cmd_create_program,
        "create-project": cmd_create_project,
        "create-thread": cmd_create_thread,
        "create-statement": cmd_create_statement,
        "create-disclosure": cmd_create_disclosure,
        "list": cmd_list,
        "show": cmd_show,
        "index": cmd_index_rebuild,
    }

    cmd = commands.get(args.command)
    if cmd:
        cmd(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
