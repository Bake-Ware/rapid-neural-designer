"""Git-backed persistence layer (SDD §5).

Manages the repository layout, entity CRUD, and path resolution.
All entities are stored as canonical JSON files in a deterministic
directory structure. The git repo is the source of truth.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Optional

from .canonical import canonicalize, content_hash
from .models import (
    ENTITY_TYPES,
    Architecture,
    Artifact,
    Disclosure,
    Experiment,
    Finding,
    Paper,
    Program,
    Project,
    Statement,
    Team,
    Thread,
)

# Repo layout constants
TEAM_FILE = "team.json"
PROGRAMS_DIR = "programs"
ARCHITECTURES_DIR = "architectures"
DISCLOSURES_DIR = "disclosures"
PAPERS_DIR = "papers"
ARTIFACTS_DIR = "artifacts"
RND_DIR = ".rnd"
CONFIG_FILE = ".rnd/config.yaml"
INDEX_FILE = ".rnd/index.sqlite"

GITIGNORE_CONTENT = """\
# RND Platform derived files
.rnd/index.sqlite
.rnd/index.sqlite-journal
.rnd/index.sqlite-wal
__pycache__/
*.pyc
"""

DEFAULT_CONFIG = """\
# RND Platform configuration
version: "0.1.0"
artifact_store: local
"""


class RNDRepo:
    """Manages an RND research repository."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    @property
    def rnd_dir(self) -> Path:
        return self.root / RND_DIR

    @property
    def index_path(self) -> Path:
        return self.root / INDEX_FILE

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init(self, team_name: str = "", user_id: str = "") -> None:
        """Initialize the repository layout (SDD §5.2)."""
        dirs = [
            self.root / PROGRAMS_DIR,
            self.root / ARCHITECTURES_DIR,
            self.root / DISCLOSURES_DIR,
            self.root / PAPERS_DIR,
            self.root / ARTIFACTS_DIR,
            self.rnd_dir / "hooks",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

        # .rnd/config.yaml
        config_path = self.root / CONFIG_FILE
        if not config_path.exists():
            config_path.write_text(DEFAULT_CONFIG, encoding="utf-8")

        # .gitignore for derived files
        gitignore = self.rnd_dir / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text(GITIGNORE_CONTENT, encoding="utf-8")

        # Create team.json if team info provided
        if team_name and user_id:
            team_path = self.root / TEAM_FILE
            if not team_path.exists():
                team = Team.create(name=team_name, user_id=user_id)
                self._write_entity(team_path, team.to_dict())

    def is_initialized(self) -> bool:
        """Check if this directory has been initialized as an RND repo."""
        return self.rnd_dir.exists()

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def _entity_path(self, entity) -> Path:
        """Resolve the filesystem path for an entity based on its type and hierarchy."""
        if isinstance(entity, Team):
            return self.root / TEAM_FILE

        if isinstance(entity, Program):
            return self.root / PROGRAMS_DIR / entity.id / "program.json"

        if isinstance(entity, Project):
            prog_dir = self._find_parent_dir("program", entity.program_id)
            return prog_dir / "projects" / entity.id / "project.json"

        if isinstance(entity, Thread):
            proj_dir = self._find_parent_dir("project", entity.project_id)
            return proj_dir / "threads" / entity.id / "thread.json"

        if isinstance(entity, Statement):
            thread_dir = self._find_parent_dir("thread", entity.thread_id)
            return thread_dir / "statements" / f"{entity.id}.json"

        if isinstance(entity, Experiment):
            thread_dir = self._find_parent_dir("thread", entity.thread_id)
            return thread_dir / "experiments" / f"{entity.id}.json"

        if isinstance(entity, Finding):
            thread_dir = self._find_parent_dir("thread", entity.thread_id)
            return thread_dir / "findings" / f"{entity.id}.json"

        if isinstance(entity, Architecture):
            return self.root / ARCHITECTURES_DIR / f"{entity.id}.json"

        if isinstance(entity, Paper):
            return self.root / PAPERS_DIR / entity.id / "paper.json"

        if isinstance(entity, Disclosure):
            return self.root / DISCLOSURES_DIR / entity.id / "disclosure.json"

        if isinstance(entity, Artifact):
            return self.root / ARTIFACTS_DIR / entity.id / "artifact.json"

        raise ValueError(f"Unknown entity type: {type(entity)}")

    def _find_parent_dir(self, parent_type: str, parent_id: str) -> Path:
        """Find the directory containing a parent entity by scanning."""
        patterns = {
            "program": f"{PROGRAMS_DIR}/{parent_id}",
            "project": f"{PROGRAMS_DIR}/*/projects/{parent_id}",
            "thread": f"{PROGRAMS_DIR}/*/projects/*/threads/{parent_id}",
        }
        pattern = patterns.get(parent_type)
        if not pattern:
            raise ValueError(f"Unknown parent type: {parent_type}")

        # Try direct path first (most common case)
        if "*" not in pattern:
            direct = self.root / pattern
            if direct.exists():
                return direct

        # Glob for it
        matches = list(self.root.glob(pattern))
        if matches:
            return matches[0]

        raise FileNotFoundError(
            f"Parent {parent_type} '{parent_id}' not found in repo"
        )

    def _find_entity_file(self, entity_type: str, entity_id: str) -> Optional[Path]:
        """Find an entity's JSON file by type and ID via filesystem scan."""
        patterns = {
            "team": [TEAM_FILE],
            "program": [f"{PROGRAMS_DIR}/*/program.json"],
            "project": [f"{PROGRAMS_DIR}/*/projects/*/project.json"],
            "thread": [f"{PROGRAMS_DIR}/*/projects/*/threads/*/thread.json"],
            "statement": [f"{PROGRAMS_DIR}/*/projects/*/threads/*/statements/*.json"],
            "experiment": [f"{PROGRAMS_DIR}/*/projects/*/threads/*/experiments/*.json"],
            "finding": [f"{PROGRAMS_DIR}/*/projects/*/threads/*/findings/*.json"],
            "architecture": [f"{ARCHITECTURES_DIR}/*.json"],
            "paper": [f"{PAPERS_DIR}/*/paper.json"],
            "disclosure": [f"{DISCLOSURES_DIR}/*/disclosure.json"],
            "artifact": [f"{ARTIFACTS_DIR}/*/artifact.json"],
        }

        for pattern in patterns.get(entity_type, []):
            for path in self.root.glob(pattern):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if data.get("id") == entity_id:
                        return path
                except (json.JSONDecodeError, KeyError):
                    continue
        return None

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def _write_entity(self, path: Path, data: dict) -> Path:
        """Write canonical JSON to a file, creating parent dirs."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(canonicalize(data), encoding="utf-8")
        return path

    def save(self, entity) -> Path:
        """Save an entity to its canonical location. Returns the file path."""
        path = self._entity_path(entity)
        return self._write_entity(path, entity.to_dict())

    def load(self, entity_type: str, entity_id: str):
        """Load an entity by type name and ID. Returns the entity or None."""
        path = self._find_entity_file(entity_type, entity_id)
        if path is None:
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        cls = ENTITY_TYPES.get(entity_type)
        if cls is None:
            raise ValueError(f"Unknown entity type: {entity_type}")
        return cls.from_dict(data)

    def load_file(self, path: Path):
        """Load an entity from a known file path. Infers type from ID prefix."""
        data = json.loads(path.read_text(encoding="utf-8"))
        entity_id = data.get("id", "")
        prefix = entity_id.split("-")[0] if "-" in entity_id else ""
        # Reverse-lookup type from prefix
        prefix_to_type = {
            "team": "team", "prog": "program", "proj": "project",
            "thread": "thread", "stmt": "statement", "exp": "experiment",
            "find": "finding", "arch": "architecture", "paper": "paper",
            "disc": "disclosure", "art": "artifact",
        }
        type_name = prefix_to_type.get(prefix)
        if type_name is None:
            raise ValueError(f"Cannot infer entity type from ID: {entity_id}")
        cls = ENTITY_TYPES[type_name]
        return cls.from_dict(data)

    def list_entities(self, entity_type: str, **filters) -> list:
        """List all entities of a given type, optionally filtered."""
        patterns = {
            "team": [TEAM_FILE],
            "program": [f"{PROGRAMS_DIR}/*/program.json"],
            "project": [f"{PROGRAMS_DIR}/*/projects/*/project.json"],
            "thread": [f"{PROGRAMS_DIR}/*/projects/*/threads/*/thread.json"],
            "statement": [f"{PROGRAMS_DIR}/*/projects/*/threads/*/statements/*.json"],
            "experiment": [f"{PROGRAMS_DIR}/*/projects/*/threads/*/experiments/*.json"],
            "finding": [f"{PROGRAMS_DIR}/*/projects/*/threads/*/findings/*.json"],
            "architecture": [f"{ARCHITECTURES_DIR}/*.json"],
            "paper": [f"{PAPERS_DIR}/*/paper.json"],
            "disclosure": [f"{DISCLOSURES_DIR}/*/disclosure.json"],
            "artifact": [f"{ARTIFACTS_DIR}/*/artifact.json"],
        }

        results = []
        cls = ENTITY_TYPES.get(entity_type)
        if cls is None:
            raise ValueError(f"Unknown entity type: {entity_type}")

        for pattern in patterns.get(entity_type, []):
            for path in self.root.glob(pattern):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    entity = cls.from_dict(data)
                    # Apply filters
                    match = True
                    for key, value in filters.items():
                        if getattr(entity, key, None) != value:
                            match = False
                            break
                    if match:
                        results.append(entity)
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
        return results

    def archive(self, entity_type: str, entity_id: str) -> bool:
        """Soft-delete: set archived=True on an entity. Returns True if found."""
        path = self._find_entity_file(entity_type, entity_id)
        if path is None:
            return False
        data = json.loads(path.read_text(encoding="utf-8"))
        data["archived"] = True
        self._write_entity(path, data)
        return True

    # ------------------------------------------------------------------
    # Architecture-specific operations
    # ------------------------------------------------------------------

    def import_architecture(self, name: str, content: dict,
                            variant_of: str | None = None) -> Architecture:
        """Import an architecture JSON, compute its content hash, and save."""
        arch_hash = content_hash(content)
        arch = Architecture.create(
            name=name, content=content,
            content_hash=arch_hash, variant_of=variant_of,
        )
        self.save(arch)
        return arch

    def find_architecture_by_hash(self, hash_value: str) -> Optional[Architecture]:
        """Look up an architecture by its content hash."""
        for arch in self.list_entities("architecture"):
            if arch.content_hash == hash_value:
                return arch
        return None

    # ------------------------------------------------------------------
    # Disclosure content
    # ------------------------------------------------------------------

    def save_disclosure_content(self, disclosure: Disclosure,
                                content: str) -> Path:
        """Write the markdown content file for a Disclosure."""
        content_path = self.root / disclosure.content_ref
        content_path.parent.mkdir(parents=True, exist_ok=True)
        content_path.write_text(content, encoding="utf-8")
        return content_path

    def load_disclosure_content(self, disclosure: Disclosure) -> str:
        """Read the markdown content of a Disclosure."""
        content_path = self.root / disclosure.content_ref
        if content_path.exists():
            return content_path.read_text(encoding="utf-8")
        return ""

    # ------------------------------------------------------------------
    # Paper section content
    # ------------------------------------------------------------------

    def save_paper_section_content(self, paper: Paper,
                                    section: "PaperSection",
                                    content: str) -> Path:
        """Write the markdown content for a paper section."""
        content_path = self.root / section.content_ref
        content_path.parent.mkdir(parents=True, exist_ok=True)
        content_path.write_text(content, encoding="utf-8")
        return content_path

    def load_paper_section_content(self, section: "PaperSection") -> str:
        """Read the markdown content of a paper section."""
        content_path = self.root / section.content_ref
        if content_path.exists():
            return content_path.read_text(encoding="utf-8")
        return ""

    # ------------------------------------------------------------------
    # Git operations
    # ------------------------------------------------------------------

    def _git(self, *args: str) -> subprocess.CompletedProcess:
        """Run a git command in the repo root."""
        return subprocess.run(
            ["git"] + list(args),
            cwd=self.root,
            capture_output=True,
            text=True,
        )

    def save_as_result(self, label: str, message: str = "") -> str:
        """Tag the current commit with a user-provided label (SDD §5.3)."""
        tag_name = f"result/{label}"
        msg = message or f"Result: {label}"
        result = self._git("tag", "-a", tag_name, "-m", msg)
        if result.returncode != 0:
            raise RuntimeError(f"git tag failed: {result.stderr}")
        return tag_name

    def create_reproducibility_release(self, paper: Paper,
                                        manifest: dict) -> str:
        """Create a git tag pinning all entities cited by a paper (SDD §7.5)."""
        tag_name = f"release/{paper.id}"
        # Write manifest to papers dir
        manifest_path = self.root / PAPERS_DIR / paper.id / "manifest.json"
        self._write_entity(manifest_path, manifest)

        # Stage and commit manifest
        self._git("add", str(manifest_path.relative_to(self.root)))
        self._git("commit", "-m", f"Reproducibility release for {paper.title}")

        # Tag
        result = self._git("tag", "-a", tag_name, "-m",
                           f"Reproducibility release: {paper.title}")
        if result.returncode != 0:
            raise RuntimeError(f"git tag failed: {result.stderr}")
        return tag_name
