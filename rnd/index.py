"""SQLite derived index for fast queries (SDD §5.6).

This index is DERIVED, not authoritative. The JSON files are the source
of truth. The index is gitignored and rebuilt from files on demand.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    parent_id TEXT,
    file_path TEXT NOT NULL,
    name TEXT,
    state TEXT,
    content_hash TEXT,
    archived INTEGER DEFAULT 0,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS evidence_links (
    experiment_id TEXT NOT NULL,
    statement_id TEXT NOT NULL,
    sign TEXT NOT NULL,
    strength TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES entities(id),
    FOREIGN KEY (statement_id) REFERENCES entities(id)
);

CREATE TABLE IF NOT EXISTS citations (
    citing_entity_id TEXT NOT NULL,
    disclosure_id TEXT,
    external_type TEXT,
    external_id TEXT,
    relevance TEXT,
    FOREIGN KEY (citing_entity_id) REFERENCES entities(id)
);

CREATE TABLE IF NOT EXISTS architecture_variants (
    architecture_id TEXT NOT NULL,
    variant_of TEXT NOT NULL,
    FOREIGN KEY (architecture_id) REFERENCES entities(id),
    FOREIGN KEY (variant_of) REFERENCES entities(id)
);

CREATE TABLE IF NOT EXISTS paper_bindings (
    paper_id TEXT NOT NULL,
    section_id TEXT NOT NULL,
    binding_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    span_start INTEGER,
    span_end INTEGER,
    relevance TEXT,
    FOREIGN KEY (paper_id) REFERENCES entities(id)
);

CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_parent ON entities(parent_id);
CREATE INDEX IF NOT EXISTS idx_entities_hash ON entities(content_hash);
CREATE INDEX IF NOT EXISTS idx_evidence_experiment ON evidence_links(experiment_id);
CREATE INDEX IF NOT EXISTS idx_evidence_statement ON evidence_links(statement_id);
CREATE INDEX IF NOT EXISTS idx_citations_citing ON citations(citing_entity_id);
CREATE INDEX IF NOT EXISTS idx_citations_disclosure ON citations(disclosure_id);
CREATE INDEX IF NOT EXISTS idx_bindings_paper ON paper_bindings(paper_id);
CREATE INDEX IF NOT EXISTS idx_bindings_entity ON paper_bindings(entity_id);
"""


class DerivedIndex:
    """SQLite-backed query index over the RND repository."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.conn: Optional[sqlite3.Connection] = None

    def open(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # Rebuild from files
    # ------------------------------------------------------------------

    def rebuild(self, repo_root: Path) -> dict:
        """Drop and rebuild the entire index from JSON files. Returns stats."""
        assert self.conn is not None, "Index not open"
        cur = self.conn.cursor()

        # Clear existing data
        for table in ["paper_bindings", "architecture_variants",
                      "citations", "evidence_links", "entities"]:
            cur.execute(f"DELETE FROM {table}")

        stats = {"entities": 0, "evidence_links": 0, "citations": 0,
                 "variants": 0, "bindings": 0}

        # Scan patterns for each entity type
        scan_patterns = {
            "team": ["team.json"],
            "program": ["programs/*/program.json"],
            "project": ["programs/*/projects/*/project.json"],
            "thread": ["programs/*/projects/*/threads/*/thread.json"],
            "statement": ["programs/*/projects/*/threads/*/statements/*.json"],
            "experiment": ["programs/*/projects/*/threads/*/experiments/*.json"],
            "finding": ["programs/*/projects/*/threads/*/findings/*.json"],
            "architecture": ["architectures/*.json"],
            "paper": ["papers/*/paper.json"],
            "disclosure": ["disclosures/*/disclosure.json"],
            "artifact": ["artifacts/*/artifact.json"],
        }

        for entity_type, patterns in scan_patterns.items():
            for pattern in patterns:
                for path in repo_root.glob(pattern):
                    try:
                        data = json.loads(path.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue

                    entity_id = data.get("id", "")
                    rel_path = str(path.relative_to(repo_root))

                    # Determine parent_id based on type
                    parent_id = (
                        data.get("team_id")
                        or data.get("program_id")
                        or data.get("project_id")
                        or data.get("thread_id")
                        or data.get("parent_id")
                    )

                    # Determine name
                    name = (
                        data.get("name")
                        or data.get("title")
                        or data.get("question")
                        or data.get("hypothesis")
                        or data.get("summary")
                    )

                    cur.execute(
                        """INSERT OR REPLACE INTO entities
                        (id, entity_type, parent_id, file_path, name, state,
                         content_hash, archived, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            entity_id, entity_type, parent_id, rel_path,
                            name, data.get("state"),
                            data.get("content_hash"),
                            1 if data.get("archived") else 0,
                            data.get("created_at"), data.get("updated_at"),
                        ),
                    )
                    stats["entities"] += 1

                    # Index evidence links from experiments
                    if entity_type == "experiment":
                        for ev in data.get("evidence", []):
                            cur.execute(
                                """INSERT INTO evidence_links
                                (experiment_id, statement_id, sign, strength)
                                VALUES (?, ?, ?, ?)""",
                                (entity_id, ev["statement_id"],
                                 ev["sign"], ev["strength"]),
                            )
                            stats["evidence_links"] += 1

                    # Index citations
                    for cit in data.get("citations", []):
                        disc_id = cit.get("disclosure_id")
                        ext = cit.get("external", {})
                        cur.execute(
                            """INSERT INTO citations
                            (citing_entity_id, disclosure_id, external_type,
                             external_id, relevance)
                            VALUES (?, ?, ?, ?, ?)""",
                            (entity_id, disc_id,
                             ext.get("type"), ext.get("id"),
                             cit.get("relevance") or ext.get("relevance")),
                        )
                        stats["citations"] += 1

                    # Index architecture variants
                    if entity_type == "architecture" and data.get("variant_of"):
                        cur.execute(
                            """INSERT INTO architecture_variants
                            (architecture_id, variant_of)
                            VALUES (?, ?)""",
                            (entity_id, data["variant_of"]),
                        )
                        stats["variants"] += 1

                    # Index paper bindings
                    if entity_type == "paper":
                        for section in data.get("sections", []):
                            for binding in section.get("bindings", []):
                                span = binding.get("span", {})
                                cur.execute(
                                    """INSERT INTO paper_bindings
                                    (paper_id, section_id, binding_type,
                                     entity_id, span_start, span_end, relevance)
                                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                                    (entity_id, section["id"],
                                     binding["type"], binding["id"],
                                     span.get("start"), span.get("end"),
                                     binding.get("relevance")),
                                )
                                stats["bindings"] += 1

        self.conn.commit()
        return stats

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def find_by_id(self, entity_id: str) -> Optional[dict]:
        """Look up an entity's index record by ID."""
        assert self.conn is not None
        row = self.conn.execute(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        return dict(row) if row else None

    def find_by_type(self, entity_type: str, include_archived: bool = False) -> list[dict]:
        """List entities by type."""
        assert self.conn is not None
        if include_archived:
            rows = self.conn.execute(
                "SELECT * FROM entities WHERE entity_type = ? ORDER BY created_at",
                (entity_type,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM entities WHERE entity_type = ? AND archived = 0 ORDER BY created_at",
                (entity_type,),
            ).fetchall()
        return [dict(r) for r in rows]

    def find_children(self, parent_id: str) -> list[dict]:
        """Find all entities whose parent is the given ID."""
        assert self.conn is not None
        rows = self.conn.execute(
            "SELECT * FROM entities WHERE parent_id = ? ORDER BY created_at",
            (parent_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def find_by_hash(self, content_hash: str) -> Optional[dict]:
        """Find an entity by content hash."""
        assert self.conn is not None
        row = self.conn.execute(
            "SELECT * FROM entities WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        return dict(row) if row else None

    def get_evidence_for_statement(self, statement_id: str) -> list[dict]:
        """Get all evidence links pointing to a statement."""
        assert self.conn is not None
        rows = self.conn.execute(
            "SELECT * FROM evidence_links WHERE statement_id = ?",
            (statement_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_citations_for_entity(self, entity_id: str) -> list[dict]:
        """Get all citations from a given entity."""
        assert self.conn is not None
        rows = self.conn.execute(
            "SELECT * FROM citations WHERE citing_entity_id = ?",
            (entity_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_disclosure_backlinks(self, disclosure_id: str) -> list[dict]:
        """Find all entities that cite a given disclosure."""
        assert self.conn is not None
        rows = self.conn.execute(
            "SELECT e.* FROM entities e JOIN citations c ON e.id = c.citing_entity_id WHERE c.disclosure_id = ?",
            (disclosure_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_bindings_for_paper(self, paper_id: str) -> list[dict]:
        """Get all bindings in a paper."""
        assert self.conn is not None
        rows = self.conn.execute(
            "SELECT * FROM paper_bindings WHERE paper_id = ?",
            (paper_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_architecture_variants(self, architecture_id: str) -> list[dict]:
        """Find all variants of an architecture."""
        assert self.conn is not None
        rows = self.conn.execute(
            """SELECT e.* FROM entities e
               JOIN architecture_variants v ON e.id = v.architecture_id
               WHERE v.variant_of = ?""",
            (architecture_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def search(self, query: str, entity_type: str | None = None) -> list[dict]:
        """Simple text search across entity names."""
        assert self.conn is not None
        if entity_type:
            rows = self.conn.execute(
                "SELECT * FROM entities WHERE name LIKE ? AND entity_type = ? AND archived = 0",
                (f"%{query}%", entity_type),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM entities WHERE name LIKE ? AND archived = 0",
                (f"%{query}%",),
            ).fetchall()
        return [dict(r) for r in rows]
