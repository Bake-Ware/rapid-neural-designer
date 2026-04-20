"""Domain models for the RND Platform (SDD §4).

All entities are dataclasses with to_dict()/from_dict() for JSON round-tripping.
IDs follow the pattern {type-prefix}-{uuid}.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, fields, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_id(prefix: str) -> str:
    """Generate a stable entity ID: {prefix}-{uuid4}."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def now_iso() -> str:
    """Current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _serialize(value: Any) -> Any:
    """Recursively convert a value for JSON serialization."""
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, list):
        return [_serialize(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    return value


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ThreadState(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"
    STALE = "stale"
    REOPENED = "reopened"


class ExperimentStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETE = "complete"
    ABANDONED = "abandoned"


class PaperStatus(str, Enum):
    DRAFT = "draft"
    REVIEW = "review"
    PUBLISHED = "published"
    RETRACTED = "retracted"


class EvidenceSign(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    IRRELEVANT = "irrelevant"


class EvidenceStrength(str, Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


class FindingResolution(str, Enum):
    CONFIRMS = "confirms"
    REFUTES = "refutes"
    COMPLICATES = "complicates"


class DisclosureType(str, Enum):
    CLAUDE_TRANSCRIPT = "claude-transcript"
    CHAT_LOG = "chat-log"
    PERSONAL_NOTE = "personal-note"
    MEETING_NOTES = "meeting-notes"
    VOICE_MEMO = "voice-memo"
    EMAIL = "email"
    ACADEMIC_PAPER = "academic-paper"
    WEB_CLIP = "web-clip"
    OTHER = "other"


class Visibility(str, Enum):
    TEAM_PRIVATE = "team-private"
    PUBLISHED = "published"
    PERSONAL = "personal"


class CitationRelevance(str, Enum):
    ORIGINATING = "originating"
    SUPPORTING = "supporting"
    CONTEXTUAL = "contextual"
    ADJACENT = "adjacent"


class ArtifactCategory(str, Enum):
    DEFINITION = "definition"
    OUTPUT = "output"
    PUBLICATION = "publication"


class SectionType(str, Enum):
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    METHOD = "method"
    RESULTS = "results"
    DISCUSSION = "discussion"
    RELATED_WORK = "related_work"
    OTHER = "other"


class BindingType(str, Enum):
    ARCHITECTURE = "architecture"
    EXPERIMENT = "experiment"
    FINDING = "finding"
    DISCLOSURE = "disclosure"
    EXTERNAL = "external"


# ---------------------------------------------------------------------------
# Sub-models (embedded in entities, not standalone)
# ---------------------------------------------------------------------------

@dataclass
class Citation:
    """A reference to a Disclosure or external source."""
    disclosure_id: Optional[str] = None
    external_type: Optional[str] = None  # arxiv, doi, url, book, other
    external_id: Optional[str] = None
    external_title: Optional[str] = None
    relevance: str = "contextual"
    span_start: Optional[int] = None
    span_end: Optional[int] = None

    def to_dict(self) -> dict:
        d = {}
        if self.disclosure_id:
            d["disclosure_id"] = self.disclosure_id
            d["relevance"] = self.relevance
            if self.span_start is not None:
                d["span"] = {"start": self.span_start, "end": self.span_end}
        else:
            d["external"] = {
                "type": self.external_type,
                "id": self.external_id,
                "title": self.external_title,
                "relevance": self.relevance,
            }
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Citation:
        if "disclosure_id" in data:
            span = data.get("span", {})
            return cls(
                disclosure_id=data["disclosure_id"],
                relevance=data.get("relevance", "contextual"),
                span_start=span.get("start"),
                span_end=span.get("end"),
            )
        ext = data.get("external", {})
        return cls(
            external_type=ext.get("type"),
            external_id=ext.get("id"),
            external_title=ext.get("title"),
            relevance=ext.get("relevance", "contextual"),
        )


@dataclass
class EvidenceLink:
    """Links an Experiment to a Statement with interpretation."""
    statement_id: str
    sign: EvidenceSign
    strength: EvidenceStrength
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "statement_id": self.statement_id,
            "sign": self.sign.value,
            "strength": self.strength.value,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> EvidenceLink:
        return cls(
            statement_id=data["statement_id"],
            sign=EvidenceSign(data["sign"]),
            strength=EvidenceStrength(data["strength"]),
            note=data.get("note", ""),
        )


@dataclass
class StatementResolution:
    """How a Finding resolves a Statement."""
    statement_id: str
    resolution: FindingResolution
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "statement_id": self.statement_id,
            "resolution": self.resolution.value,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> StatementResolution:
        return cls(
            statement_id=data["statement_id"],
            resolution=FindingResolution(data["resolution"]),
            note=data.get("note", ""),
        )


@dataclass
class ExperimentInputs:
    """Hash-anchored inputs for an Experiment."""
    architecture_ref: str
    architecture_hash: str
    config_hash: str = ""
    data_manifest_hash: str = ""
    runtime_hash: str = ""

    def to_dict(self) -> dict:
        return {
            "architecture_ref": self.architecture_ref,
            "architecture_hash": self.architecture_hash,
            "config_hash": self.config_hash,
            "data_manifest_hash": self.data_manifest_hash,
            "runtime_hash": self.runtime_hash,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ExperimentInputs:
        return cls(**{k: data.get(k, "") for k in
                      ["architecture_ref", "architecture_hash", "config_hash",
                       "data_manifest_hash", "runtime_hash"]})


@dataclass
class ObservedResults:
    """Observed outputs from an Experiment."""
    metrics: dict = field(default_factory=dict)
    artifact_refs: list[str] = field(default_factory=list)
    logs_ref: Optional[str] = None

    def to_dict(self) -> dict:
        d: dict = {"metrics": self.metrics, "artifact_refs": self.artifact_refs}
        if self.logs_ref:
            d["logs_ref"] = self.logs_ref
        return d

    @classmethod
    def from_dict(cls, data: dict) -> ObservedResults:
        return cls(
            metrics=data.get("metrics", {}),
            artifact_refs=data.get("artifact_refs", []),
            logs_ref=data.get("logs_ref"),
        )


@dataclass
class Binding:
    """Connects a span of paper prose to a research entity."""
    binding_type: BindingType
    entity_id: str
    span_start: int
    span_end: int
    relevance: str = ""

    def to_dict(self) -> dict:
        return {
            "type": self.binding_type.value,
            "id": self.entity_id,
            "span": {"start": self.span_start, "end": self.span_end},
            "relevance": self.relevance,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Binding:
        span = data.get("span", {})
        return cls(
            binding_type=BindingType(data["type"]),
            entity_id=data["id"],
            span_start=span.get("start", 0),
            span_end=span.get("end", 0),
            relevance=data.get("relevance", ""),
        )


@dataclass
class PaperSection:
    """A section of a Paper with content reference and bindings."""
    id: str
    section_type: SectionType
    title: str
    content_ref: str
    bindings: list[Binding] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.section_type.value,
            "title": self.title,
            "content_ref": self.content_ref,
            "bindings": [b.to_dict() for b in self.bindings],
        }

    @classmethod
    def from_dict(cls, data: dict) -> PaperSection:
        return cls(
            id=data["id"],
            section_type=SectionType(data["type"]),
            title=data["title"],
            content_ref=data["content_ref"],
            bindings=[Binding.from_dict(b) for b in data.get("bindings", [])],
        )


@dataclass
class Attachment:
    """A file attached to a Disclosure."""
    path: str
    content_hash: str

    def to_dict(self) -> dict:
        return {"path": self.path, "hash": self.content_hash}

    @classmethod
    def from_dict(cls, data: dict) -> Attachment:
        return cls(path=data["path"], content_hash=data["hash"])


@dataclass
class TeamMember:
    """A user's membership in a team, with history."""
    user_id: str
    joined_at: str
    left_at: Optional[str] = None
    role: str = "member"

    def to_dict(self) -> dict:
        d = {"user_id": self.user_id, "joined_at": self.joined_at, "role": self.role}
        if self.left_at:
            d["left_at"] = self.left_at
        return d

    @classmethod
    def from_dict(cls, data: dict) -> TeamMember:
        return cls(
            user_id=data["user_id"],
            joined_at=data["joined_at"],
            left_at=data.get("left_at"),
            role=data.get("role", "member"),
        )


# ---------------------------------------------------------------------------
# Entity models
# ---------------------------------------------------------------------------

@dataclass
class Team:
    id: str
    name: str
    created_at: str
    updated_at: str
    members: list[TeamMember] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "members": [m.to_dict() for m in self.members],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Team:
        return cls(
            id=data["id"],
            name=data["name"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            members=[TeamMember.from_dict(m) for m in data.get("members", [])],
        )

    @classmethod
    def create(cls, name: str, user_id: str) -> Team:
        now = now_iso()
        return cls(
            id=generate_id("team"),
            name=name,
            created_at=now,
            updated_at=now,
            members=[TeamMember(user_id=user_id, joined_at=now)],
        )


@dataclass
class Program:
    id: str
    team_id: str
    name: str
    description: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict:
        return _serialize(asdict(self))

    @classmethod
    def from_dict(cls, data: dict) -> Program:
        return cls(**{f.name: data[f.name] for f in fields(cls)})

    @classmethod
    def create(cls, team_id: str, name: str, description: str = "") -> Program:
        now = now_iso()
        return cls(
            id=generate_id("prog"),
            team_id=team_id,
            name=name,
            description=description,
            created_at=now,
            updated_at=now,
        )


@dataclass
class Project:
    id: str
    program_id: str
    name: str
    description: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict:
        return _serialize(asdict(self))

    @classmethod
    def from_dict(cls, data: dict) -> Project:
        return cls(**{f.name: data[f.name] for f in fields(cls)})

    @classmethod
    def create(cls, program_id: str, name: str, description: str = "") -> Project:
        now = now_iso()
        return cls(
            id=generate_id("proj"),
            program_id=program_id,
            name=name,
            description=description,
            created_at=now,
            updated_at=now,
        )


@dataclass
class Thread:
    id: str
    project_id: str
    question: str
    resolution_criterion: str
    state: ThreadState
    created_at: str
    updated_at: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "question": self.question,
            "resolution_criterion": self.resolution_criterion,
            "state": self.state.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Thread:
        return cls(
            id=data["id"],
            project_id=data["project_id"],
            question=data["question"],
            resolution_criterion=data["resolution_criterion"],
            state=ThreadState(data["state"]),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )

    @classmethod
    def create(cls, project_id: str, question: str,
               resolution_criterion: str = "") -> Thread:
        now = now_iso()
        return cls(
            id=generate_id("thread"),
            project_id=project_id,
            question=question,
            resolution_criterion=resolution_criterion,
            state=ThreadState.OPEN,
            created_at=now,
            updated_at=now,
        )


@dataclass
class Statement:
    id: str
    thread_id: str
    hypothesis: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict:
        return _serialize(asdict(self))

    @classmethod
    def from_dict(cls, data: dict) -> Statement:
        return cls(**{f.name: data[f.name] for f in fields(cls)})

    @classmethod
    def create(cls, thread_id: str, hypothesis: str) -> Statement:
        now = now_iso()
        return cls(
            id=generate_id("stmt"),
            thread_id=thread_id,
            hypothesis=hypothesis,
            created_at=now,
            updated_at=now,
        )


@dataclass
class Experiment:
    id: str
    thread_id: str
    created_by: str
    status: ExperimentStatus
    inputs: ExperimentInputs
    method: dict
    hypothesis: str
    expected: str
    assumptions: list[str]
    observed: Optional[ObservedResults]
    interpretation: Optional[str]
    evidence: list[EvidenceLink]
    citations: list[Citation]
    created_at: str
    updated_at: str
    imported: bool = False

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "thread_id": self.thread_id,
            "created_by": self.created_by,
            "status": self.status.value,
            "inputs": self.inputs.to_dict(),
            "method": self.method,
            "hypothesis": self.hypothesis,
            "expected": self.expected,
            "assumptions": self.assumptions,
            "interpretation": self.interpretation,
            "evidence": [e.to_dict() for e in self.evidence],
            "citations": [c.to_dict() for c in self.citations],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "imported": self.imported,
        }
        if self.observed:
            d["observed"] = self.observed.to_dict()
        else:
            d["observed"] = None
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Experiment:
        observed = None
        if data.get("observed"):
            observed = ObservedResults.from_dict(data["observed"])
        return cls(
            id=data["id"],
            thread_id=data["thread_id"],
            created_by=data["created_by"],
            status=ExperimentStatus(data["status"]),
            inputs=ExperimentInputs.from_dict(data["inputs"]),
            method=data.get("method", {}),
            hypothesis=data.get("hypothesis", ""),
            expected=data.get("expected", ""),
            assumptions=data.get("assumptions", []),
            observed=observed,
            interpretation=data.get("interpretation"),
            evidence=[EvidenceLink.from_dict(e) for e in data.get("evidence", [])],
            citations=[Citation.from_dict(c) for c in data.get("citations", [])],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            imported=data.get("imported", False),
        )

    @classmethod
    def create(cls, thread_id: str, created_by: str,
               inputs: ExperimentInputs, hypothesis: str = "",
               expected: str = "", method: dict | None = None,
               imported: bool = False) -> Experiment:
        now = now_iso()
        return cls(
            id=generate_id("exp"),
            thread_id=thread_id,
            created_by=created_by,
            status=ExperimentStatus.PLANNED,
            inputs=inputs,
            method=method or {},
            hypothesis=hypothesis,
            expected=expected,
            assumptions=[],
            observed=None,
            interpretation=None,
            evidence=[],
            citations=[],
            created_at=now,
            updated_at=now,
            imported=imported,
        )


@dataclass
class Finding:
    id: str
    thread_id: str
    summary: str
    reasoning: str
    statement_resolutions: list[StatementResolution]
    experiment_refs: list[str]
    citations: list[Citation]
    created_at: str
    updated_at: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "summary": self.summary,
            "reasoning": self.reasoning,
            "statement_resolutions": [s.to_dict() for s in self.statement_resolutions],
            "experiment_refs": self.experiment_refs,
            "citations": [c.to_dict() for c in self.citations],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Finding:
        return cls(
            id=data["id"],
            thread_id=data["thread_id"],
            summary=data["summary"],
            reasoning=data["reasoning"],
            statement_resolutions=[StatementResolution.from_dict(s)
                                   for s in data.get("statement_resolutions", [])],
            experiment_refs=data.get("experiment_refs", []),
            citations=[Citation.from_dict(c) for c in data.get("citations", [])],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )

    @classmethod
    def create(cls, thread_id: str, summary: str, reasoning: str,
               statement_resolutions: list[StatementResolution] | None = None,
               experiment_refs: list[str] | None = None) -> Finding:
        now = now_iso()
        return cls(
            id=generate_id("find"),
            thread_id=thread_id,
            summary=summary,
            reasoning=reasoning,
            statement_resolutions=statement_resolutions or [],
            experiment_refs=experiment_refs or [],
            citations=[],
            created_at=now,
            updated_at=now,
        )


@dataclass
class Architecture:
    id: str
    name: str
    content: dict
    content_hash: str
    created_at: str
    updated_at: str
    variant_of: Optional[str] = None
    archived: bool = False

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "name": self.name,
            "content": self.content,
            "content_hash": self.content_hash,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "archived": self.archived,
        }
        if self.variant_of:
            d["variant_of"] = self.variant_of
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Architecture:
        return cls(
            id=data["id"],
            name=data["name"],
            content=data["content"],
            content_hash=data["content_hash"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            variant_of=data.get("variant_of"),
            archived=data.get("archived", False),
        )

    @classmethod
    def create(cls, name: str, content: dict, content_hash: str,
               variant_of: str | None = None) -> Architecture:
        now = now_iso()
        return cls(
            id=generate_id("arch"),
            name=name,
            content=content,
            content_hash=content_hash,
            created_at=now,
            updated_at=now,
            variant_of=variant_of,
        )


@dataclass
class Paper:
    id: str
    title: str
    authors: list[str]
    status: PaperStatus
    target: str
    program_id: str
    project_ids: list[str]
    thread_ids: list[str]
    sections: list[PaperSection]
    metadata: dict
    created_at: str
    updated_at: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "authors": self.authors,
            "status": self.status.value,
            "target": self.target,
            "scope": {
                "program_id": self.program_id,
                "project_ids": self.project_ids,
                "thread_ids": self.thread_ids,
            },
            "sections": [s.to_dict() for s in self.sections],
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Paper:
        scope = data.get("scope", {})
        return cls(
            id=data["id"],
            title=data["title"],
            authors=data.get("authors", []),
            status=PaperStatus(data["status"]),
            target=data.get("target", "arxiv"),
            program_id=scope.get("program_id", ""),
            project_ids=scope.get("project_ids", []),
            thread_ids=scope.get("thread_ids", []),
            sections=[PaperSection.from_dict(s) for s in data.get("sections", [])],
            metadata=data.get("metadata", {}),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )

    @classmethod
    def create(cls, title: str, authors: list[str], program_id: str,
               target: str = "arxiv") -> Paper:
        now = now_iso()
        return cls(
            id=generate_id("paper"),
            title=title,
            authors=authors,
            status=PaperStatus.DRAFT,
            target=target,
            program_id=program_id,
            project_ids=[],
            thread_ids=[],
            sections=[],
            metadata={},
            created_at=now,
            updated_at=now,
        )


@dataclass
class Disclosure:
    id: str
    title: str
    disclosure_type: DisclosureType
    created_by: str
    content_ref: str
    tags: list[str]
    visibility: Visibility
    source_url: Optional[str]
    source_metadata: dict
    attachments: list[Attachment]
    created_at: str
    updated_at: str

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "title": self.title,
            "type": self.disclosure_type.value,
            "created_by": self.created_by,
            "content_ref": self.content_ref,
            "tags": self.tags,
            "visibility": self.visibility.value,
            "source": {
                "url": self.source_url,
                "metadata": self.source_metadata,
            },
            "attachments": [a.to_dict() for a in self.attachments],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Disclosure:
        source = data.get("source", {})
        return cls(
            id=data["id"],
            title=data["title"],
            disclosure_type=DisclosureType(data["type"]),
            created_by=data["created_by"],
            content_ref=data["content_ref"],
            tags=data.get("tags", []),
            visibility=Visibility(data.get("visibility", "team-private")),
            source_url=source.get("url"),
            source_metadata=source.get("metadata", {}),
            attachments=[Attachment.from_dict(a) for a in data.get("attachments", [])],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )

    @classmethod
    def create(cls, title: str, disclosure_type: DisclosureType,
               created_by: str, content: str = "",
               tags: list[str] | None = None) -> Disclosure:
        now = now_iso()
        disc_id = generate_id("disc")
        return cls(
            id=disc_id,
            title=title,
            disclosure_type=disclosure_type,
            created_by=created_by,
            content_ref=f"disclosures/{disc_id}/content.md",
            tags=tags or [],
            visibility=Visibility.TEAM_PRIVATE,
            source_url=None,
            source_metadata={},
            attachments=[],
            created_at=now,
            updated_at=now,
        )


@dataclass
class Artifact:
    id: str
    parent_type: str
    parent_id: str
    category: ArtifactCategory
    name: str
    path: str
    content_hash: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "parent_type": self.parent_type,
            "parent_id": self.parent_id,
            "category": self.category.value,
            "name": self.name,
            "path": self.path,
            "content_hash": self.content_hash,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Artifact:
        return cls(
            id=data["id"],
            parent_type=data["parent_type"],
            parent_id=data["parent_id"],
            category=ArtifactCategory(data["category"]),
            name=data["name"],
            path=data["path"],
            content_hash=data["content_hash"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )

    @classmethod
    def create(cls, parent_type: str, parent_id: str,
               category: ArtifactCategory, name: str,
               path: str, content_hash: str) -> Artifact:
        now = now_iso()
        return cls(
            id=generate_id("art"),
            parent_type=parent_type,
            parent_id=parent_id,
            category=category,
            name=name,
            path=path,
            content_hash=content_hash,
            created_at=now,
            updated_at=now,
        )


# ---------------------------------------------------------------------------
# Entity type registry (for generic operations)
# ---------------------------------------------------------------------------

ENTITY_TYPES: dict[str, type] = {
    "team": Team,
    "program": Program,
    "project": Project,
    "thread": Thread,
    "statement": Statement,
    "experiment": Experiment,
    "finding": Finding,
    "architecture": Architecture,
    "paper": Paper,
    "disclosure": Disclosure,
    "artifact": Artifact,
}

ENTITY_PREFIX: dict[str, str] = {
    "team": "team",
    "program": "prog",
    "project": "proj",
    "thread": "thread",
    "statement": "stmt",
    "experiment": "exp",
    "finding": "find",
    "architecture": "arch",
    "paper": "paper",
    "disclosure": "disc",
    "artifact": "art",
}
