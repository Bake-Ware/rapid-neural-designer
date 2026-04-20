# SDD: RND Platform Expansion

**Author:** Bake
**Status:** Draft v1
**Date:** 2026-04-20
**Audience:** Bake + agentic coders implementing the system

---

## 1. Overview

This SDD specifies the expansion of RND (Rapid Neural Designer) from a visual neural network editor into a full research program management platform. The expanded system — provisionally called **RND Platform** — keeps the visual architecture editor as one entry point, and adds a research domain model, git-backed persistence, experiment tracking, paper authoring with binding validation, a disclosure reference system, and automation surfaces (REST / MCP / webhooks) for agent orchestration.

The platform's core value proposition is **grounded research**: every claim in every paper traces back through binding references to the architecture and experiments that support it. Reproducibility is structural, not aspirational. Audit is mechanical.

This document is organized into three concentric scopes:

- **Section 3** — Current RND state (what exists).
- **Sections 4–9** — Full platform target (the whole buffalo).
- **Section 10** — MVP scope (what ships first).
- **Section 11** — Post-MVP roadmap.

---

## 2. Goals and Non-Goals

### Goals

- Provide a structured research domain model that distinguishes questions, hypotheses, evidence, and conclusions as first-class objects.
- Anchor all research claims cryptographically to the architectures and experiments that support them.
- Version every research artifact through git, with the user's own git backend (self-hosted, GitHub, GitLab, etc.) as the substrate.
- Enable automation via structured APIs so external agents (Hermes, Claude Code, Gemini) can operate on research objects without hallucination.
- Produce publication artifacts (arxiv preprints, model cards, reproducibility releases) that are mechanically derivable from the structured data.
- Keep customer data in customer infrastructure — the platform is a tool, not a SaaS that holds research.

### Non-Goals

- The platform does not train or run neural networks itself beyond what RND's existing browser-based interpreter supports. Heavy compute is dispatched to the user's infrastructure.
- The platform does not host or re-host external literature. It references arxiv / DOI / URL citations without mirroring content.
- The platform does not replace peer review. Skeptic-style audits validate internal consistency, not scientific correctness.
- The platform does not enforce research methodology. Researchers decide what hypotheses to test; the system only structures the record.

---

## 3. Current RND State

This subsystem exists and is foundational. The expansion adds to it; nothing here is being rewritten.

### 3.1 Capabilities

- Visual graph-based editor for neural network architectures.
- Modular building blocks (layers, attention blocks, SSM units, custom components).
- Architectures serialized as JSON.
- Interpreter translates JSON to canned Python (primary) and C# (secondary).
- In-browser execution and validation of generated code against small reference inputs.
- Hierarchical IR as an internal representation between the visual graph and the generated code.

### 3.2 Canonical Representation

The JSON serialization of a model architecture is the canonical representation. All downstream subsystems (experiment tracking, paper authoring, diagram generation) treat this JSON as ground truth. Anti-hallucination guarantees for agentic systems depend on this — agents read the JSON and write about *that*, rather than paraphrasing from context.

### 3.3 Gaps the Expansion Addresses

- No notion of experiments, findings, or papers — only architectures.
- No persistence model beyond file export / save.
- No cross-architecture relationships (variants, ablation families, lineage).
- No external API; everything happens inside the editor UI.
- No integration path for automation or agent orchestration.

---

## 4. Domain Model

This is the core taxonomy. Every entity has a stable ID, a created_at, a last_modified_at, and is owned by exactly one Team (except cross-cutting Disclosures, scoped below). Relationships are explicit; access control descends from the Team boundary.

### 4.1 Hierarchy

```
Organization (billing, SSO, institutional identity)
└── Users (individuals, members of 1+ organizations)

Team (body-of-research identity, composed of users from possibly multiple orgs)
├── Disclosures (wiki-style reference pages, cited by research objects)
└── Program (long-lived research direction)
    └── Project (time-bounded investigation)
        └── Thread (answerable question / testable assumption)
            ├── Statement (candidate hypothesis / claim under investigation)
            ├── Experiment (specific run producing evidence)
            └── Finding (resolution of the thread, paper-spine)
                └── Artifacts (model weights, architecture JSON, papers, tags, etc.)
```

### 4.2 Entity Definitions

**Organization** — Billing and SSO boundary. Users authenticate through an Organization. Organizations do not own research directly; they own *seats* that Users occupy and fund *Teams* that their Users participate in.

**User** — An individual with credentials. A User belongs to 1+ Organizations and participates in 0+ Teams. The User is the authorship unit for attribution.

**Team** — A body-of-research identity. A Team is the set of Users working together on a coherent research program's lifetime. Team composition is recorded historically; when composition materially changes, a Team Fork creates a new Team that inherits or takes over the Program (see §4.6). Authors-of-record on published artifacts are the Team membership at the time of publication.

**Program** — A long-lived research direction owned by one Team. Multi-year scope. Persists as a git branch that rarely merges. Examples: MESSY BOI, ADREAN, DROGA lineages.

**Project** — A time-bounded investigation within a Program. Typically 6–18 months. Tied to a specific paper or small cluster of papers. Exists as a git branch off its Program.

**Thread** — An answerable question or testable assumption that the thread is intended to resolve. Has lifecycle state: `open`, `resolved`, `stale`, `reopened`. A Thread is not a folder; it is a well-formed research question with a resolution criterion. Examples: "Does multi-tier recurrence transfer across task families?", "Does loop-index RoPE improve phase differentiation?"

**Statement** — A candidate answer or hypothesis within a Thread. Accumulates evidence (from Experiments) with a sign: `supports`, `contradicts`, `irrelevant`. Multiple Statements can coexist within a Thread as competing hypotheses.

**Experiment** — A specific execution (or batched execution-set) of an Architecture with specific inputs, producing specific outputs. Anchored to an Architecture by content hash. Has status: `planned`, `running`, `complete`, `abandoned`. Produces results that attach as evidence to Statements.

**Finding** — A consolidated belief-update that resolves a Thread. References the Statement(s) it confirms or refutes, the Experiments that support it, and the reasoning that connects them. A Finding transitions a Thread from `open` to `resolved`. Findings are the paper-spine — papers are organized around resolved Threads and their Findings.

**Artifact** — A produced object attached to Experiments, Findings, Projects, or Programs. Categorized by lifecycle:

- *Definition artifacts*: architecture JSON, configs, dataset manifests, tags. Small, diffable, git-native, hash-anchored.
- *Output artifacts*: weights, checkpoints, loss curves, generated samples, figures. Often binary, usually offloaded to a configured artifact store.
- *Publication artifacts*: papers (PDF + source), posters, model cards, released models on HF, GitHub releases. Outward-facing, may have DOIs or external URIs.

**Disclosure** — A wiki-style page containing raw research context: conversation transcripts, meeting notes, personal notes, imported external content. Disclosures exist independently of the hierarchy. Research objects cite Disclosures; Disclosures do not know who cites them. See §8.

### 4.3 Relationships

| Relation | Cardinality | Notes |
|---|---|---|
| Organization → User | N:N | Users can belong to multiple orgs |
| Team → User | N:N (with membership history) | Historical record preserved for attribution |
| Team → Program | 1:N | Programs belong to exactly one Team at a time |
| Program → Project | 1:N | |
| Project → Thread | 1:N | |
| Thread → Statement | 1:N | |
| Thread → Experiment | 1:N | |
| Thread → Finding | 0:N | A Thread may have zero (unresolved) or more (if reopened) Findings |
| Statement ↔ Experiment | N:N with `sign` | Evidence relation: supports / contradicts / irrelevant |
| Finding → Statement | N:N with `resolution` | confirms / refutes / complicates |
| Finding → Experiment | N:N | Supporting experiments |
| Experiment → Architecture | N:1 (by hash) | Architecture identified by content hash, not file path |
| Architecture ↔ Architecture | N:N (variant relation with diff) | Parent / variant / ablation lineage |
| Artifact → parent entity | N:1 | Artifacts attach at Experiment, Finding, Project, or Program |
| Research object → Disclosure | N:N (citation) | With optional span and relevance annotation |
| Research object → External reference | N:N (citation) | arxiv / DOI / URL with type |

### 4.4 Identity and Stability

Every entity has a stable permanent ID of the form `{type-prefix}-{uuid}` (e.g., `thread-a3f2...`, `disc-b71c...`). IDs never change across renames, moves, or restructuring. All cross-references use IDs; all human-visible names are metadata.

### 4.5 Hash Anchoring

Architectures are identified both by stable ID and by *content hash* of their canonical JSON representation (see §5.5 for canonical serialization). Experiments bind to the content hash at the time of execution. This provides cryptographic proof that a given experiment ran against a specific architecture state, independent of subsequent edits.

When an architecture is edited, the content hash changes. Prior experiments retain their old hash reference and remain valid records; future experiments get the new hash. The system can compute architecture lineage by walking hashes through the git history.

Artifacts (weights, datasets, configs) also carry content hashes, enabling storage-backend migration without breaking experiment records.

### 4.6 Team Forks

When a Team's composition changes materially (new institution joins, key members leave, project focus shifts), any Team member may propose a Team Fork. A Fork creates a new Team that either:

- **Inherits** the Program (original Team retains the Program, new Team starts fresh), or
- **Takes over** the Program (Program is transferred to the new Team, original is archived).

Fork events are recorded with timestamp, initiator, reason, and member transitions. Historical attribution queries respect Fork history — authors-of-record on a paper are the Team membership at the time of the paper's publication, not the current Team.

---

## 5. Persistence and Versioning

### 5.1 Git as First-Class Substrate

All research artifacts are stored as files in a git repository. Customers bring their own git backend (GitHub, GitLab, Gitea, self-hosted). The platform reads and writes files in a known format; nothing lives in a proprietary database that the customer cannot take with them.

The repository structure follows a convention; the platform does not require a specific remote or hosting choice. Repos can be public, private, or mirrored.

### 5.2 Repository Layout

```
/
├── team.json                    # Team metadata
├── programs/
│   └── {program-id}/
│       ├── program.json
│       ├── README.md
│       └── projects/
│           └── {project-id}/
│               ├── project.json
│               ├── README.md
│               └── threads/
│                   └── {thread-id}/
│                       ├── thread.json
│                       ├── statements/
│                       │   └── {statement-id}.json
│                       ├── experiments/
│                       │   └── {experiment-id}.json
│                       └── findings/
│                           └── {finding-id}.json
├── architectures/
│   └── {architecture-id}.json   # Content-hash-anchored
├── disclosures/
│   └── {disclosure-id}/
│       ├── disclosure.json      # Metadata
│       └── content.md           # Wiki page content
├── papers/
│   └── {paper-id}/
│       ├── paper.json           # Structured paper with bindings
│       └── content.md           # Narrative prose
├── artifacts/                   # Small artifacts only; large ones offloaded
│   └── {artifact-id}/
└── .rnd/                        # Platform-managed
    ├── config.yaml              # Artifact store, publishing targets, etc.
    ├── index.sqlite             # Derived index (gitignored, rebuildable)
    └── hooks/                   # Git hooks for automation
```

### 5.3 Branch Model

**Three-tier branching reflects the research hierarchy:**

- **Lineage / Program branches** — long-lived, rarely merged. One per Program. Effectively trunk branches for that research direction.
- **Project branches** — medium-lived, branched off a Program branch. Accumulate Threads, Statements, Experiments, Findings. Merged back to Program when the Project concludes.
- **Thread / Experiment branches** — short-lived, branched off a Project branch for specific investigations. Merged back to Project when the Thread resolves.

**Lifecycle operations are user-triggered verbs, not automated rules:**

- `save-as-result` — Tags the current commit with a user-provided label. Lightweight bookmark. No branching overhead.
- `wrap-up-experiment` — Merges the current experiment branch into a target parent (Project). User picks target and whether to squash or preserve history. Default: preserve history. Red-button override: squash-merge for messy histories.
- `promote-to-lineage` — Merges a Project branch into its Program. User chooses squash or preserve; default for Project → Program is typically squash (synthesis commit summarizing the project's findings) because step-by-step experiment history is less valuable at the Program level.
- `fork-team` — Creates a new Team inheriting or taking over the current Program (see §4.6).

Users can always bypass these verbs and use raw git. The verbs provide structured convenience, not a cage.

### 5.4 Commits vs. Experiments

**A commit is not an experiment.** Commits version the state of the research program at a moment in time. Experiments are first-class JSON objects inside commits, anchored to architecture hashes.

This separation matters for:

- Running the same architecture with multiple seeds (multiple experiment records in one commit, not N empty commits).
- Editing a paper without it appearing as a new experiment.
- Mid-experiment checkpointing without polluting history.
- Correctly representing that the experiment's *inputs* are the architecture hash and config, not the commit hash of the repo.

Experiments carry a `code_version` field (a separate hash of the RND interpreter version or environment) so that reproducibility includes runtime, not just architecture.

### 5.5 Canonical JSON Serialization

To make content hashing deterministic:

- Keys sorted alphabetically at all nesting levels.
- No trailing whitespace, no trailing newline.
- UTF-8 encoded.
- Numbers in shortest round-trip representation.
- No insignificant whitespace between tokens.

All hashing uses SHA-256 over the canonical serialization. A `canonicalize` utility is provided and must be used before hashing or diffing.

### 5.6 Derived Index

A SQLite database (`.rnd/index.sqlite`) provides fast queries over relations. It is **derived**, not authoritative — the JSON files are the source of truth. The index is gitignored and rebuilt from files on demand via a `rnd index rebuild` command. This preserves the portable, self-contained nature of the repo while giving interactive performance for graph queries.

---

## 6. Experiment Engine

### 6.1 Experiment Record Schema

```yaml
id: exp-{uuid}
thread_id: thread-{uuid}
created_at: ISO8601
created_by: user-{uuid}
status: planned | running | complete | abandoned

inputs:
  architecture_ref: architectures/{id}.json
  architecture_hash: sha256:{hex}
  data_manifest_hash: sha256:{hex}
  config_hash: sha256:{hex}
  runtime_hash: sha256:{hex}

method:
  hyperparameters: { ... }
  seed: int
  data_split: string
  # ... arbitrary structured config

hypothesis: string             # What you expect to observe
expected: string | metrics     # Predicted outcome
assumptions: [string]          # Claims being treated as given

observed:
  metrics: { ... }
  artifact_refs: [artifact-{id}]
  logs_ref: artifact-{id} | null

interpretation: string         # What the result means
evidence:
  - statement_id: stmt-{id}
    sign: supports | contradicts | irrelevant
    strength: weak | moderate | strong
    note: string

citations:
  - disclosure_id: disc-{id}
  - external: { type: arxiv, id: "...", relevance: ... }
```

### 6.2 Execution Dispatch

Execution happens outside the platform's process space. The Experiment Engine maintains a dispatcher abstraction:

- **Local execution** — the user runs the experiment locally (via RND's browser interpreter or a CLI tool) and reports results back.
- **Remote dispatch** — the platform sends the experiment spec to a configured runner (SSH target, Kubernetes job, serverless function). Runner responsibility includes hash verification of inputs before execution.
- **Batched execution** — a single experiment record may specify N seeds or N config variants; the dispatcher fans out, collects, and rolls up results under one record.

The dispatcher is pluggable. Default runners: local subprocess, SSH, HTTP POST to a user-configured endpoint.

### 6.3 Result Collection

Running experiments stream results back to the Experiment record incrementally. The platform supports:

- Structured metrics (numeric, logged per step).
- Artifact uploads (referenced via content hash, routed to configured artifact store).
- Log streams (stored as artifact, referenced in the record).

A running experiment is a live document. On completion, the record is finalized: status transitions to `complete`, all hashes verified, no further updates except via explicit re-open.

### 6.4 Evidence Linking

When results are in, the user (or an Archivist agent) attaches the experiment's evidence to Statements. This is an explicit step because it's where interpretation enters — raw metrics don't automatically "support" a hypothesis; a researcher decides they do. The evidence link carries a sign, a strength estimate, and a note.

### 6.5 Ablation and Variant Management

Architectures may be linked by variant relations (parent → variant, with diff). An Ablation Matrix query walks the variant graph and enumerates all experiments on related architectures, producing a table suitable for a paper's ablation section.

Batched experiments on a single architecture with varying hyperparameters produce a seed/config matrix within one record, rather than requiring separate architecture variants.

---

## 7. Paper Authoring

### 7.1 Paper Record Schema

A Paper is a structured document, not freeform prose:

```yaml
id: paper-{uuid}
title: string
authors: [user-{id}]         # Snapshot at draft time, editable until published
status: draft | review | published | retracted
target: arxiv | conference | journal | internal_report

scope:
  program_id: prog-{id}
  project_ids: [proj-{id}]
  thread_ids: [thread-{id}]  # Resolved threads that define paper scope

sections:
  - id: section-{id}
    type: abstract | introduction | method | results | discussion | related_work | other
    title: string
    content_ref: papers/{paper-id}/sections/{section-id}.md
    bindings:
      - type: architecture | experiment | finding | disclosure | external
        id: {entity-id}
        span: { start: int, end: int }   # Character range in content
        relevance: string                 # "method-section reference", "key result", etc.

metadata:
  arxiv_id: string | null
  doi: string | null
  github_release_tag: string | null
  hf_model_ids: [string]
```

Each section's content is a markdown file. Bindings connect spans of prose to structured entities. The binding is the anchor — when prose makes a claim, the binding says *what backs the claim*.

### 7.2 Binding Validation (Skeptic Audit)

The Skeptic skill validates bindings mechanically:

- **False binding** — prose asserts X; the bound entity does not support X. Example: paper says "tier ratio 7:1 achieves 89% accuracy"; bound experiment reports 82%. Flagged.
- **Missing binding** — prose makes a substantive claim with no binding. Flagged.
- **Weak binding** — prose asserts strong confidence; bound evidence is marginal. Flagged.
- **Stale binding** — bound architecture hash has been superseded since the binding was created. Flagged.
- **Orphaned binding** — bound entity no longer exists. Flagged.

Skeptic produces a structured report with per-claim validation results. The report can be viewed as PR-style comments in the Paper authoring UI or exported as a JSON structure for external tools.

### 7.3 Drafter Skill

Drafter generates paper prose from structured research data:

- **Input**: scope (which Threads / Findings), target format (arxiv / conference / internal), section-level instructions.
- **Process**: reads architecture JSON, experiment records, finding records, cited disclosures, external references. Generates prose per section with bindings populated.
- **Output**: updated Paper record with section content and bindings. Drafter never writes claims without bindings.

Drafter is implemented as a Claude Code skill (subprocess invocation, subscription-backed billing). Inputs are structured JSON; outputs are structured JSON. See §9.3.

### 7.4 Rendering

The Paper record plus section markdown plus bindings can be rendered to:

- **arxiv LaTeX** — for preprint submission. Bindings become `\cite{}` entries (with internal disclosures rendered as personal-communication citations).
- **HTML** — for review UI and web publishing.
- **PDF** — via LaTeX rendering toolchain.
- **Markdown** — for direct consumption or conversion to other formats.

Rendering is deterministic from the structured data. The same Paper record always produces the same rendered output.

### 7.5 Reproducibility Release

Publishing a paper triggers a **Reproducibility Release**: a git tag (or GitHub release) pinning the exact state of all architectures, experiments, and findings cited by the paper. The release includes:

- The rendered paper (PDF + source).
- Links to arxiv, DOI, or other external IDs.
- A manifest of cited entity hashes.
- A reproducibility script (`reproduce.sh` or equivalent) that, given the release, re-runs the experiments.

Because everything is hash-anchored, a reader cloning the release can verify they have the exact state the paper was based on.

---

## 8. Disclosure System

### 8.1 Purpose

Disclosures are the wiki-style reference substrate for raw research context: conversation transcripts, notes, external literature, meeting records. They serve as cited references for structured research objects but exist independently of the research hierarchy.

### 8.2 Disclosure Record Schema

```yaml
id: disc-{uuid}
title: string
created_at: ISO8601
updated_at: ISO8601
created_by: user-{uuid}

type: claude-transcript | chat-log | personal-note | meeting-notes |
      voice-memo | email | academic-paper | web-clip | other

source:
  url: string | null                    # Original source if external
  metadata: { ... }                     # Participants, timestamps, original format

content_ref: disclosures/{id}/content.md
attachments:
  - path: disclosures/{id}/attachments/{filename}
    hash: sha256:{hex}

tags: [string]
visibility: team-private | published | personal
```

Disclosures are stored as a `disclosure.json` metadata file plus a `content.md` file plus optional attachments in the same directory.

### 8.3 Citation Model

Research objects (Statements, Experiments, Findings, Papers, etc.) carry a `citations` list. Each citation references either a Disclosure or an external source:

```yaml
citations:
  - disclosure_id: disc-{id}
    relevance: originating | supporting | contextual
    span: { start: int, end: int } | null
  - external:
      type: arxiv | doi | url | book | other
      id: string
      title: string
      relevance: originating | supporting | contextual | adjacent
```

Citations are asymmetric: the citing object knows about the Disclosure; the Disclosure does not know about its citers. Backlinks are computed by the derived index.

### 8.4 Ingestion Pipelines

Disclosures are created through multiple ingestion channels:

- **Claude transcript import** — paste URL or upload export; parsed into disclosure with turn structure preserved.
- **Slack export** — OAuth workspace, select channels/threads, incremental sync.
- **Markdown paste** — raw input via UI or API.
- **Email forward** — team-specific forwarding address; incoming emails become disclosures.
- **Voice memo upload** — audio file in, transcription out via Whisper or similar, both retained.
- **Meeting notes** — direct paste or Zoom/Meet/Teams integration.
- **Browser extension** — highlight text anywhere, send to team with optional attachment to a specific Project or Thread.

Ingestion is low-friction by design. Classification and tagging happen after ingestion, often by agents (Archivist).

### 8.5 Wiki-Style Linking

Disclosures can link to other Disclosures via wiki-style syntax (`[[disc-{id}]]` or `[[Disclosure Title]]`). These inter-disclosure links form a second graph on top of the citation graph, useful for navigating related context.

Backlinks from research objects (via citations) and from other Disclosures (via wiki links) are both surfaced in the UI when viewing a Disclosure.

### 8.6 Search and Discovery

Discovery is primary because Disclosures do not pre-commit to attachments:

- **Full-text search** over all Disclosure content accessible to the user's Teams.
- **Semantic search** via embeddings. Embeddings are generated on ingestion and re-generated on update.
- **Filtered queries** — by type, tag, date range, participants, citation count.

Archivist agents perform background suggestion: on new research object creation, search recent Disclosures for candidate citations and offer them to the user.

### 8.7 Visibility and Publishing

Disclosures have independent visibility from the research objects that cite them:

- **Team-private** (default) — visible only to Team members. Citations render as "internal communication" in published papers.
- **Published** — exposed as part of a paper's reproducibility release. Visible to anyone with paper access.
- **Personal** — visible only to the creator. Useful for personal notes the creator is not yet ready to share.

Changing visibility from team-private to published is an explicit action with confirmation, because it can inadvertently leak sensitive early-stage thinking.

---

## 9. API Surfaces and Automation

### 9.1 REST API

Resource-oriented, standard HTTP verbs, JSON request/response. OpenAPI spec auto-generated.

**Core endpoint families:**

- `/teams/{id}` — Team metadata, membership, fork history.
- `/programs/{id}` — Program CRUD, cross-project queries.
- `/projects/{id}` — Project CRUD, thread enumeration.
- `/threads/{id}` — Thread CRUD, state transitions, statement/experiment/finding lists.
- `/statements/{id}` — Statement CRUD, evidence relations.
- `/experiments/{id}` — Experiment CRUD, dispatch, status, results.
- `/findings/{id}` — Finding CRUD, thread resolution.
- `/architectures/{id}` — Architecture CRUD, variant queries, hash lookups, rendering.
- `/papers/{id}` — Paper CRUD, section management, binding validation, rendering, publishing.
- `/disclosures/{id}` — Disclosure CRUD, search, citation backlinks.
- `/artifacts/{id}` — Artifact upload, download, hash verification, publishing.

**Query endpoints:**

- `/search` — Unified search across entities.
- `/graph/query` — Graph queries over relations (ablation matrices, lineage traces, citation networks).

**Authentication:** token-based. Tokens scoped to specific Teams or Programs. Tokens have granular permissions (read / write / publish).

### 9.2 MCP Server

Exposes tools to LLM agents for reasoning about and operating on research entities. MCP tool descriptions include semantic guidance, not just parameter schemas, so agents can choose appropriate tools without hand-coded workflows.

**Tool surface:**

- `rnd_search_architectures` — Natural language search over architecture library.
- `rnd_read_architecture` — Fetch full JSON of a specific architecture.
- `rnd_diff_architectures` — Semantic comparison of two architectures.
- `rnd_render_diagram` — Generate publication-quality architecture diagram.
- `rnd_create_experiment` — Create experiment record with inputs and hypothesis.
- `rnd_run_experiment` — Dispatch experiment to execution backend.
- `rnd_query_experiments` — Filter experiments by criteria.
- `rnd_create_statement` — Add statement to a thread.
- `rnd_attach_evidence` — Link experiment to statement with sign and strength.
- `rnd_resolve_thread` — Create finding and transition thread to resolved.
- `rnd_audit_claim` — Validate a claim against its binding (Skeptic primitive).
- `rnd_generate_paper_section` — Given scope and section type, generate draft prose with bindings.
- `rnd_search_disclosures` — Full-text and semantic search over disclosures.
- `rnd_cite_disclosure` — Add citation to a research object.
- `rnd_publish_artifact` — Push a model or paper to configured external platforms.

The MCP server runs as a process alongside the user's RND installation. Agents (Hermes, Claude Code, Gemini) connect via standard MCP transport.

### 9.3 Skill Integration (Claude Code)

Paper authoring and auditing are implemented as Claude Code skills, invoked as subprocesses for subscription-backed billing:

- **Drafter skill** — `claude -p "[drafter skill invocation]" < input.json > output.json`
- **Skeptic skill** — same pattern, outputs structured critique.
- **Diagram-Maker skill** — same pattern, outputs SVG or TikZ.

Each skill has a SKILL.md defining its inputs, outputs, procedures. The platform does not embed LLMs directly; it orchestrates skill invocations.

### 9.4 Webhooks

Outbound events for reactive automation:

- `experiment.complete` — fires on experiment status transition to complete.
- `thread.resolved` — fires on thread resolution (finding created).
- `paper.draft.ready` — fires on paper moving to review status.
- `paper.audit.complete` — fires when Skeptic audit finishes.
- `artifact.published` — fires on external publication.
- `disclosure.ingested` — fires on new disclosure creation.

Webhook subscribers (Hermes, custom integrations, backup systems) register endpoints and event filters. Payloads include entity IDs and relevant metadata; subscribers fetch full records via REST as needed.

### 9.5 Agent Orchestration Pattern

The canonical orchestration pattern uses Hermes as the workflow engine, with the API surfaces as the operational substrate:

1. **Archivist agent** (Gemini-backed, always-on) watches for new Disclosures and research objects via webhooks. Tags, suggests citations, flags candidate findings.
2. **Priority Watcher** (Gemini-backed, weekly cron) runs Literature Mapper passes over candidate findings; flags candidates where adjacent work is closing in.
3. **Human decision point** — user reviews candidates and selects one for paper drafting.
4. **Drafter** (Claude Code skill) generates paper draft from selected Finding and its supporting entities.
5. **Skeptic** (Claude Code skill) audits the draft against its bindings.
6. **Diagram-Maker** (Claude Code skill) generates any diagrams the draft references.
7. **Human review** — user reviews draft and audit report.
8. **Publication** — user triggers publication; platform creates Reproducibility Release, pushes to arxiv / HF / GitHub as configured.

The platform provides the substrate; Hermes (or any equivalent orchestrator) drives the workflow.

---

## 10. MVP Scope

The MVP exists to produce **one real arxiv preprint from existing DROGA research**. Any feature not on the critical path to that outcome is out of scope for MVP.

### 10.1 In Scope for MVP

- Single-team, single-user operation.
- Domain model: Team (1), Program (1), Project (1+), Thread, Statement, Experiment, Finding, Architecture, Paper, Disclosure, Artifact.
- Git-backed persistence with the repository layout in §5.2.
- Canonical JSON serialization and SHA-256 hashing.
- Basic branch model (§5.3) with user-triggered lifecycle verbs.
- Experiment record schema and local execution dispatch. Remote dispatch deferred.
- Architecture import from existing RND JSON files.
- Paper record schema with bindings.
- Drafter skill (Claude Code subprocess).
- Skeptic skill (binding validation only — false / missing / stale).
- Disclosure ingestion: Claude transcript paste, markdown paste. Other pipelines deferred.
- Citation system connecting research objects to Disclosures and external references.
- Derived SQLite index with rebuild command.
- REST API for all entities (CRUD + search).
- Minimal MCP server exposing read-only tools + create_experiment, create_statement, create_finding, audit_claim.
- arxiv LaTeX rendering.
- Git tag-based Reproducibility Release (GitHub Release integration deferred).

### 10.2 Out of Scope for MVP

- Multi-team operation, organizations, SSO.
- Team Forks.
- Remote experiment dispatch beyond local subprocess.
- Slack / email / voice memo / browser extension ingestion.
- Hugging Face publishing.
- Webhooks beyond a minimal inbound payload format.
- Semantic search over Disclosures (full-text only).
- Ablation matrix auto-generation.
- Pluggable artifact stores (local filesystem only; git-LFS auto-config deferred).
- Program / Project status dashboards, cross-program synthesis.
- Plugin extension points.

### 10.3 MVP Success Criteria

- Bake can import existing DROGA architecture JSONs into the system.
- Bake can create experiment records for DROGA runs 1–9 with their observed results.
- Bake can create a Thread for "multi-tier recurrence transfer learning" with statements, link Run 6c as evidence, and create a Finding resolving the thread.
- Bake can paste relevant Claude conversation(s) as Disclosures.
- Drafter skill generates a paper draft from the resolved Thread with bindings.
- Skeptic skill audits the draft and produces a validation report.
- Bake can render the final paper as arxiv LaTeX.
- The output preprint can be submitted to arxiv.
- A Reproducibility Release git tag exists that pins all cited entities.

### 10.4 MVP Anti-Goals

- No attempt to make the UI production-polished.
- No multi-user concurrency handling.
- No performance optimization beyond "tolerable for single-user."
- No security hardening beyond basic token auth.
- No migration tooling from other research-tracking systems.

---

## 11. Post-MVP Roadmap

### 11.1 v1.5 — Publication and Collaboration

- Hugging Face publishing for model artifacts (with auto-generated model cards).
- GitHub Releases integration for Reproducibility Releases.
- Full webhook surface.
- Slack and email Disclosure ingestion.
- Semantic search over Disclosures.
- Pluggable artifact stores (S3, Azure Blob, git-LFS with auto-config).

### 11.2 v2 — Multi-User and Organizations

- Organizations and SSO (SAML, OIDC).
- Multi-user Teams with role-based permissions.
- Team Forks.
- Cross-team citation and sharing mechanisms.
- Remote experiment dispatch (Kubernetes, SSH, HTTP).
- Program and Project status dashboards.
- Ablation matrix auto-generation.

### 11.3 v2.5 — Platform Extensions

- Plugin extension points for custom skills, audit rules, publishing targets.
- Browser extension for Disclosure capture.
- Voice memo ingestion with Whisper transcription.
- Cross-program synthesis queries (for lab-tier customers).
- Institutional repository integrations.

### 11.4 v3 — Enterprise

- Advanced audit logs and compliance features.
- Custom branding and white-labeling.
- Enterprise support and SLAs.
- Managed hosting option (for customers who want SaaS despite the BYO-git default).

---

## 12. Open Design Questions

These require explicit decisions before MVP implementation begins. Each has a recommended default; the implementing agent should flag any deviation from the default.

1. **Canonical JSON library** — Which implementation provides deterministic serialization across the languages RND uses (current RND is likely TypeScript + Python)? *Default: implement canonical serializer per language; cross-verify hashes in integration tests.*

2. **Experiment execution on first import of DROGA data** — Existing DROGA runs are historical; they were not executed through this system. How are they represented? *Default: import as `complete` experiments with retrospective metadata; flag as `imported` rather than `executed_by_platform`.*

3. **Hash stability across RND interpreter versions** — If the interpreter changes code generation for the same architecture JSON, the `runtime_hash` changes. Does the `architecture_hash` also change? *Default: no — architecture_hash is of the architecture JSON, runtime_hash separately tracks interpreter version.*

4. **Drafter skill prompt versioning** — The Drafter skill will evolve. Should its version be recorded in the Paper? *Default: yes — add `drafter_version` to Paper metadata, allowing regeneration to use a specific skill version.*

5. **Citation ambiguity** — A research object can cite a Disclosure; a Paper can also cite the same Disclosure. Are these separate citation records or one shared? *Default: separate; each citing entity maintains its own citation with its own relevance annotation.*

6. **Thread lifecycle edge cases** — What happens when a Thread's resolving Finding is later contradicted by a new Experiment? *Default: the Thread enters `reopened` state; a new Finding can supersede the old one; old Finding is retained in history; papers based on the old Finding are flagged for potential retraction.*

7. **Architecture deletion** — If an Architecture is deleted but Experiments reference it by hash, what happens? *Default: architectures are never hard-deleted; archive state removes them from default views but preserves the content for hash-anchored references.*

8. **Team ownership of orphaned Programs** — If a Team is dissolved (all members leave), what happens to its Programs? *Default: Programs enter `orphaned` state; the platform admin can transfer them to a new Team; they remain read-only until transferred.*

---

## 13. Implementation Notes for Agentic Coders

### 13.1 Build Order

1. **Domain model and schemas** — Define JSON Schemas for every entity. This is the contract everything else depends on.
2. **Canonical serializer and hashing** — Implement and test before anything uses hashes.
3. **Git persistence layer** — File I/O, repository layout, gitignore management, hook installation.
4. **Derived index** — SQLite schema, rebuild command, incremental update on file changes.
5. **REST API** — Entity CRUD first; graph queries and search second.
6. **Architecture import from RND** — Bridge to existing RND JSON format.
7. **Experiment engine** — Record management, local dispatch, result collection.
8. **Disclosure system** — Record management, markdown ingestion, citation tracking.
9. **Paper authoring** — Record schema, binding management, rendering.
10. **MCP server** — Expose read-only tools first; write tools after REST stability.
11. **Drafter skill** — Claude Code subprocess integration, structured I/O contract.
12. **Skeptic skill** — Binding validation logic, structured critique format.
13. **arxiv LaTeX rendering** — Template-based rendering from Paper records.
14. **Reproducibility Release** — Git tag creation with manifest.

### 13.2 Testing Priorities

- **Hash determinism** — Same architecture JSON always produces the same hash across environments.
- **Canonical serialization round-trip** — Parse → serialize → parse produces identical structure.
- **Binding validation** — Known-good and known-bad paper drafts produce expected audit reports.
- **Import idempotency** — Importing the same RND JSON twice produces no duplicate entities.
- **Git operations** — Lifecycle verbs produce expected repository state (commits, branches, tags).
- **Index rebuild** — Rebuilding the derived index from files produces the same query results.

### 13.3 Anti-Hallucination Practices for Agent-Assisted Development

- Agents implementing this system must read `architectures/*.json` to understand architecture structure, not paraphrase from conversation context.
- Agents must read `thread.json` and related records to understand thread state, not infer from file paths.
- When generating code that manipulates entity records, agents must validate against JSON Schemas before writing.
- When generating paper prose, agents must populate bindings before writing claims. A claim without a binding is a bug.

### 13.4 Where to Resist Scope Creep

- Do not build a UI beyond minimum viable for MVP validation. CLI + API is acceptable; a React app is out of scope for MVP.
- Do not build orchestration inside the platform. Orchestration is Hermes's job; the platform exposes APIs.
- Do not build LLM capabilities into the platform itself. LLM work happens via skill invocation.
- Do not build features that require multi-user permissions before single-user works end-to-end.

---

## 14. Glossary

- **Architecture** — A neural network model design, serialized as JSON, visualizable in RND editor.
- **Binding** — A structured reference from a span of paper prose to a research entity.
- **Content hash** — SHA-256 over canonical JSON serialization; provides cryptographic proof of entity state.
- **Disclosure** — A wiki-style reference page containing raw research context (conversations, notes, literature).
- **Experiment** — A record of a specific execution of an architecture, anchored to architecture hash.
- **Finding** — A consolidated belief-update that resolves a Thread; the spine of a paper.
- **Lifecycle verb** — A user-triggered operation that performs structured git work (save-as-result, wrap-up-experiment, promote-to-lineage, fork-team).
- **Program** — A long-lived research direction. Highest research-content level below the Team.
- **Project** — A time-bounded investigation within a Program, typically paper-oriented.
- **Reproducibility Release** — A pinned snapshot of all entities cited by a published paper.
- **Statement** — A candidate hypothesis within a Thread; accumulates evidence.
- **Team** — A body-of-research identity; the set of Users working on a coherent research program's lifetime.
- **Thread** — An answerable question or testable assumption; scoped unit of investigation with lifecycle state.
