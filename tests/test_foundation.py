"""Foundation tests for RND Platform.

Covers: hash determinism, canonical serialization round-trip,
repo init/save/load, and index rebuild (SDD §13.2).
"""

import json
import tempfile
from pathlib import Path

from rnd.canonical import canonicalize, content_hash
from rnd.index import DerivedIndex
from rnd.models import (
    Architecture,
    Citation,
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
    Program,
    Project,
    Statement,
    StatementResolution,
    Team,
    Thread,
    ThreadState,
    Visibility,
)
from rnd.repo import RNDRepo


# ------------------------------------------------------------------
# Canonical serialization
# ------------------------------------------------------------------

def test_canonicalize_sorts_keys():
    obj = {"z": 1, "a": 2, "m": 3}
    result = canonicalize(obj)
    assert result == '{"a":2,"m":3,"z":1}'


def test_canonicalize_nested_sort():
    obj = {"b": {"z": 1, "a": 2}, "a": 0}
    result = canonicalize(obj)
    assert result == '{"a":0,"b":{"a":2,"z":1}}'


def test_canonicalize_no_whitespace():
    obj = {"key": "value", "num": 42}
    result = canonicalize(obj)
    assert " " not in result.replace('"key"', "").replace('"value"', "")


def test_canonicalize_unicode():
    obj = {"name": "Ünïcödé"}
    result = canonicalize(obj)
    assert "Ünïcödé" in result  # ensure_ascii=False


def test_content_hash_deterministic():
    obj = {"architecture": "transformer", "layers": 12}
    h1 = content_hash(obj)
    h2 = content_hash(obj)
    assert h1 == h2
    assert h1.startswith("sha256:")


def test_content_hash_key_order_invariant():
    """Same data, different insertion order → same hash."""
    obj1 = {"a": 1, "b": 2}
    obj2 = {"b": 2, "a": 1}
    assert content_hash(obj1) == content_hash(obj2)


def test_content_hash_differs_on_change():
    obj1 = {"layers": 12}
    obj2 = {"layers": 13}
    assert content_hash(obj1) != content_hash(obj2)


# ------------------------------------------------------------------
# Model round-trip
# ------------------------------------------------------------------

def test_team_round_trip():
    team = Team.create(name="Test Team", user_id="user-abc123")
    data = team.to_dict()
    restored = Team.from_dict(data)
    assert restored.id == team.id
    assert restored.name == team.name
    assert len(restored.members) == 1
    assert restored.members[0].user_id == "user-abc123"


def test_thread_round_trip():
    thread = Thread.create(project_id="proj-abc123",
                           question="Does X improve Y?",
                           resolution_criterion="p < 0.05")
    data = thread.to_dict()
    restored = Thread.from_dict(data)
    assert restored.question == thread.question
    assert restored.state == ThreadState.OPEN


def test_experiment_round_trip():
    inputs = ExperimentInputs(
        architecture_ref="architectures/arch-abc.json",
        architecture_hash="sha256:deadbeef",
        config_hash="sha256:cafebabe",
    )
    exp = Experiment.create(
        thread_id="thread-abc123",
        created_by="user-abc123",
        inputs=inputs,
        hypothesis="X improves Y",
        imported=True,
    )
    data = exp.to_dict()
    restored = Experiment.from_dict(data)
    assert restored.id == exp.id
    assert restored.imported is True
    assert restored.inputs.architecture_hash == "sha256:deadbeef"
    assert restored.status == ExperimentStatus.PLANNED


def test_architecture_round_trip():
    content = {"type": "transformer", "layers": [{"type": "attention"}]}
    h = content_hash(content)
    arch = Architecture.create(name="test-arch", content=content, content_hash=h)
    data = arch.to_dict()
    restored = Architecture.from_dict(data)
    assert restored.content_hash == h
    assert restored.content == content
    assert restored.archived is False


def test_finding_round_trip():
    finding = Finding.create(
        thread_id="thread-abc123",
        summary="X does improve Y",
        reasoning="Evidence from experiments shows...",
        statement_resolutions=[
            StatementResolution(
                statement_id="stmt-abc123",
                resolution=FindingResolution.CONFIRMS,
                note="Strong evidence"
            )
        ],
        experiment_refs=["exp-abc123", "exp-def456"],
    )
    data = finding.to_dict()
    restored = Finding.from_dict(data)
    assert restored.summary == finding.summary
    assert len(restored.statement_resolutions) == 1
    assert restored.statement_resolutions[0].resolution == FindingResolution.CONFIRMS


def test_citation_disclosure():
    cit = Citation(disclosure_id="disc-abc123", relevance="supporting",
                   span_start=10, span_end=50)
    data = cit.to_dict()
    restored = Citation.from_dict(data)
    assert restored.disclosure_id == "disc-abc123"
    assert restored.span_start == 10


def test_citation_external():
    cit = Citation(external_type="arxiv", external_id="2401.12345",
                   external_title="Some Paper", relevance="adjacent")
    data = cit.to_dict()
    restored = Citation.from_dict(data)
    assert restored.external_type == "arxiv"
    assert restored.external_id == "2401.12345"


# ------------------------------------------------------------------
# Canonical round-trip (serialize → parse → serialize = identical)
# ------------------------------------------------------------------

def test_canonical_round_trip():
    arch_content = {"type": "transformer", "heads": 8, "dims": 512}
    canonical_str = canonicalize(arch_content)
    parsed = json.loads(canonical_str)
    canonical_str2 = canonicalize(parsed)
    assert canonical_str == canonical_str2


# ------------------------------------------------------------------
# Repo operations
# ------------------------------------------------------------------

def test_repo_init():
    with tempfile.TemporaryDirectory() as tmp:
        repo = RNDRepo(tmp)
        repo.init(team_name="Test", user_id="user-test")
        assert repo.is_initialized()
        assert (Path(tmp) / "programs").is_dir()
        assert (Path(tmp) / "architectures").is_dir()
        assert (Path(tmp) / "team.json").exists()


def test_repo_save_load_program():
    with tempfile.TemporaryDirectory() as tmp:
        repo = RNDRepo(tmp)
        repo.init(team_name="Test", user_id="user-test")

        prog = Program.create(team_id="team-abc", name="DROGA")
        repo.save(prog)

        loaded = repo.load("program", prog.id)
        assert loaded is not None
        assert loaded.name == "DROGA"
        assert loaded.id == prog.id


def test_repo_save_load_thread_hierarchy():
    with tempfile.TemporaryDirectory() as tmp:
        repo = RNDRepo(tmp)
        repo.init(team_name="Test", user_id="user-test")

        prog = Program.create(team_id="team-abc", name="DROGA")
        repo.save(prog)

        proj = Project.create(program_id=prog.id, name="Transfer Learning")
        repo.save(proj)

        thread = Thread.create(project_id=proj.id,
                               question="Does multi-tier recurrence transfer?")
        repo.save(thread)

        stmt = Statement.create(thread_id=thread.id,
                                hypothesis="Multi-tier recurrence transfers across task families")
        repo.save(stmt)

        # Load them back
        loaded_thread = repo.load("thread", thread.id)
        assert loaded_thread is not None
        assert loaded_thread.question == "Does multi-tier recurrence transfer?"

        loaded_stmt = repo.load("statement", stmt.id)
        assert loaded_stmt is not None
        assert loaded_stmt.thread_id == thread.id


def test_repo_import_architecture():
    with tempfile.TemporaryDirectory() as tmp:
        repo = RNDRepo(tmp)
        repo.init()

        content = {"type": "transformer", "layers": 12, "heads": 8}
        arch = repo.import_architecture(name="mini-transformer", content=content)

        assert arch.content_hash == content_hash(content)

        loaded = repo.load("architecture", arch.id)
        assert loaded is not None
        assert loaded.name == "mini-transformer"
        assert loaded.content_hash == arch.content_hash


def test_repo_import_idempotent_hash():
    """Importing the same content twice produces different entities but same hash."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = RNDRepo(tmp)
        repo.init()

        content = {"type": "rnn", "hidden": 256}
        arch1 = repo.import_architecture(name="rnn-v1", content=content)
        arch2 = repo.import_architecture(name="rnn-v1-dup", content=content)

        assert arch1.id != arch2.id  # different entities
        assert arch1.content_hash == arch2.content_hash  # same content


def test_repo_list_entities():
    with tempfile.TemporaryDirectory() as tmp:
        repo = RNDRepo(tmp)
        repo.init()

        repo.import_architecture("a1", {"type": "cnn"})
        repo.import_architecture("a2", {"type": "rnn"})

        archs = repo.list_entities("architecture")
        assert len(archs) == 2


def test_repo_archive():
    with tempfile.TemporaryDirectory() as tmp:
        repo = RNDRepo(tmp)
        repo.init()

        arch = repo.import_architecture("temp", {"type": "test"})
        assert repo.archive("architecture", arch.id)

        loaded = repo.load("architecture", arch.id)
        assert loaded.archived is True


def test_repo_disclosure_content():
    with tempfile.TemporaryDirectory() as tmp:
        repo = RNDRepo(tmp)
        repo.init()

        disc = Disclosure.create(
            title="Test Conversation",
            disclosure_type=DisclosureType.CLAUDE_TRANSCRIPT,
            created_by="user-test",
        )
        repo.save(disc)
        repo.save_disclosure_content(disc, "# Test\n\nSome conversation content.")

        content = repo.load_disclosure_content(disc)
        assert "Some conversation content" in content


# ------------------------------------------------------------------
# Index operations
# ------------------------------------------------------------------

def test_index_rebuild():
    with tempfile.TemporaryDirectory() as tmp:
        repo = RNDRepo(tmp)
        repo.init(team_name="Test", user_id="user-test")

        # Create some entities
        prog = Program.create(team_id="team-abc", name="DROGA")
        repo.save(prog)

        content = {"type": "transformer"}
        arch = repo.import_architecture("test-arch", content)

        # Rebuild index
        with DerivedIndex(repo.index_path) as idx:
            stats = idx.rebuild(repo.root)
            assert stats["entities"] >= 3  # team + program + architecture

            # Query by type
            archs = idx.find_by_type("architecture")
            assert len(archs) == 1
            assert archs[0]["name"] == "test-arch"

            # Query by ID
            found = idx.find_by_id(prog.id)
            assert found is not None
            assert found["name"] == "DROGA"

            # Query by hash
            found = idx.find_by_hash(arch.content_hash)
            assert found is not None


def test_index_search():
    with tempfile.TemporaryDirectory() as tmp:
        repo = RNDRepo(tmp)
        repo.init(team_name="Test", user_id="user-test")

        repo.import_architecture("llama-mini", {"type": "transformer"})
        repo.import_architecture("gpt-baseline", {"type": "transformer"})

        with DerivedIndex(repo.index_path) as idx:
            idx.rebuild(repo.root)
            results = idx.search("llama")
            assert len(results) == 1
            assert results[0]["name"] == "llama-mini"


if __name__ == "__main__":
    import sys
    # Simple test runner if pytest not available
    test_funcs = [v for k, v in globals().items() if k.startswith("test_")]
    passed = failed = 0
    for func in test_funcs:
        try:
            func()
            passed += 1
            print(f"  PASS  {func.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {func.__name__}: {e}")
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
