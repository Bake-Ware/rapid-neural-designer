"""Tests for the authz layer: ownership, team tenancy, derived client credentials.

Run: python -m pytest tests/test_authz.py -v
"""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from flask import Flask, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rnd.auth import AuthDB
from rnd.authz import AccessControlledRepo, Forbidden, DEFAULT_OWNER

TEAM = "team-test-alpha"


class FakeRepo:
    """Minimal raw-repo stand-in: dict of entities with .id."""

    def __init__(self):
        self.store = {}

    def load(self, entity_type, entity_id):
        return self.store.get(entity_id)

    def list_entities(self, entity_type, **filters):
        return [e for e in self.store.values()
                if type(e).__name__.lower() == entity_type]

    def save(self, entity):
        self.store[entity.id] = entity
        return Path("/dev/null")

    def archive(self, entity_type, entity_id):
        return self.store.pop(entity_id, None) is not None


class Paper(SimpleNamespace):
    pass


@pytest.fixture
def env():
    with tempfile.TemporaryDirectory() as d:
        db = AuthDB(Path(d) / "auth.sqlite")
        db.open()
        alice = db.register("alice", "pw-alice")
        bob = db.register("bob", "pw-bob")
        repo = AccessControlledRepo(FakeRepo(), db)
        app = Flask(__name__)
        yield SimpleNamespace(db=db, repo=repo, app=app, alice=alice, bob=bob)
        db.close()


def as_user(env, user):
    ctx = env.app.test_request_context("/")
    ctx.push()
    request.user = user
    return ctx


def test_owner_full_access_others_forbidden(env):
    ctx = as_user(env, env.alice)
    p = Paper(id="paper-1")
    env.repo.save(p)  # stamps alice as owner
    assert env.repo.load("paper", "paper-1") is p
    ctx.pop()

    ctx = as_user(env, env.bob)
    with pytest.raises(Forbidden):
        env.repo.load("paper", "paper-1")
    with pytest.raises(Forbidden):
        env.repo.save(p)
    assert env.repo.list_entities("paper") == []
    ctx.pop()


def test_team_membership_grants_and_leaving_revokes(env):
    """Coordinator-specified: leaving a team instantly revokes borrowed access;
    citing entities keep their hash refs intact and loadable."""
    env.db.add_team_member(TEAM, env.alice["id"])
    env.db.add_team_member(TEAM, env.bob["id"])

    # alice creates an experiment; single-team default scopes it to TEAM
    ctx = as_user(env, env.alice)
    exp = Paper(id="exp-alice-1")
    env.repo.save(exp)
    assert env.db.get_entity_acl("exp-alice-1")["team_id"] == TEAM
    ctx.pop()

    # bob (teammate) creates a paper citing alice's experiment by hash-anchored ref
    ctx = as_user(env, env.bob)
    citing = Paper(id="paper-bob-1",
                   bindings=["exp-alice-1", "sha256:deadbeef"])
    env.repo.save(citing)
    assert env.repo.load("paper", "exp-alice-1") is exp  # borrowed visibility
    ctx.pop()

    # alice leaves the team -> bob's borrowed access is gone, no orphaned grants
    env.db.remove_team_member(TEAM, env.alice["id"])
    ctx = as_user(env, env.bob)
    with pytest.raises(Forbidden):
        env.repo.load("paper", "exp-alice-1")
    # citing paper still loads with bindings intact (verifiable hash refs)
    still = env.repo.load("paper", "paper-bob-1")
    assert still.bindings == ["exp-alice-1", "sha256:deadbeef"]
    ctx.pop()

    # alice still owns her entity — ownership traveled with her
    ctx = as_user(env, env.alice)
    assert env.repo.load("paper", "exp-alice-1") is exp
    ctx.pop()


def test_legacy_entities_default_to_bake(env):
    env.repo.raw.store["thread-legacy"] = Paper(id="thread-legacy")
    ctx = as_user(env, env.bob)
    with pytest.raises(Forbidden):
        env.repo.load("paper", "thread-legacy")
    ctx.pop()
    ctx = as_user(env, {"id": DEFAULT_OWNER, "username": "bake"})
    assert env.repo.load("paper", "thread-legacy") is not None
    ctx.pop()


def test_client_credentials_derivation_and_rotation(env):
    creds = env.db.get_client_credentials(env.alice["id"])
    assert creds and len(creds["client_id"]) == 64 and len(creds["client_secret"]) == 64
    # valid pair authenticates to the right user
    u = env.db.validate_client_credentials(creds["client_id"], creds["client_secret"])
    assert u and u["id"] == env.alice["id"]
    # deterministic
    assert env.db.get_client_credentials(env.alice["id"]) == creds
    # wrong secret rejected
    assert env.db.validate_client_credentials(creds["client_id"], "0" * 64) is None


def test_pass_the_hash_rejected(env):
    """The raw stored password hash must NOT work as a client secret."""
    row = env.db.conn.execute(
        "SELECT password_hash FROM users WHERE id = ?", (env.alice["id"],)
    ).fetchone()
    creds = env.db.get_client_credentials(env.alice["id"])
    assert env.db.validate_client_credentials(creds["client_id"], row["password_hash"]) is None


def test_no_identity_is_internal_trusted(env):
    # startup/backfill paths run without request context and are not blocked
    p = Paper(id="paper-boot")
    env.repo.save(p)
    assert env.repo.load("paper", "paper-boot") is p
