"""Authorization layer for RND.

Ownership model (user-confirmed spec):
- Every entity has an owner (user id). Ownership ALWAYS travels with the user.
- team_id on an entity is borrowed visibility, never custody: any current
  member of that team can read/write the entity while the grant stands.
- Removing a team membership instantly revokes that user's borrowed access;
  access is computed live from (owner, entity.team_id, current memberships),
  so there are no per-user grants to orphan.
- No transfer-on-leave. Citing entities keep their hash-anchored references
  (architecture_hash, experiment ids); those remain intact and verifiable
  even after access to the referenced entity is revoked. Team Forks (SDD v2)
  are the future answer for teams wanting snapshots.
- Legacy entities without an ACL row belong to DEFAULT_OWNER (backfilled).

Enforcement is centralized in AccessControlledRepo, a proxy over RNDRepo used
by both the REST layer and the MCP endpoint. Identity comes from the Flask
request context (request.user, set by the auth gates); calls with no request
identity (startup, backfill, explicit .raw usage in public routes) bypass
enforcement — every externally reachable path authenticates before the repo
is touched.
"""

from __future__ import annotations

from flask import has_request_context, request

DEFAULT_OWNER = "user-6fa0fe03542f"  # 'bake' — owner of all pre-authz entities

# Entity types that are shared platform scaffolding, readable by any
# authenticated user (they carry no research content themselves).
UNSCOPED_TYPES = {"team"}


class Forbidden(Exception):
    """Raised when the requesting user cannot access an entity."""

    def __init__(self, entity_id: str, action: str):
        self.entity_id = entity_id
        self.action = action
        super().__init__(f"forbidden: {action} {entity_id}")


def _current_user() -> dict | None:
    if has_request_context():
        return getattr(request, "user", None)
    return None


class AccessControlledRepo:
    """Proxy over RNDRepo enforcing owner/team access on load/save/list.

    Anything not overridden delegates to the raw repo. Public routes that must
    serve unauthenticated readers use `.raw` explicitly and are responsible
    for only exposing published material.
    """

    def __init__(self, raw, auth_db):
        self.raw = raw
        self.auth = auth_db

    # ---- access rule ----

    def _acl(self, entity_id: str) -> dict:
        acl = self.auth.get_entity_acl(entity_id)
        if acl is None:
            acl = {"owner_user_id": DEFAULT_OWNER, "team_id": None}
        return acl

    def can_access(self, user: dict | None, entity_id: str) -> bool:
        if user is None:
            return True  # no-identity calls are internal/trusted (see module docstring)
        acl = self._acl(entity_id)
        if acl["owner_user_id"] == user["id"]:
            return True
        team = acl.get("team_id")
        # Borrowed visibility is only live while the OWNER remains a member:
        # a user leaving a team takes their entities with them (no orphaned
        # grants, no transfer-on-leave). Computed live — nothing to clean up.
        if (team and self.auth.is_team_member(team, user["id"])
                and self.auth.is_team_member(team, acl["owner_user_id"])):
            return True
        return False

    def _check(self, entity_id: str, action: str):
        user = _current_user()
        if not self.can_access(user, entity_id):
            raise Forbidden(entity_id, action)

    # ---- guarded repo surface ----

    def load(self, entity_type: str, entity_id: str):
        entity = self.raw.load(entity_type, entity_id)
        if entity is None or entity_type in UNSCOPED_TYPES:
            return entity
        self._check(entity_id, "read")
        return entity

    def list_entities(self, entity_type: str, **filters) -> list:
        entities = self.raw.list_entities(entity_type, **filters)
        if entity_type in UNSCOPED_TYPES:
            return entities
        user = _current_user()
        if user is None:
            return entities
        return [e for e in entities if self.can_access(user, e.id)]

    def save(self, entity):
        entity_type = type(entity).__name__.lower()
        if entity_type not in UNSCOPED_TYPES:
            user = _current_user()
            existing_acl = self.auth.get_entity_acl(entity.id)
            if existing_acl is None:
                if user is not None:
                    # New entity: stamp ownership; default visibility = the
                    # creator's single team if they have exactly one (tenant
                    # default so team automation stays mutually visible).
                    teams = self.auth.user_team_ids(user["id"])
                    self.auth.set_entity_acl(
                        entity.id, user["id"],
                        teams[0] if len(teams) == 1 else None,
                    )
            else:
                self._check(entity.id, "write")
        return self.raw.save(entity)

    def archive(self, entity_type: str, entity_id: str) -> bool:
        if entity_type not in UNSCOPED_TYPES:
            self._check(entity_id, "archive")
        return self.raw.archive(entity_type, entity_id)

    def import_architecture(self, name, content, variant_of=None, **kw):
        arch = self.raw.import_architecture(name, content, variant_of=variant_of, **kw)
        user = _current_user()
        if user is not None:
            teams = self.auth.user_team_ids(user["id"])
            self.auth.set_entity_acl(
                arch.id, user["id"], teams[0] if len(teams) == 1 else None
            )
        return arch

    def save_paper_section_content(self, paper, section, content):
        self._check(paper.id, "write")
        return self.raw.save_paper_section_content(paper, section, content)

    def load_paper_section_content(self, section):
        # Section content access is guarded by the paper load that precedes it
        # in every route; content_ref alone does not identify the paper.
        return self.raw.load_paper_section_content(section)

    def __getattr__(self, name):
        return getattr(self.raw, name)
