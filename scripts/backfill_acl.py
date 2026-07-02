#!/usr/bin/env python3
"""One-time ACL backfill: assign every pre-authz entity to its legacy owner.

- owner: user-6fa0fe03542f ('bake') for all existing entities
- team scope: team-5d1c4879f375 (borrowed visibility for team members)
- membership: ensures bake + claude-code are members of the team

Idempotent; safe to re-run. Run from the repo root on the server:
    venv/bin/python scripts/backfill_acl.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rnd.repo import RNDRepo
from rnd.auth import AuthDB

OWNER = "user-6fa0fe03542f"          # bake
CLAUDE = "user-d965024dd4b8"         # claude-code service account
TEAM = "team-5d1c4879f375"
TYPES = ["program", "project", "thread", "statement", "experiment",
         "finding", "architecture", "paper", "disclosure"]


def main():
    root = Path(__file__).resolve().parent.parent
    repo = RNDRepo(root)
    db = AuthDB(repo.rnd_dir / "auth.sqlite")
    db.open()

    db.add_team_member(TEAM, OWNER)
    db.add_team_member(TEAM, CLAUDE)
    print(f"team {TEAM} members: {[m['username'] for m in db.list_team_members(TEAM)]}")

    stamped = skipped = 0
    for t in TYPES:
        for e in repo.list_entities(t):
            if db.get_entity_acl(e.id):
                skipped += 1
                continue
            db.set_entity_acl(e.id, OWNER, TEAM)
            stamped += 1
    print(f"backfill: stamped {stamped}, already had acl {skipped}")
    db.close()


if __name__ == "__main__":
    main()
