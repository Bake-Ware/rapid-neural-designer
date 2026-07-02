"""User authentication for RND Platform.

Simple username/password accounts with token-based sessions.
Users stored in SQLite. Passwords hashed with PBKDF2.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
from pathlib import Path
from datetime import datetime, timezone, timedelta

AUTH_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS mcp_clients (
    client_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    secret_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    label TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS oauth_codes (
    code TEXT PRIMARY KEY,
    client_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    redirect_uri TEXT NOT NULL,
    code_challenge TEXT NOT NULL,
    code_challenge_method TEXT NOT NULL DEFAULT 'S256',
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    used INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS oauth_tokens (
    access_token TEXT PRIMARY KEY,
    client_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS user_components (
    id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'component',
    category TEXT NOT NULL DEFAULT '',
    definition TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (owner_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS server_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entity_acl (
    entity_id TEXT PRIMARY KEY,
    owner_user_id TEXT NOT NULL,
    team_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS team_members (
    team_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    added_at TEXT NOT NULL,
    PRIMARY KEY (team_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_entity_acl_owner ON entity_acl(owner_user_id);
CREATE INDEX IF NOT EXISTS idx_team_members_user ON team_members(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_mcp_clients_user ON mcp_clients(user_id);
CREATE INDEX IF NOT EXISTS idx_oauth_tokens_user ON oauth_tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_user_components_owner ON user_components(owner_id);
"""

SESSION_DAYS = 30


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000
    ).hex()


class AuthDB:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.conn: sqlite3.Connection | None = None

    def open(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(AUTH_SCHEMA)
        self._ensure_server_secret()

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    # ---- Users ----

    def register(self, username: str, password: str, display_name: str = "") -> dict | None:
        """Create a new user. Returns user dict or None if username taken."""
        assert self.conn
        username = username.strip().lower()
        display_name = display_name.strip() or username

        # Check if exists
        row = self.conn.execute(
            "SELECT id FROM users WHERE username = ?", (username,)
        ).fetchone()
        if row:
            return None

        user_id = f"user-{secrets.token_hex(6)}"
        salt = secrets.token_hex(16)
        password_hash = _hash_password(password, salt)
        now = datetime.now(timezone.utc).isoformat()

        self.conn.execute(
            "INSERT INTO users (id, username, display_name, password_hash, salt, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, username, display_name, password_hash, salt, now),
        )
        self.conn.commit()
        return {"id": user_id, "username": username, "display_name": display_name}

    def login(self, username: str, password: str) -> dict | None:
        """Verify credentials and create a session. Returns {token, user} or None."""
        assert self.conn
        username = username.strip().lower()

        row = self.conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        if not row:
            return None

        expected = _hash_password(password, row["salt"])
        if expected != row["password_hash"]:
            return None

        # Create session token
        token = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        expires = now + timedelta(days=SESSION_DAYS)

        self.conn.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, row["id"], now.isoformat(), expires.isoformat()),
        )
        self.conn.commit()

        return {
            "token": token,
            "user": {
                "id": row["id"],
                "username": row["username"],
                "display_name": row["display_name"],
            },
        }

    def validate_token(self, token: str) -> dict | None:
        """Validate a session token. Returns user dict or None."""
        assert self.conn
        row = self.conn.execute(
            """SELECT u.id, u.username, u.display_name, s.expires_at
               FROM sessions s JOIN users u ON s.user_id = u.id
               WHERE s.token = ?""",
            (token,),
        ).fetchone()
        if not row:
            return None

        # Check expiry
        expires = datetime.fromisoformat(row["expires_at"])
        if datetime.now(timezone.utc) > expires:
            self.conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            self.conn.commit()
            return None

        return {
            "id": row["id"],
            "username": row["username"],
            "display_name": row["display_name"],
        }

    def logout(self, token: str):
        """Delete a session token."""
        assert self.conn
        self.conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        self.conn.commit()

    def get_user(self, user_id: str) -> dict | None:
        """Get user by ID."""
        assert self.conn
        row = self.conn.execute(
            "SELECT id, username, display_name, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        return dict(row) if row else None

    # ---- MCP Clients ----

    def create_mcp_client(self, user_id: str, label: str = "") -> dict | None:
        """Create an MCP client credential tied to a user. Returns {client_id, secret} or None."""
        assert self.conn
        # Verify user exists
        if not self.get_user(user_id):
            return None

        client_id = f"mcp-{secrets.token_hex(8)}"
        secret = secrets.token_urlsafe(32)
        salt = secrets.token_hex(16)
        secret_hash = _hash_password(secret, salt)
        now = datetime.now(timezone.utc).isoformat()

        self.conn.execute(
            "INSERT INTO mcp_clients (client_id, user_id, secret_hash, salt, label, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (client_id, user_id, secret_hash, salt, label, now),
        )
        self.conn.commit()
        return {"client_id": client_id, "secret": secret, "user_id": user_id, "label": label}

    def validate_mcp_client(self, client_id: str, secret: str) -> dict | None:
        """Validate MCP client credentials. Returns user dict or None."""
        assert self.conn
        row = self.conn.execute(
            "SELECT mc.*, u.username, u.display_name FROM mcp_clients mc "
            "JOIN users u ON mc.user_id = u.id WHERE mc.client_id = ?",
            (client_id,),
        ).fetchone()
        if not row:
            return None

        expected = _hash_password(secret, row["salt"])
        if expected != row["secret_hash"]:
            return None

        return {
            "id": row["user_id"],
            "username": row["username"],
            "display_name": row["display_name"],
            "client_id": client_id,
        }

    def list_mcp_clients(self, user_id: str) -> list[dict]:
        """List MCP clients for a user (without secrets)."""
        assert self.conn
        rows = self.conn.execute(
            "SELECT client_id, user_id, label, created_at FROM mcp_clients WHERE user_id = ?",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def revoke_mcp_client(self, client_id: str) -> bool:
        """Delete an MCP client credential."""
        assert self.conn
        cursor = self.conn.execute("DELETE FROM mcp_clients WHERE client_id = ?", (client_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    # ---- OAuth (for claude.ai MCP connectors) ----

    def create_oauth_code(self, client_id: str, user_id: str, redirect_uri: str,
                          code_challenge: str, code_challenge_method: str = "S256") -> str:
        """Create an OAuth authorization code. Returns the code."""
        assert self.conn
        code = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        expires = now + timedelta(minutes=10)
        self.conn.execute(
            "INSERT INTO oauth_codes (code, client_id, user_id, redirect_uri, "
            "code_challenge, code_challenge_method, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (code, client_id, user_id, redirect_uri, code_challenge,
             code_challenge_method, now.isoformat(), expires.isoformat()),
        )
        self.conn.commit()
        return code

    def exchange_oauth_code(self, code: str, code_verifier: str, redirect_uri: str) -> dict | None:
        """Exchange an authorization code + PKCE verifier for an access token.
        Returns {access_token, token_type, expires_in} or None."""
        assert self.conn
        row = self.conn.execute(
            "SELECT * FROM oauth_codes WHERE code = ? AND used = 0", (code,)
        ).fetchone()
        if not row:
            return None

        # Check expiry
        expires = datetime.fromisoformat(row["expires_at"])
        if datetime.now(timezone.utc) > expires:
            self.conn.execute("DELETE FROM oauth_codes WHERE code = ?", (code,))
            self.conn.commit()
            return None

        # Check redirect_uri matches
        if row["redirect_uri"] != redirect_uri:
            return None

        # Verify PKCE challenge
        import base64
        digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
        computed_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        if computed_challenge != row["code_challenge"]:
            return None

        # Mark code as used
        self.conn.execute("UPDATE oauth_codes SET used = 1 WHERE code = ?", (code,))

        # Create access token
        access_token = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        token_expires = now + timedelta(days=90)
        self.conn.execute(
            "INSERT INTO oauth_tokens (access_token, client_id, user_id, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (access_token, row["client_id"], row["user_id"], now.isoformat(), token_expires.isoformat()),
        )
        self.conn.commit()
        return {
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": 90 * 24 * 3600,
        }

    def validate_oauth_token(self, access_token: str) -> dict | None:
        """Validate an OAuth access token. Returns user dict or None."""
        assert self.conn
        row = self.conn.execute(
            "SELECT ot.*, u.username, u.display_name FROM oauth_tokens ot "
            "JOIN users u ON ot.user_id = u.id WHERE ot.access_token = ?",
            (access_token,),
        ).fetchone()
        if not row:
            return None
        expires = datetime.fromisoformat(row["expires_at"])
        if datetime.now(timezone.utc) > expires:
            self.conn.execute("DELETE FROM oauth_tokens WHERE access_token = ?", (access_token,))
            self.conn.commit()
            return None
        return {
            "id": row["user_id"],
            "username": row["username"],
            "display_name": row["display_name"],
            "client_id": row["client_id"],
        }

    def get_mcp_client_user(self, client_id: str) -> dict | None:
        """Get the user associated with an MCP client (no secret check)."""
        assert self.conn
        row = self.conn.execute(
            "SELECT mc.user_id, u.username, u.display_name FROM mcp_clients mc "
            "JOIN users u ON mc.user_id = u.id WHERE mc.client_id = ?",
            (client_id,),
        ).fetchone()
        return dict(row) if row else None

    # ---- Derived API tokens (client_credentials-style, no minting) ----

    def _ensure_server_secret(self) -> str:
        """Get-or-create the stable secret used to sign derived API tokens."""
        assert self.conn
        row = self.conn.execute(
            "SELECT value FROM server_meta WHERE key = 'api_token_secret'"
        ).fetchone()
        if row:
            return row["value"]
        secret = os.environ.get("RND_API_SECRET") or secrets.token_hex(32)
        self.conn.execute(
            "INSERT INTO server_meta (key, value) VALUES ('api_token_secret', ?)", (secret,)
        )
        self.conn.commit()
        return secret

    def _compute_api_token(self, user_id: str, password_hash: str) -> str:
        secret = self._ensure_server_secret()
        sig = hmac.new(
            secret.encode("utf-8"),
            f"{user_id}:{password_hash}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"{user_id}.{sig}"

    def get_api_token(self, user_id: str) -> str | None:
        """Derive the deterministic API token for an existing user.

        It's an HMAC over (user_id, password_hash), so it needs no storage,
        is stable until the user changes their password (which rotates it),
        and can be shown on a logged-in 'get my token' page.
        """
        assert self.conn
        row = self.conn.execute(
            "SELECT id, password_hash FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if not row:
            return None
        return self._compute_api_token(row["id"], row["password_hash"])

    def validate_api_token(self, token: str) -> dict | None:
        """Validate a derived API token. Returns user dict or None."""
        assert self.conn
        if not token or "." not in token:
            return None
        user_id = token.split(".", 1)[0]
        row = self.conn.execute(
            "SELECT id, username, display_name, password_hash FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if not row:
            return None
        expected = self._compute_api_token(row["id"], row["password_hash"])
        if not hmac.compare_digest(expected, token):
            return None
        return {
            "id": row["id"],
            "username": row["username"],
            "display_name": row["display_name"],
        }

    # ---- Entity ACLs (ownership travels with the user; team = borrowed visibility) ----

    def set_entity_acl(self, entity_id: str, owner_user_id: str, team_id: str | None = None):
        assert self.conn
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "INSERT INTO entity_acl (entity_id, owner_user_id, team_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(entity_id) DO UPDATE SET owner_user_id=excluded.owner_user_id, "
            "team_id=excluded.team_id, updated_at=excluded.updated_at",
            (entity_id, owner_user_id, team_id, now, now),
        )
        self.conn.commit()

    def get_entity_acl(self, entity_id: str) -> dict | None:
        assert self.conn
        row = self.conn.execute(
            "SELECT entity_id, owner_user_id, team_id FROM entity_acl WHERE entity_id = ?",
            (entity_id,),
        ).fetchone()
        return dict(row) if row else None

    def set_entity_team(self, entity_id: str, team_id: str | None) -> bool:
        assert self.conn
        now = datetime.now(timezone.utc).isoformat()
        cur = self.conn.execute(
            "UPDATE entity_acl SET team_id = ?, updated_at = ? WHERE entity_id = ?",
            (team_id, now, entity_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    # ---- Team membership (tenancy; removal instantly revokes borrowed access) ----

    def add_team_member(self, team_id: str, user_id: str):
        assert self.conn
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "INSERT OR IGNORE INTO team_members (team_id, user_id, added_at) VALUES (?, ?, ?)",
            (team_id, user_id, now),
        )
        self.conn.commit()

    def remove_team_member(self, team_id: str, user_id: str) -> bool:
        assert self.conn
        cur = self.conn.execute(
            "DELETE FROM team_members WHERE team_id = ? AND user_id = ?",
            (team_id, user_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def is_team_member(self, team_id: str, user_id: str) -> bool:
        assert self.conn
        row = self.conn.execute(
            "SELECT 1 FROM team_members WHERE team_id = ? AND user_id = ?",
            (team_id, user_id),
        ).fetchone()
        return row is not None

    def user_team_ids(self, user_id: str) -> list[str]:
        assert self.conn
        rows = self.conn.execute(
            "SELECT team_id FROM team_members WHERE user_id = ?", (user_id,)
        ).fetchall()
        return [r["team_id"] for r in rows]

    def list_team_members(self, team_id: str) -> list[dict]:
        assert self.conn
        rows = self.conn.execute(
            "SELECT tm.user_id, u.username, tm.added_at FROM team_members tm "
            "JOIN users u ON u.id = tm.user_id WHERE tm.team_id = ?",
            (team_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ---- Derived client credentials (agent access to a user's private stack) ----
    # client_id     = sha256(username:user_id) — stable, public, identifies the user
    # client_secret = HMAC(server_secret, "mcp-client:" + user_id + ":" + password_hash)
    # Nothing is stored: validation recomputes from the user row, so a users-table
    # leak alone yields no usable secret (pass-the-hash is rejected by construction),
    # and changing the password rotates the secret automatically.

    @staticmethod
    def _client_id_for(username: str, user_id: str) -> str:
        return hashlib.sha256(f"{username}:{user_id}".encode("utf-8")).hexdigest()

    def _client_secret_for(self, user_id: str, password_hash: str) -> str:
        secret = self._ensure_server_secret()
        return hmac.new(
            secret.encode("utf-8"),
            f"mcp-client:{user_id}:{password_hash}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def get_client_credentials(self, user_id: str) -> dict | None:
        """Derive the client_id/client_secret pair for a user (shown to them once
        per request; regenerated by changing the password)."""
        assert self.conn
        row = self.conn.execute(
            "SELECT id, username, password_hash FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if not row:
            return None
        return {
            "client_id": self._client_id_for(row["username"], row["id"]),
            "client_secret": self._client_secret_for(row["id"], row["password_hash"]),
        }

    def validate_client_credentials(self, client_id: str, client_secret: str) -> dict | None:
        """Validate a derived client pair. Returns user dict or None."""
        assert self.conn
        rows = self.conn.execute(
            "SELECT id, username, display_name, password_hash FROM users"
        ).fetchall()
        for row in rows:
            if self._client_id_for(row["username"], row["id"]) == client_id:
                expected = self._client_secret_for(row["id"], row["password_hash"])
                if hmac.compare_digest(expected, client_secret):
                    return {
                        "id": row["id"],
                        "username": row["username"],
                        "display_name": row["display_name"],
                        "client_id": client_id,
                    }
                return None
        return None

    # ---- Admin ----

    def is_admin(self, user_id: str) -> bool:
        """Check if user is the first registered user (admin)."""
        assert self.conn
        row = self.conn.execute(
            "SELECT id FROM users ORDER BY created_at ASC LIMIT 1"
        ).fetchone()
        return row is not None and row["id"] == user_id

    # ---- User Components ----

    def create_user_component(self, owner_id: str, kind: str, definition: str,
                              category: str = "") -> dict:
        """Create a user component/atomic. Returns the record dict."""
        assert self.conn
        comp_id = f"uco-{secrets.token_hex(6)}"
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "INSERT INTO user_components (id, owner_id, kind, category, definition, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (comp_id, owner_id, kind, category, definition, now, now),
        )
        self.conn.commit()
        return {"id": comp_id, "owner_id": owner_id, "kind": kind, "category": category,
                "definition": definition, "created_at": now, "updated_at": now}

    def list_user_components(self, owner_id: str) -> list[dict]:
        """List all components owned by a user."""
        assert self.conn
        rows = self.conn.execute(
            "SELECT * FROM user_components WHERE owner_id = ? ORDER BY created_at DESC",
            (owner_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_user_component(self, comp_id: str) -> dict | None:
        """Get a user component by ID."""
        assert self.conn
        row = self.conn.execute(
            "SELECT * FROM user_components WHERE id = ?", (comp_id,)
        ).fetchone()
        return dict(row) if row else None

    def update_user_component(self, comp_id: str, owner_id: str,
                              definition: str | None = None,
                              category: str | None = None) -> dict | None:
        """Update a user component. Only the owner can update."""
        assert self.conn
        row = self.conn.execute(
            "SELECT * FROM user_components WHERE id = ? AND owner_id = ?",
            (comp_id, owner_id),
        ).fetchone()
        if not row:
            return None
        now = datetime.now(timezone.utc).isoformat()
        if definition is not None:
            self.conn.execute(
                "UPDATE user_components SET definition = ?, updated_at = ? WHERE id = ?",
                (definition, now, comp_id),
            )
        if category is not None:
            self.conn.execute(
                "UPDATE user_components SET category = ?, updated_at = ? WHERE id = ?",
                (category, now, comp_id),
            )
        self.conn.commit()
        return self.get_user_component(comp_id)

    def delete_user_component(self, comp_id: str, owner_id: str) -> bool:
        """Delete a user component. Only the owner can delete."""
        assert self.conn
        cursor = self.conn.execute(
            "DELETE FROM user_components WHERE id = ? AND owner_id = ?",
            (comp_id, owner_id),
        )
        self.conn.commit()
        return cursor.rowcount > 0
