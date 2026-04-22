"""User authentication for RND Platform.

Simple username/password accounts with token-based sessions.
Users stored in SQLite. Passwords hashed with PBKDF2.
"""

from __future__ import annotations

import hashlib
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

CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
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
