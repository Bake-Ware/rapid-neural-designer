"""Canonical JSON serialization and content hashing.

Canonical form (SDD §5.5):
- Keys sorted alphabetically at all nesting levels
- No trailing whitespace or newline
- UTF-8 encoded
- Numbers in shortest round-trip representation
- No insignificant whitespace between tokens

All hashing uses SHA-256 over the canonical serialization.
"""

import hashlib
import json


def canonicalize(obj: dict) -> str:
    """Produce canonical JSON string from a dict.

    Python's json.dumps with sort_keys=True and compact separators
    satisfies all SDD requirements: sorted keys, no whitespace,
    shortest float repr via float.__repr__().
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_hash(obj: dict) -> str:
    """SHA-256 hash of the canonical JSON form. Returns 'sha256:{hex}'."""
    canonical = canonicalize(obj)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"
