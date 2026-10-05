"""Read-only, conservative inventory of tracked paths before any release work.

This is a path gate, not a content/license/PII scanner. It intentionally does
not create a sanitized archive or change the Git index.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
from typing import Iterable


_ROOT = Path(__file__).resolve().parents[2]
_PRIVATE_SUFFIXES = (".sqlite", ".sqlite3", ".db", ".pem", ".key", ".p12", ".pfx", ".log")
_PRIVATE_SIDECARS = ("-wal", "-shm")
_LOCAL_DIRS = frozenset({
    ".wanan", "artifacts", "logs", "log", "traces", "output", "runtime",
    "run", "runs", ".git", ".cache", "__pycache__",
})
_SECRET_DIRS = frozenset({"keys", "secrets", "credentials", "private"})
_SECRET_STEMS = frozenset({
    "secret", "secrets", "credential", "credentials", "id_rsa", "id_ed25519",
    "api_key", "access_token", "auth_token", "token", "password", "passwd",
})
_SECRET_TOKENS = frozenset({
    "secret", "secrets", "credential", "credentials", "token", "password", "passwd",
    "private",
})
_SAFE_APP_SUBDIRS = frozenset({
    "agent", "api", "core", "database", "evaluation", "observability",
    "rag", "release", "schemas", "services",
})
_SAFE_ROOT_FILES = frozenset({
    ".env.example", ".gitignore", ".gitattributes", "AGENTS.md", "HANDOFF.md",
    "requirements.txt", "requirements.lock.txt", "pyproject.toml", "pytest.ini",
    "docker-compose.yml", "README.md", "经验学习.md",
})


def _category(path: str) -> str | None:
    if type(path) is not str or not path or "\\" in path or "\x00" in path or ":" in path:
        return "INVALID_PATH"
    if path.startswith("/") or any(part in {"", ".", ".."} for part in path.split("/")):
        return "INVALID_PATH"
    parts = [part.casefold() for part in path.split("/")]
    name = parts[-1]
    if any(part in _LOCAL_DIRS for part in parts[:-1]):
        return "LOCAL_ARTIFACT"
    if any(part in _SECRET_DIRS for part in parts[:-1]):
        return "CREDENTIAL_FILE"
    if name == ".env" or name.startswith(".env.") and name != ".env.example":
        return "CREDENTIAL_FILE"
    if name.split(".", 1)[0] in _SECRET_STEMS:
        return "CREDENTIAL_FILE"
    if _SECRET_TOKENS.intersection(re.split(r"[._-]+", name)) or "api-key" in name or "api_key" in name:
        return "CREDENTIAL_FILE"
    if name == "trace" or name.startswith(("trace.", "trace_", "trace-")):
        return "LOCAL_ARTIFACT"
    if name.endswith(_PRIVATE_SUFFIXES):
        return "CREDENTIAL_OR_DATABASE"
    if name.endswith(_PRIVATE_SIDECARS):
        return "DATABASE_SIDECAR"
    if parts[0] == "data":
        return "UNREVIEWED_DATA"
    if parts[0] == "eval":
        return "EVALUATION_LICENSE"
    if len(parts) == 1 and path in _SAFE_ROOT_FILES:
        return None
    if parts[0] == "app" and name.endswith(".py") and (
        len(parts) == 2 or len(parts) == 3 and parts[1] in _SAFE_APP_SUBDIRS
    ):
        return None
    if parts[0] == "frontend" and name.endswith(".py") and (
        len(parts) == 2 or len(parts) == 3 and parts[1] == "pages"
    ):
        return None
    if parts[0] in {"tests", "scripts"} and len(parts) == 2 and name.endswith(".py"):
        return None
    if parts[0] == "frontend" and len(parts) == 3 and parts[1] == "static" and name.endswith(".css"):
        return None
    if parts[0] == "spec" and len(parts) == 2 and name.endswith(".md"):
        return None
    return "UNREVIEWED_PATH"


def classify_tracked_paths(paths: Iterable[str]) -> dict[str, object]:
    """Only aggregate controlled categories; never return an input path."""
    if isinstance(paths, (str, bytes, bytearray)):
        return {"status": "INVENTORY_ERROR"}
    categories: dict[str, int] = {}
    tracked_count = 0
    try:
        for path in paths:
            tracked_count += 1
            category = _category(path)
            if category is not None:
                categories[category] = categories.get(category, 0) + 1
    except Exception:
        return {"status": "INVENTORY_ERROR"}
    blocked_count = sum(categories.values())
    return {
        "status": "BLOCKED" if blocked_count else "PATH_GATE_CLEAR",
        "tracked_count": tracked_count,
        "blocked_count": blocked_count,
        "categories": dict(sorted(categories.items())),
    }


def inventory_current_index() -> dict[str, object]:
    """Read the index only; any Git/decoding error returns a fixed code."""
    try:
        # Git environment overrides can silently select an empty alternate index.
        # They must not affect the release inventory's repository identity.
        clean_env = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
        root_check = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=_ROOT, env=clean_env, capture_output=True, check=True, timeout=10,
        )
        if Path(root_check.stdout.decode("utf-8", errors="strict").strip()).resolve() != _ROOT.resolve():
            return {"status": "INVENTORY_ERROR"}
        completed = subprocess.run(
            ["git", "ls-files", "--stage", "-z"],
            cwd=_ROOT, env=clean_env, capture_output=True, check=True, timeout=10,
        )
        raw_entries = completed.stdout.split(b"\x00")
        if raw_entries and raw_entries[-1] == b"":
            raw_entries.pop()
        categories: dict[str, int] = {}
        for entry in raw_entries:
            metadata, raw_path = entry.split(b"\t", 1)
            mode, object_id, stage = metadata.split(b" ")
            if len(object_id) not in (40, 64) or re.fullmatch(rb"[0-9a-f]+", object_id) is None:
                return {"status": "INVENTORY_ERROR"}
            path = raw_path.decode("utf-8", errors="strict")
            category = (
                _category(path) if mode in (b"100644", b"100755") and stage == b"0"
                else "NONREGULAR_INDEX_ENTRY"
            )
            if category is not None:
                categories[category] = categories.get(category, 0) + 1
        blocked_count = sum(categories.values())
        return {
            "status": "BLOCKED" if blocked_count else "PATH_GATE_CLEAR",
            "tracked_count": len(raw_entries),
            "blocked_count": blocked_count,
            "categories": dict(sorted(categories.items())),
        }
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError, TypeError):
        return {"status": "INVENTORY_ERROR"}


def main() -> int:
    result = inventory_current_index()
    print(json.dumps(result, sort_keys=True, ensure_ascii=True))
    return 0 if result["status"] == "PATH_GATE_CLEAR" else 2


if __name__ == "__main__":
    raise SystemExit(main())
