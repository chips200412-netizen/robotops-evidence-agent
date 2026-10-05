"""Fail-closed, redacted scan of indexed source text; not a release audit."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess


_ROOT = Path(__file__).resolve().parents[2]
_MAX_FILE_BYTES = 512 * 1024
_MAX_TOTAL_BYTES = 32 * 1024 * 1024
_ROOT_TEXT = frozenset({
    ".dockerignore", ".env.example", ".gitattributes", ".gitignore", "AGENTS.md",
    "Dockerfile", "HANDOFF.md", "README.md", "design-qa.md", "docker-compose.yml",
    "requirements.txt", "requirements.lock.txt", "经验学习.md",
})
_PATTERNS = {
    "PRIVATE_KEY": re.compile(
        r"-----BEGIN (?:(?:RSA|EC|OPENSSH|DSA|ENCRYPTED) )?PRIVATE KEY-----|"
        r"-----BEGIN PGP PRIVATE KEY " + r"BLOCK-----"
    ),
    "OPENAI_TOKEN": re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}\b"),
    "GITHUB_TOKEN": re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b"),
    "GITHUB_FINE_GRAINED_TOKEN": re.compile(r"(?i)\bgithub_pat_[a-z0-9_]{20,}\b"),
    "GITLAB_TOKEN": re.compile(r"(?i)\bglpat-[a-z0-9_-]{20,}\b"),
    "SLACK_TOKEN": re.compile(r"(?i)\bxox[baprs]-[a-z0-9-]{20,}\b"),
    "AWS_ACCESS_KEY": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "GOOGLE_API_KEY": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    "GENERIC_CREDENTIAL": re.compile(
        r"(?im)^\s*[A-Z_]*(?:API_KEY|SECRET|TOKEN|PASSWORD|PRIVATE_KEY)\s*[:=]\s*[\"']"
        r"(?!test[-_]|example[-_]|placeholder[-_]|changeme\b)[A-Za-z0-9_./+=-]{24,}[\"']"
    ),
}


def in_source_scope(path: str) -> bool:
    """A narrow incremental scope; data/eval and history need separate release gates."""
    if type(path) is not str or not path or "\\" in path or "\x00" in path or ":" in path:
        return False
    parts = path.split("/")
    if path.startswith("/") or any(part in {"", ".", ".."} for part in parts):
        return False
    if len(parts) == 1:
        return path in _ROOT_TEXT
    folded = [part.casefold() for part in parts]
    if folded[0] in {"app", "frontend", "scripts", "tests", "spec"}:
        # A new source filename must be scanned or fail decoding/size checks,
        # never silently omitted because its extension was not anticipated.
        return True
    return len(parts) >= 3 and folded[:2] == [".github", "workflows"]


def scan_source_bytes(raw: bytes) -> set[str]:
    if type(raw) is not bytes or len(raw) > _MAX_FILE_BYTES or b"\x00" in raw:
        raise ValueError("source is not bounded text")
    source = raw.decode("utf-8", errors="strict")
    return {category for category, pattern in _PATTERNS.items() if pattern.search(source)}


def scan_indexed_sources() -> dict[str, object]:
    """Scan staged blobs, not mutable working-tree files; emit no path or match."""
    try:
        clean_env = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], cwd=_ROOT, env=clean_env,
            capture_output=True, check=True, timeout=10,
        )
        if Path(root.stdout.decode("utf-8", errors="strict").strip()).resolve() != _ROOT.resolve():
            raise ValueError("wrong repository")
        index = subprocess.run(
            ["git", "ls-files", "--stage", "-z"], cwd=_ROOT, env=clean_env,
            capture_output=True, check=True, timeout=10,
        )
        selected: list[bytes] = []
        for entry in index.stdout.split(b"\x00"):
            if not entry:
                continue
            metadata, raw_path = entry.split(b"\t", 1)
            mode, object_id, stage = metadata.split(b" ")
            path = raw_path.decode("utf-8", errors="strict")
            if not in_source_scope(path):
                continue
            if mode not in (b"100644", b"100755") or stage != b"0":
                raise ValueError("unsupported index entry")
            if len(object_id) not in (40, 64) or re.fullmatch(rb"[0-9a-f]+", object_id) is None:
                raise ValueError("invalid object id")
            selected.append(object_id)
        batch = subprocess.run(
            ["git", "cat-file", "--batch"], input=b"\n".join(selected) + (b"\n" if selected else b""),
            cwd=_ROOT, env=clean_env, capture_output=True, check=True, timeout=30,
        )
        offset = 0
        total = 0
        categories: dict[str, int] = {}
        for object_id in selected:
            header_end = batch.stdout.index(b"\n", offset)
            header = batch.stdout[offset:header_end].split(b" ")
            if len(header) != 3 or header[0] != object_id or header[1] != b"blob":
                raise ValueError("invalid blob header")
            size = int(header[2])
            if size < 0 or size > _MAX_FILE_BYTES:
                raise ValueError("oversized source")
            total += size
            if total > _MAX_TOTAL_BYTES:
                raise ValueError("oversized source set")
            start = header_end + 1
            end = start + size
            if end >= len(batch.stdout) or batch.stdout[end:end + 1] != b"\n":
                raise ValueError("truncated blob")
            for category in scan_source_bytes(batch.stdout[start:end]):
                categories[category] = categories.get(category, 0) + 1
            offset = end + 1
        if offset != len(batch.stdout):
            raise ValueError("unexpected blob data")
        return {
            "status": "BLOCKED" if categories else "CLEAR",
            "scanned_count": len(selected),
            "categories": dict(sorted(categories.items())),
        }
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError, TypeError, IndexError):
        return {"status": "SCAN_ERROR"}


def main() -> int:
    report = scan_indexed_sources()
    print(json.dumps(report, sort_keys=True, ensure_ascii=True))
    return 0 if report["status"] == "CLEAR" else 2


if __name__ == "__main__":
    raise SystemExit(main())
