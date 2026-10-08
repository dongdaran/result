"""Atomic files, append-only records, and process locking for resumable runs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def fingerprint(value: object) -> str:
    return digest_bytes(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_json(path: Path, value: object) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode())


@contextmanager
def run_lock(directory: Path):
    """An OS lock is released even if a process crashes (Linux/macOS)."""
    import fcntl

    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is using {directory}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def read_journal(path: Path) -> dict[str, dict]:
    """Repair only a final partial line after a killed process; other corruption fails."""
    if not path.exists():
        return {}
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    result = {}
    good_bytes = 0
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            if index == len(lines) - 1 and not line.endswith(b"\n"):
                atomic_bytes(path, data[:good_bytes])
                break
            raise ValueError(f"Corrupt journal: {path}, line {index + 1}") from exc
        if not isinstance(row, dict) or not isinstance(row.get("key"), str) or not row["key"]:
            raise ValueError(f"Invalid journal schema: {path}, line {index + 1}")
        key = row["key"]
        if key in result:
            raise ValueError(f"Duplicate journal key in {path}: {key}")
        result[key] = row
        good_bytes += len(line)
    # A valid JSON line missing its final newline must not join the next append.
    if lines and good_bytes == len(data) and not data.endswith(b"\n"):
        atomic_bytes(path, data + b"\n")
    return result


def append_record(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write((json.dumps(value, ensure_ascii=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
