"""Persistence tests for interrupted appends, corruption, and OS run locks."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from hb_steering.storage import append_record, read_journal, run_lock


def test_torn_final_append_is_repaired_before_retry(tmp_path):
    path = tmp_path / "records.jsonl"
    append_record(path, {"key": "completed", "value": "kept"})
    complete_bytes = path.read_bytes()
    with path.open("ab") as stream:
        stream.write(b'{"key":"interrupted","value":"half')
    assert read_journal(path) == {"completed": {"key": "completed", "value": "kept"}}
    assert path.read_bytes() == complete_bytes
    append_record(path, {"key": "interrupted", "value": "retried"})
    assert set(read_journal(path)) == {"completed", "interrupted"}


def test_complete_json_without_newline_is_preserved_before_append(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_bytes(b'{"key":"first","value":1}')
    assert read_journal(path)["first"]["value"] == 1
    assert path.read_bytes().endswith(b"\n")
    append_record(path, {"key": "second", "value": 2})
    assert set(read_journal(path)) == {"first", "second"}


@pytest.mark.parametrize("tail", (b'{"key":7}', b"{}", b"[]", b'{"key":""}'))
def test_valid_json_with_invalid_schema_is_not_discarded_as_torn(tmp_path, tail):
    path = tmp_path / "records.jsonl"
    path.write_bytes(b'{"key":"first"}\n' + tail)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="schema"):
        read_journal(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "content",
    (b'{"key":"one"}\ninvalid\n', b'invalid\n{"key":"last"}', b'{"key":"one"}\n{"key":"one"}'),
)
def test_corrupt_completed_lines_and_duplicate_keys_are_rejected(tmp_path, content):
    path = tmp_path / "records.jsonl"
    path.write_bytes(content)
    with pytest.raises(ValueError, match="Corrupt journal|Duplicate journal"):
        read_journal(path)
    assert path.read_bytes() == content


def test_run_lock_blocks_another_process_and_releases_after_exception(tmp_path):
    directory = tmp_path / "run"
    source_root = str(Path(__file__).parents[1] / "src")
    child = """
import sys
from pathlib import Path
from hb_steering.storage import run_lock
try:
    with run_lock(Path(sys.argv[1])):
        pass
except RuntimeError:
    sys.exit(23)
"""
    environment = {**os.environ, "PYTHONPATH": source_root}
    with pytest.raises(RuntimeError, match="intentional"):
        with run_lock(directory):
            blocked = subprocess.run(
                [sys.executable, "-c", child, str(directory)],
                env=environment,
                capture_output=True,
                timeout=10,
            )
            assert blocked.returncode == 23, blocked.stderr.decode()
            raise RuntimeError("intentional")
    allowed = subprocess.run(
        [sys.executable, "-c", child, str(directory)],
        env=environment,
        capture_output=True,
        timeout=10,
    )
    assert allowed.returncode == 0, allowed.stderr.decode()
