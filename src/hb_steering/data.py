"""Pinned HarmBench ingestion, explicit labels, and grouped OOF data folds.

The downloaded CSV remains unchanged. Severity levels are project annotations,
not HarmBench categories; this module never infers or fabricates those labels.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import random
import re
import tempfile
import time
import unicodedata
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

OFFICIAL_COLUMNS = (
    "Behavior",
    "FunctionalCategory",
    "SemanticCategory",
    "Tags",
    "ContextString",
    "BehaviorID",
)
DEFAULT_CATEGORIES = ("standard", "contextual")
SOURCE_PATH = "data/behavior_datasets/harmbench_behaviors_text_all.csv"
SPLIT_NAMES = ("fit", "dev", "test")
DOWNLOAD_TIMEOUT = 60
DOWNLOAD_ATTEMPTS = 3
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temp_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _reader(text: str) -> csv.DictReader:
    reader = csv.DictReader(io.StringIO(text))
    missing = set(OFFICIAL_COLUMNS) - set(reader.fieldnames or ())
    if missing:
        raise ValueError("HarmBench CSV missing columns: " + ", ".join(sorted(missing)))
    return reader


def download_harmbench(root: Path, revision: str) -> dict:
    """Download the official CSV at a full commit SHA into ``root/data/raw``.

    A matching cached download is reused only after verifying its SHA-256.
    A different revision, corrupt cache, or unpinned ref is an explicit error;
    callers can move/remove the raw cache before changing data provenance.
    """
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("HarmBench revision must be a full 40-character Git commit SHA")
    revision = revision.lower()
    raw_dir = Path(root) / "data" / "raw"
    path = raw_dir / "harmbench_behaviors_text_all.csv"
    manifest_path = raw_dir / "manifest.json"
    source_url = (
        f"https://raw.githubusercontent.com/centerforaisafety/HarmBench/{revision}/{SOURCE_PATH}"
    )
    if path.exists() or manifest_path.exists():
        if not (path.is_file() and manifest_path.is_file()):
            raise RuntimeError("Incomplete HarmBench cache; remove data/raw files and retry")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if (
                manifest["revision"] != revision
                or manifest["source_url"] != source_url
                or manifest["sha256"] != digest
                or manifest["bytes"] != path.stat().st_size
            ):
                raise RuntimeError("HarmBench cache provenance/hash mismatch; inspect data/raw")
            _reader(path.read_text(encoding="utf-8-sig"))
            manifest["path"] = str(path.resolve())
            return manifest
        except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError("Invalid HarmBench cache; inspect data/raw before retrying") from exc

    request = Request(source_url, headers={"User-Agent": "harmbench-steering/0.1"})
    content = None
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            with urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
                content = response.read(MAX_DOWNLOAD_BYTES + 1)
            if len(content) > MAX_DOWNLOAD_BYTES:
                raise RuntimeError("HarmBench CSV exceeds the download size limit")
            # Validate format before committing either cache file.
            parsed = _reader(content.decode("utf-8-sig"))
            if not any(True for _ in parsed):
                raise ValueError("HarmBench CSV has no records")
            break
        except HTTPError as exc:
            if exc.code not in (408, 429, 500, 502, 503, 504):
                raise RuntimeError(f"HarmBench download failed (HTTP {exc.code})") from exc
            if attempt == DOWNLOAD_ATTEMPTS - 1:
                raise RuntimeError("HarmBench download failed after retries") from exc
        except (URLError, TimeoutError, OSError) as exc:
            if attempt == DOWNLOAD_ATTEMPTS - 1:
                raise RuntimeError("HarmBench download failed after retries") from exc
        except (UnicodeError, ValueError, csv.Error) as exc:
            raise RuntimeError("Downloaded HarmBench CSV is invalid") from exc
        time.sleep(2**attempt)
    assert content is not None
    manifest = {
        "dataset": "centerforaisafety/HarmBench",
        "revision": revision,
        "source_url": source_url,
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "path": str(path.resolve()),
    }
    _atomic_write(path, content)
    _atomic_write(manifest_path, (json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    return manifest


def load_behaviors(path: Path, categories: tuple[str, ...] = DEFAULT_CATEGORIES) -> list[dict]:
    """Read supported text behavior categories; preserve context and source tags."""
    if not categories or set(categories) - set(DEFAULT_CATEGORIES):
        raise ValueError("Supported functional categories are standard and contextual")
    try:
        reader = _reader(Path(path).read_text(encoding="utf-8-sig"))
        records = []
        seen = set()
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed HarmBench CSV row {line_number}")
            category = row["FunctionalCategory"].strip()
            if category not in categories:
                continue
            behavior_id = row["BehaviorID"].strip()
            if not behavior_id or not row["Behavior"].strip():
                raise ValueError(f"Empty behavior ID or text in CSV row {line_number}")
            if behavior_id in seen:
                raise ValueError(f"Duplicate behavior ID in CSV row {line_number}")
            if category == "contextual" and not row["ContextString"].strip():
                raise ValueError(f"Missing context for contextual CSV row {line_number}")
            seen.add(behavior_id)
            records.append(
                {
                    "behavior_id": behavior_id,
                    "behavior": row["Behavior"],
                    "context": row["ContextString"],
                    "functional_category": category,
                    "semantic_category": row["SemanticCategory"],
                    "tags": row["Tags"],
                }
            )
    except (UnicodeError, csv.Error) as exc:
        raise ValueError("Could not parse HarmBench CSV") from exc
    if not records:
        raise ValueError("No supported HarmBench behaviors were selected")
    return records


def _record_ids(records: list[dict]) -> list[str]:
    ids = [record["behavior_id"] for record in records]
    if any(not isinstance(item, str) or not item.strip() for item in ids):
        raise ValueError("Behavior IDs must be nonempty strings")
    if len(ids) != len(set(ids)):
        raise ValueError("Behavior records contain duplicate IDs")
    return ids


def write_label_template(records: list[dict], path: Path) -> None:
    """Create an exclusive, blank severity-label template without overwriting."""
    ids = sorted(_record_ids(records))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("behavior_id", "level", "rationale"))
            writer.writeheader()
            writer.writerows({"behavior_id": item, "level": "", "rationale": ""} for item in ids)
    except FileExistsError as exc:
        raise ValueError(f"Label file already exists; refusing to overwrite: {path}") from exc


def load_labels(path: Path, records: list[dict]) -> dict[str, int]:
    """Require exactly one explicit level (1, 2, or 3) and rationale per ID."""
    expected = set(_record_ids(records))
    labels = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = {"behavior_id", "level", "rationale"} - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError("Label CSV requires behavior_id, level, rationale columns")
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed label CSV row {line_number}")
            behavior_id = row["behavior_id"].strip()
            if behavior_id in labels:
                raise ValueError(f"Duplicate label ID in row {line_number}")
            if behavior_id not in expected:
                raise ValueError(f"Unknown behavior ID in label row {line_number}")
            level = row["level"].strip()
            if level not in ("1", "2", "3"):
                raise ValueError(f"Label row {line_number} needs an explicit level 1, 2, or 3")
            if not row["rationale"].strip():
                raise ValueError(f"Label row {line_number} needs a nonempty rationale")
            labels[behavior_id] = int(level)
    missing = expected - labels.keys()
    if missing:
        raise ValueError(f"Missing severity labels for {len(missing)} behaviors")
    return labels


def _normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _groups(records: list[dict], labels: dict[str, int]) -> list[list[str]]:
    ids = _record_ids(records)
    if set(ids) != set(labels):
        raise ValueError("Severity labels must exactly cover behavior records")
    if any(type(value) is not int or value not in (1, 2, 3) for value in labels.values()):
        raise ValueError("Severity labels must be integers 1, 2, or 3")
    # Connected components retain both repeated requests and repeated nonempty
    # context snippets. Empty context never connects standard behaviors.
    parents = {item: item for item in ids}

    def find(item):
        while parents[item] != item:
            parents[item] = parents[parents[item]]
            item = parents[item]
        return item

    def union(first, second):
        first, second = find(first), find(second)
        if first != second:
            parents[max(first, second)] = min(first, second)

    text_owners, context_owners = {}, {}
    for record in records:
        item = record["behavior_id"]
        text = _normalize_text(record["behavior"])
        if not text:
            raise ValueError("Cannot split an empty behavior")
        if text in text_owners:
            other = text_owners[text]
            if labels[item] != labels[other]:
                raise ValueError("Duplicate behavior group has conflicting severity labels")
            union(item, other)
        else:
            text_owners[text] = item
        context = _normalize_text(record.get("context", ""))
        if context:
            if context in context_owners:
                union(item, context_owners[context])
            else:
                context_owners[context] = item
    groups = {}
    for item in sorted(ids):
        groups.setdefault(find(item), []).append(item)
    return sorted(groups.values())


def behavior_groups(records: list[dict], labels: dict[str, int]) -> list[list[str]]:
    """Return stable connected components for normalized text/shared context.

    This catches exact duplicates after Unicode/case/whitespace normalization;
    it does not claim to detect semantic paraphrases.
    """
    return _groups(records, labels)


def _validate_fold_count(n_folds: int) -> None:
    if type(n_folds) is not int or not 2 <= n_folds <= 10:
        raise ValueError("n_folds must be an integer between 2 and 10")


def make_folds(
    records: list[dict],
    labels: dict[str, int],
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Build deterministic grouped, approximately stratified OOF folds.

    Every behavior is held out once. Each fold's fit is the exact complement
    of its held-out test IDs; no inner dev set or test-based tuning is implied.
    Both fit and test contain all three severity levels. Large shared-context
    groups can prevent exactly equal row counts, so grouping takes priority.
    """
    _validate_fold_count(n_folds)
    if type(seed) is not int:
        raise ValueError("Fold seed must be an integer")
    groups = behavior_groups(records, labels)
    rng = random.Random(seed)
    by_mask = {mask: [] for mask in range(1, 8)}
    for group in groups:
        mask = 0
        for item in group:
            mask |= 1 << (labels[item] - 1)
        by_mask[mask].append(group)
    for mask_groups in by_mask.values():
        rng.shuffle(mask_groups)
        mask_groups.sort(key=len, reverse=True)
    anchors = _coverage_anchors(tuple(len(by_mask[mask]) for mask in range(1, 8)), n_folds)
    test_ids = [[] for _ in range(n_folds)]
    counts = [[0, 0, 0] for _ in range(n_folds)]

    def assign(fold, group):
        test_ids[fold].extend(group)
        for item in group:
            counts[fold][labels[item] - 1] += 1

    for fold, mask in anchors:
        assign(fold, by_mask[mask].pop())
    targets = [sum(value == level for value in labels.values()) / n_folds for level in (1, 2, 3)]
    remaining = [group for mask_groups in by_mask.values() for group in mask_groups]
    rng.shuffle(remaining)
    remaining.sort(key=len, reverse=True)
    for group in remaining:
        contribution = [sum(labels[item] == level for item in group) for level in (1, 2, 3)]

        def cost(fold):
            return sum(
                (
                    (counts[fold][level] + contribution[level] - targets[level]) ** 2
                    - (counts[fold][level] - targets[level]) ** 2
                )
                / max(targets[level], 1.0)
                for level in range(3)
            )

        assign(min(range(n_folds), key=lambda fold: (cost(fold), fold)), group)
    all_ids = set(labels)
    result = {
        "n_folds": n_folds,
        "folds": [
            {"fold": index + 1, "fit": sorted(all_ids - set(ids)), "test": sorted(ids)}
            for index, ids in enumerate(test_ids)
        ],
        "group_ids": {item: group[0] for group in groups for item in group},
    }
    validate_folds(records, labels, result)
    return result


def validate_folds(records: list[dict], labels: dict[str, int], harmbench: dict) -> None:
    """Validate fold structure, held-out coverage, grouping, and fit complements."""
    groups = behavior_groups(records, labels)
    if not isinstance(harmbench, dict) or set(harmbench) != {"n_folds", "folds", "group_ids"}:
        raise ValueError("HarmBench folds require n_folds, folds, and group_ids")
    n_folds = harmbench["n_folds"]
    _validate_fold_count(n_folds)
    folds = harmbench["folds"]
    if not isinstance(folds, list) or len(folds) != n_folds:
        raise ValueError("HarmBench fold count does not match n_folds")
    expected_groups = {item: group[0] for group in groups for item in group}
    if harmbench["group_ids"] != expected_groups:
        raise ValueError("HarmBench group IDs differ from normalized behavior/context groups")
    all_ids = set(labels)
    test_owners = {}
    for number, fold in enumerate(folds, start=1):
        if (
            not isinstance(fold, dict)
            or set(fold) != {"fold", "fit", "test"}
            or type(fold["fold"]) is not int
            or fold["fold"] != number
        ):
            raise ValueError("HarmBench folds must have unique ordered fold numbers starting at 1")
        for name in ("fit", "test"):
            ids = fold[name]
            if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
                raise ValueError("Fold IDs must be lists of behavior ID strings")
            if len(ids) != len(set(ids)):
                raise ValueError(f"Behavior ID occurs more than once in fold {number} {name}")
            if not set(ids) <= all_ids:
                raise ValueError(f"Unknown behavior ID in fold {number} {name}")
            if {labels[item] for item in ids} != {1, 2, 3}:
                raise ValueError(f"Fold {number} {name} must contain all three severity levels")
        if set(fold["fit"]) != all_ids - set(fold["test"]):
            raise ValueError(f"Fold {number} fit must be the exact complement of its test IDs")
        for item in fold["test"]:
            if item in test_owners:
                raise ValueError("Behavior ID is held out more than once across folds")
            test_owners[item] = number
    if set(test_owners) != all_ids:
        raise ValueError("OOF test folds must cover every labeled behavior exactly once")
    for group in groups:
        if len({test_owners[item] for item in group}) != 1:
            raise ValueError("Duplicate behavior text or shared context leaks between folds")


def _coverage_anchors(mask_counts: tuple[int, ...], split_count: int) -> list[tuple[int, int]]:
    """Find a small exact assignment giving each split all severity levels.

    Only the seven possible level-presence masks matter to coverage, so this
    bounded search needs at most three assignments per split/fold.
    """
    failed = set()

    def search(counts, coverage):
        if all(mask == 7 for mask in coverage):
            return []
        state = (counts, coverage)
        if state in failed:
            return None
        for level in (1, 2, 4):
            available = sum(counts[mask - 1] for mask in range(1, 8) if mask & level)
            missing = sum(not (mask & level) for mask in coverage)
            if available < missing:
                failed.add(state)
                return None
        # Resolve scarce levels first, breaking ties by split/level order.
        pairs = [
            (sum(counts[mask - 1] for mask in range(1, 8) if mask & level), split, level)
            for split, covered in enumerate(coverage)
            for level in (1, 2, 4)
            if not (covered & level)
        ]
        _, split, level = min(pairs)
        masks = [mask for mask in range(1, 8) if counts[mask - 1] and mask & level]
        masks.sort(key=lambda mask: (-((mask & ~coverage[split]).bit_count()), mask))
        for mask in masks:
            new_counts = list(counts)
            new_counts[mask - 1] -= 1
            new_coverage = list(coverage)
            new_coverage[split] |= mask
            result = search(tuple(new_counts), tuple(new_coverage))
            if result is not None:
                return [(split, mask), *result]
        failed.add(state)
        return None

    anchors = search(mask_counts, (0,) * split_count)
    if anchors is None:
        raise ValueError(
            "Each severity level needs enough distinct behavior groups for all splits; "
            "duplicate behavior/context groups make coverage impossible"
        )
    return anchors


def make_splits(
    records: list[dict],
    labels: dict[str, int],
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> dict[str, list[str]]:
    """Stratify duplicate groups by severity with stable, seeded allocation.

    Behavior/context components stay intact; ratios target per-level row counts
    and may be approximate when large duplicate groups constrain allocation.
    Every positive-ratio split must contain each of the three severity levels.
    The same seed gives identical splits independent of input record ordering.
    """
    if len(ratios) != 3 or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in ratios
    ):
        raise ValueError("Split ratios must be three finite nonnegative numbers")
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
        raise ValueError("Split ratios must sum to 1")
    if type(seed) is not int:
        raise ValueError("Split seed must be an integer")
    groups = _groups(records, labels)
    rng = random.Random(seed)
    splits = {name: [] for name in SPLIT_NAMES}
    active = [index for index, ratio in enumerate(ratios) if ratio > 0]
    by_mask = {mask: [] for mask in range(1, 8)}
    for group in groups:
        mask = 0
        for item in group:
            mask |= 1 << (labels[item] - 1)
        by_mask[mask].append(group)
    for mask_groups in by_mask.values():
        rng.shuffle(mask_groups)
        # Small groups serve as coverage anchors; keep large groups available
        # for allocation toward the larger requested split.
        mask_groups.sort(key=len, reverse=True)
    anchors = _coverage_anchors(tuple(len(by_mask[mask]) for mask in range(1, 8)), len(active))
    level_counts = [[0, 0, 0] for _ in SPLIT_NAMES]

    def assign(split, group):
        splits[SPLIT_NAMES[split]].extend(group)
        for item in group:
            level_counts[split][labels[item] - 1] += 1

    for slot, mask in anchors:
        assign(active[slot], by_mask[mask].pop())
    totals = [sum(value == level for value in labels.values()) for level in (1, 2, 3)]
    targets = [[ratios[split] * total for total in totals] for split in range(3)]
    remaining = [group for mask_groups in by_mask.values() for group in mask_groups]
    rng.shuffle(remaining)
    remaining.sort(key=len, reverse=True)
    for group in remaining:
        contribution = [sum(labels[item] == level for item in group) for level in (1, 2, 3)]

        def cost(split):
            return sum(
                (
                    (level_counts[split][level] + contribution[level] - targets[split][level]) ** 2
                    - (level_counts[split][level] - targets[split][level]) ** 2
                )
                / max(targets[split][level], 1.0)
                for level in range(3)
            )

        assign(min(active, key=lambda split: (cost(split), split)), group)
    for ids in splits.values():
        ids.sort()
    validate_splits(
        records,
        labels,
        splits,
        required_splits=tuple(name for name, ratio in zip(SPLIT_NAMES, ratios) if ratio > 0),
    )
    return splits


def validate_splits(
    records: list[dict],
    labels: dict[str, int],
    splits: dict[str, list[str]],
    required_splits: tuple[str, ...] = SPLIT_NAMES,
) -> None:
    """Reject omitted/repeated IDs, behavior/context leakage, and missing levels."""
    groups = _groups(records, labels)
    if set(splits) != set(SPLIT_NAMES):
        raise ValueError("Splits must contain exactly fit, dev, and test keys")
    if set(required_splits) - set(SPLIT_NAMES):
        raise ValueError("Unknown required split")
    owners = {}
    for name, ids in splits.items():
        if not isinstance(ids, list):
            raise ValueError("Split IDs must be lists")
        for behavior_id in ids:
            if behavior_id not in labels:
                raise ValueError(f"Unknown behavior ID in {name} split")
            if behavior_id in owners:
                raise ValueError("Behavior ID occurs more than once across splits")
            owners[behavior_id] = name
        if name in required_splits and {labels[item] for item in ids} != {1, 2, 3}:
            raise ValueError(f"Split {name} must contain all three severity levels")
    if set(owners) != set(labels):
        raise ValueError("Splits must cover every labeled behavior exactly once")
    for group in groups:
        if len({owners[item] for item in group}) != 1:
            raise ValueError("Duplicate behavior text or context leaks between splits")
