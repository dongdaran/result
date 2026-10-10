"""Pinned, fit-free JailbreakBench transfer evaluation and overlap screening.

Only harmful rows marked Original upstream are candidates. Severity and the
explicit semantic-overlap exclusions are project AI-initial annotations. The
text screen catches exact and close wording; it cannot certify semantic novelty.
"""

from __future__ import annotations

import csv
import json
import re
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from .data import behavior_groups, load_labels
from .storage import atomic_bytes, atomic_json, digest_bytes, file_digest, fingerprint, run_lock

SOURCE = "jailbreakbench_original"
DATASET_REPO = "JailbreakBench/JBB-Behaviors"
NEAR_TEXT_THRESHOLD = 0.85
SCREEN_VERSION = "nfkc-casefold-whitespace+sequence-ratio-v1"
COLUMNS = {"Index", "Goal", "Target", "Behavior", "Category", "Source"}


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _settings(config: dict) -> dict | None:
    settings = config.get("external", {})
    if not settings.get("enabled", False):
        return None
    if settings.get("source") != SOURCE:
        raise ValueError("External source must be jailbreakbench_original")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", settings.get("revision", "")):
        raise ValueError("External revision must be a full commit SHA")
    for key in ("harmful_sha256", "benign_sha256"):
        if not re.fullmatch(r"[0-9a-fA-F]{64}", settings.get(key, "")):
            raise ValueError(f"External {key} must be a SHA-256")
    if not isinstance(settings.get("labels_path"), str) or not settings["labels_path"]:
        raise ValueError("External labels_path is required")
    return settings


def _raw_paths(root: Path) -> tuple[Path, Path]:
    directory = root / "data/raw/external"
    return directory / "harmful-behaviors.csv", directory / "benign-behaviors.csv"


def _download(url: str, path: Path, expected: str) -> None:
    if path.exists():
        if file_digest(path) != expected:
            raise ValueError(f"External raw checksum mismatch: {path.name}")
        return
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "harmbench-steering/0.1"})
            with urllib.request.urlopen(request, timeout=45) as response:
                content = response.read(10 * 1024 * 1024 + 1)
            if len(content) > 10 * 1024 * 1024 or digest_bytes(content) != expected:
                raise ValueError(f"External downloaded checksum mismatch: {path.name}")
            atomic_bytes(path, content)
            return
        except urllib.error.HTTPError as exc:
            if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == 2:
                raise RuntimeError(f"External download failed (HTTP {exc.code})") from exc
        except (OSError, TimeoutError) as exc:
            if attempt == 2:
                raise RuntimeError("External download failed after retries") from exc
        time.sleep(attempt + 1)


def _rows(path: Path, kind: str) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or ()) != COLUMNS:
            raise ValueError("Unexpected JailbreakBench CSV schema")
        records, seen = [], set()
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed JailbreakBench CSV row")
            index = row["Index"]
            if not index.isdigit() or index != str(int(index)) or index in seen:
                raise ValueError("Invalid or duplicate JailbreakBench index")
            if any(not row[key].strip() for key in ("Goal", "Behavior", "Category", "Source")):
                raise ValueError("Blank JailbreakBench field")
            if row["Source"] not in {"Original", "AdvBench", "TDC/HarmBench"}:
                raise ValueError("Unknown JailbreakBench source provenance")
            seen.add(index)
            records.append(
                {
                    "behavior_id": f"jbb_{kind}:{index}",
                    "behavior": row["Goal"],
                    "context": "",
                    "functional_category": "standard",
                    "semantic_category": row["Category"],
                    "upstream_behavior": row["Behavior"],
                    "upstream_source": row["Source"],
                    "dataset": SOURCE if kind == "harmful" else "jailbreakbench_benign",
                }
            )
    if not records:
        raise ValueError("Empty JailbreakBench dataset")
    return sorted(records, key=lambda record: int(record["behavior_id"].split(":")[1]))


def _reviewed_exclusions(path: Path, original_ids: set[str], reference_ids: set[str]) -> dict:
    exclusions = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            item = row["behavior_id"].strip()
            if item not in original_ids:
                raise ValueError("Unknown external review ID")
            excluded = row.get("exclude", "false").strip().lower()
            if excluded not in {"true", "false"}:
                raise ValueError("External annotation exclude must be true or false")
            if excluded == "true":
                reason = row.get("overlap_rationale", "").strip()
                reference = row.get("overlap_with", "").strip()
                if not reason or reference not in reference_ids:
                    raise ValueError("External reviewed exclusion needs a valid HB ID and rationale")
                exclusions[item] = {"reason": "reviewed_semantic_overlap", "with_id": reference,
                                    "detail": reason}
    return exclusions


def _references(harmful_records: list[dict], benign_fit: list[dict]) -> list[dict]:
    references = [
        {"id": item["behavior_id"], "behavior": item["behavior"],
         "context": item.get("context", ""), "dataset": "harmbench"}
        for item in harmful_records
    ] + [
        {"id": "xstest_fit:" + item["benign_id"], "behavior": item["prompt"],
         "context": "", "dataset": "xstest_fit"}
        for item in benign_fit
    ]
    if not references or any(not _normalize(item["behavior"]) for item in references):
        raise ValueError("External screening needs nonempty fit references")
    return sorted(references, key=lambda item: (item["dataset"], item["id"]))


def _overlap(record: dict, references: list[dict]) -> dict | None:
    text, context = _normalize(record["behavior"]), _normalize(record["context"])
    for reference in references:
        other = _normalize(reference["behavior"])
        if text == other or (context and context == _normalize(reference["context"])):
            return {"reason": "exact_normalized_overlap", "with_id": reference["id"]}
    closest = max(
        ((SequenceMatcher(None, text, _normalize(item["behavior"]), autojunk=False).ratio(), item)
         for item in references), key=lambda pair: pair[0],
    )
    if closest[0] >= NEAR_TEXT_THRESHOLD:
        return {"reason": "near_text_overlap", "with_id": closest[1]["id"],
                "similarity": round(closest[0], 6)}
    return None


def _jsonl(records: list[dict]) -> bytes:
    return b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()
                    for row in records)


def _build(root: Path, settings: dict, harmful_records: list[dict], benign_fit: list[dict]):
    harmful_path, benign_path = _raw_paths(root)
    for path, expected in ((harmful_path, settings["harmful_sha256"]),
                           (benign_path, settings["benign_sha256"])):
        if not path.is_file() or file_digest(path) != expected:
            raise ValueError(f"External raw checksum mismatch or missing: {path.name}")
    raw_harmful, raw_benign = _rows(harmful_path, "harmful"), _rows(benign_path, "benign")
    original = [record for record in raw_harmful if record["upstream_source"] == "Original"]
    labels_path = root / settings["labels_path"]
    labels = load_labels(labels_path, original)
    references = _references(harmful_records, benign_fit)
    reviewed = _reviewed_exclusions(labels_path, set(labels),
                                     {item["behavior_id"] for item in harmful_records})
    exclusions = [
        {"behavior_id": item["behavior_id"], "reason": "inherited_source",
         "upstream_source": item["upstream_source"]}
        for item in raw_harmful if item["upstream_source"] != "Original"
    ]
    selected_harmful, selected_benign = [], []
    for records, selected in ((original, selected_harmful), (raw_benign, selected_benign)):
        seen = {}
        for record in records:
            item = record["behavior_id"]
            overlap = reviewed.get(item) or _overlap(record, references)
            text = _normalize(record["behavior"])
            if overlap is None and text in seen:
                overlap = {"reason": "external_exact_duplicate", "with_id": seen[text]}
            if overlap:
                exclusions.append({"behavior_id": item, **overlap})
                continue
            seen[text] = item
            selected.append({**record, "level": labels[item] if records is original else 0})
    if set(record["level"] for record in selected_harmful) != {1, 2, 3}:
        raise ValueError("External harmful evaluation must retain all three severity levels")
    if not selected_benign:
        raise ValueError("External benign evaluation is empty after overlap screening")
    # Screen external safe data against the selected harmful requests as well.
    harmful_texts = {_normalize(item["behavior"]): item["behavior_id"]
                     for item in selected_harmful}
    if any(_normalize(item["behavior"]) in harmful_texts for item in selected_benign):
        raise ValueError("External harmful and benign requests overlap")
    groups = behavior_groups(selected_harmful,
                             {record["behavior_id"]: record["level"] for record in selected_harmful})
    group_ids = {item: group[0] for group in groups for item in group}
    selected_harmful = [{**record, "group_id": group_ids[record["behavior_id"]]}
                        for record in selected_harmful]
    selected_benign = [{**record, "group_id": record["behavior_id"]}
                       for record in selected_benign]
    signature = {
        "external_config": settings,
        "labels_sha256": file_digest(labels_path),
        "references_sha256": fingerprint(references),
        "screen_version": SCREEN_VERSION,
        "near_text_threshold": NEAR_TEXT_THRESHOLD,
    }
    summary = {
        "schema_version": 1,
        "fingerprint": fingerprint(signature),
        "provenance": signature,
        "source_repository": DATASET_REPO,
        "source_revision": settings["revision"],
        "harmful_total": len(selected_harmful),
        "benign_total": len(selected_benign),
        "harmful_original_candidates": len(original),
        "level_counts": dict(sorted(Counter(record["level"] for record in selected_harmful).items())),
        "group_count": len(groups),
        "harmful_ids": [record["behavior_id"] for record in selected_harmful],
        "benign_ids": [record["behavior_id"] for record in selected_benign],
        "processed_harmful_sha256": digest_bytes(_jsonl(selected_harmful)),
        "processed_benign_sha256": digest_bytes(_jsonl(selected_benign)),
        "exclusions": sorted(exclusions, key=lambda item: item["behavior_id"]),
        "label_provenance": "AI_initial_v1; project severity, not official JBB or human-verified",
        "overlap_limitation": "Provenance filtering, exact and close wording screens and explicit "
                              "AI-reviewed exclusions do not establish semantic independence.",
    }
    return selected_harmful, selected_benign, summary


def _validate_cache(directory: Path, summary: dict) -> None:
    path = directory / "manifest.json"
    if not path.is_file():
        raise ValueError("External manifest missing; run ./scripts/download first")
    if fingerprint(json.loads(path.read_text())) != fingerprint(summary):
        raise ValueError("External data/config/labels/references changed; inspect external manifest")
    for name, key in (("harmful.jsonl", "processed_harmful_sha256"),
                      ("benign.jsonl", "processed_benign_sha256")):
        path = directory / name
        if not path.is_file() or file_digest(path) != summary[key]:
            raise ValueError(f"External processed artifact changed or missing: {name}")


def prepare_external(root: Path, config: dict, harmful_records: list[dict],
                     benign_fit: list[dict], *, resplit: bool = False) -> dict:
    """Download pinned data, freeze selection, reject changes to an existing snapshot."""
    settings = _settings(config)
    if settings is None:
        return {"enabled": False}
    root = Path(root)
    directory = root / "data/processed/external"
    with run_lock(directory):
        for path, key in zip(_raw_paths(root), ("harmful_sha256", "benign_sha256")):
            url = (f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/"
                   f"{settings['revision']}/data/{path.name}")
            _download(url, path, settings[key])
        harmful, benign, summary = _build(root, settings, harmful_records, benign_fit)
        if (directory / "manifest.json").exists() and not resplit:
            _validate_cache(directory, summary)
        else:
            atomic_bytes(directory / "harmful.jsonl", _jsonl(harmful))
            atomic_bytes(directory / "benign.jsonl", _jsonl(benign))
            atomic_json(directory / "manifest.json", summary)
        return summary


def read_external(root: Path, config: dict, harmful_records: list[dict],
                  benign_fit: list[dict]) -> tuple[list[dict], list[dict]]:
    """Validate raw and labels against the frozen selection without downloading."""
    settings = _settings(config)
    if settings is None:
        return [], []
    harmful, benign, summary = _build(Path(root), settings, harmful_records, benign_fit)
    _validate_cache(Path(root) / "data/processed/external", summary)
    return harmful, benign
