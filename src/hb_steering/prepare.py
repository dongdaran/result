from __future__ import annotations

import csv
import json
import random
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from .data import download_harmbench, load_behaviors, load_labels, make_splits, validate_splits
from .storage import atomic_bytes, atomic_json, file_digest, fingerprint, run_lock


def checked_download(url: str, path: Path, expected_sha256: str) -> None:
    if path.exists():
        if file_digest(path) != expected_sha256:
            raise ValueError(
                f"Checksum mismatch: {path}; remove the corrupted cached file and retry"
            )
        return
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "harmbench-steering/0.1"})
            with urllib.request.urlopen(request, timeout=45) as response:
                data = response.read()
            from .storage import digest_bytes

            if digest_bytes(data) != expected_sha256:
                raise ValueError(f"Downloaded file checksum mismatch: {path.name}")
            atomic_bytes(path, data)
            return
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def load_benign(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        raw = list(csv.DictReader(stream))
    if not raw or not {"id", "prompt", "type", "label"} <= raw[0].keys():
        raise ValueError("Unexpected XSTest CSV schema")
    records = [
        {"benign_id": str(r["id"]), "prompt": r["prompt"], "type": r["type"]}
        for r in raw
        if r["label"] == "safe"
    ]
    if len(records) != len({r["benign_id"] for r in records}):
        raise ValueError("Duplicate benign IDs")
    if any(not r["prompt"].strip() for r in records):
        raise ValueError("Blank benign prompt")
    normalized = [" ".join(r["prompt"].casefold().split()) for r in records]
    if len(normalized) != len(set(normalized)):
        raise ValueError("Duplicate benign prompts must be grouped before splitting")
    if not records:
        raise ValueError("No safe XSTest prompts")
    return records


def benign_splits(records: list[dict], seed: int, ratios: list[float]) -> dict:
    groups = defaultdict(list)
    for record in records:
        groups[record["type"]].append(record["benign_id"])
    result = {"fit": [], "dev": [], "test": []}
    rng = random.Random(seed)
    for kind in sorted(groups):
        ids = sorted(groups[kind])
        rng.shuffle(ids)
        first, second = round(len(ids) * ratios[0]), round(len(ids) * ratios[1])
        for split, values in zip(
            result, (ids[:first], ids[first : first + second], ids[first + second :])
        ):
            result[split].extend(values)
    if any(not ids for ids in result.values()):
        raise ValueError("Benign splits must all be nonempty")
    return {key: sorted(ids) for key, ids in result.items()}


def dataset_signature(root: Path, config: dict) -> dict:
    ds = config["dataset"]
    for name, expected in (
        ("harmbench_behaviors_text_all.csv", ds["harmbench_sha256"]),
        ("xstest_prompts.csv", ds["xstest_sha256"]),
    ):
        path = root / "data/raw" / name
        if not path.exists():
            raise ValueError("Prepared data missing; run ./scripts/download first")
        if file_digest(path) != expected:
            raise ValueError(f"Raw data checksum mismatch: {name}")
    return {"dataset_config": ds, "labels_sha256": file_digest(root / ds["labels_path"])}


def prepare_dataset(root: Path, config: dict, *, resplit: bool = False) -> dict:
    ds = config["dataset"]
    # The downloader and fixed-split mutation share a process lock.
    with run_lock(root / "data/processed"):
        manifest = download_harmbench(root, ds["harmbench_revision"])
        if manifest["sha256"] != ds["harmbench_sha256"]:
            raise ValueError("HarmBench checksum differs from the pinned config")
        checked_download(
            "https://raw.githubusercontent.com/paul-rottger/xstest/"
            f"{ds['xstest_revision']}/xstest_prompts.csv",
            root / "data/raw/xstest_prompts.csv",
            ds["xstest_sha256"],
        )
        records = load_behaviors(
            root / "data/raw/harmbench_behaviors_text_all.csv", tuple(ds["categories"])
        )
        labels = load_labels(root / ds["labels_path"], records)
        benign = load_benign(root / "data/raw/xstest_prompts.csv")
        signature = dataset_signature(root, config)
        path = root / ds["splits_path"]
        if path.exists() and not resplit:
            saved = json.loads(path.read_text())
            if saved.get("fingerprint") != fingerprint(signature):
                raise ValueError(
                    "Data/labels/split settings changed; use ./scripts/download --resplit"
                )
            validate_splits(records, labels, saved["harmbench"])
        else:
            saved = {
                "schema_version": 1,
                "fingerprint": fingerprint(signature),
                "provenance": signature,
                "harmbench": make_splits(records, labels, ds["seed"], tuple(ds["ratios"])),
                "benign": benign_splits(benign, ds["seed"], ds["ratios"]),
            }
            atomic_json(path, saved)
        validate_benign_splits(benign, saved["benign"])
        atomic_bytes(
            root / "data/processed/harmbench.jsonl",
            b"".join(
                (
                    json.dumps({**r, "level": labels[r["behavior_id"]]}, ensure_ascii=False) + "\n"
                ).encode()
                for r in records
            ),
        )
        atomic_bytes(
            root / "data/processed/benign.jsonl",
            b"".join((json.dumps(r, ensure_ascii=False) + "\n").encode() for r in benign),
        )
        summary = {
            "fingerprint": saved["fingerprint"],
            "harmbench_total": len(records),
            "level_counts": dict(sorted(Counter(labels.values()).items())),
            "harmbench_splits": {
                s: dict(sorted(Counter(labels[i] for i in ids).items()))
                for s, ids in saved["harmbench"].items()
            },
            "benign_total": len(benign),
            "benign_splits": {s: len(ids) for s, ids in saved["benign"].items()},
        }
        atomic_json(root / "data/processed/summary.json", summary)
        return summary


def validate_benign_splits(records: list[dict], splits: dict) -> None:
    if set(splits) != {"fit", "dev", "test"}:
        raise ValueError("Invalid benign split names")
    ids = [key for values in splits.values() for key in values]
    if len(ids) != len(set(ids)) or set(ids) != {r["benign_id"] for r in records}:
        raise ValueError("Benign split IDs overlap or do not cover the dataset")
    if any(not values for values in splits.values()):
        raise ValueError("Empty benign split")


def read_prepared(root: Path, config: dict) -> tuple[list[dict], dict, list[dict], dict]:
    ds = config["dataset"]
    signature = dataset_signature(root, config)
    path = root / ds["splits_path"]
    if not path.exists():
        raise ValueError("Missing splits; run ./scripts/download first")
    saved = json.loads(path.read_text())
    if saved.get("schema_version") != 1 or saved.get("fingerprint") != fingerprint(signature):
        raise ValueError("Saved splits differ from data/config; run ./scripts/download --resplit")
    records = load_behaviors(
        root / "data/raw/harmbench_behaviors_text_all.csv", tuple(ds["categories"])
    )
    labels = load_labels(root / ds["labels_path"], records)
    validate_splits(records, labels, saved["harmbench"])
    benign = load_benign(root / "data/raw/xstest_prompts.csv")
    validate_benign_splits(benign, saved["benign"])
    return records, labels, benign, saved
