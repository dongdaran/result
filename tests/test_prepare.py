"""Offline end-to-end data preparation and provenance checks."""

import csv
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from hb_steering.config import load_config
from hb_steering.data import OFFICIAL_COLUMNS, make_folds
from hb_steering.prepare import (
    benign_splits,
    checked_download,
    load_benign,
    prepare_dataset,
    read_prepared,
    validate_benign_splits,
)


def encode_csv(columns, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        harmful = []
        annotations = []
        for level in (1, 2, 3):
            for index in range(3):
                item = f"example-{level}-{index}"
                category = "contextual" if level == 3 else "standard"
                harmful.append(
                    dict(
                        zip(
                            OFFICIAL_COLUMNS,
                            (
                                f"Harmless fixture request {level}-{index}",
                                category,
                                "fixture",
                                "",
                                f"Fixture context {index}" if level == 3 else "",
                                item,
                            ),
                        )
                    )
                )
                annotations.append(
                    {"behavior_id": item, "level": level, "rationale": "Fixture annotation"}
                )
        self.harmful_bytes = encode_csv(OFFICIAL_COLUMNS, harmful)
        benign_rows = [
            {
                "id": str(index),
                "prompt": f"Safe fixture request {index}",
                "type": "type_a" if index < 10 else "type_b",
                "label": "safe",
            }
            for index in range(20)
        ]
        benign_rows.extend(
            [
                {"id": "20", "prompt": "Excluded fixture 20", "type": "type_a", "label": "unsafe"},
                {"id": "21", "prompt": "Excluded fixture 21", "type": "type_b", "label": "unsafe"},
            ]
        )
        self.benign_bytes = encode_csv(("id", "prompt", "type", "label"), benign_rows)
        labels = self.root / "data/annotations/severity.csv"
        labels.parent.mkdir(parents=True)
        labels.write_bytes(encode_csv(("behavior_id", "level", "rationale"), annotations))
        self.config = {
            "dataset": {
                "harmbench_revision": "a" * 40,
                "harmbench_sha256": hashlib.sha256(self.harmful_bytes).hexdigest(),
                "xstest_revision": "b" * 40,
                "xstest_sha256": hashlib.sha256(self.benign_bytes).hexdigest(),
                "categories": ["standard", "contextual"],
                "labels_path": "data/annotations/severity.csv",
                "splits_path": "data/splits/splits.json",
                "seed": 42,
                "n_folds": 3,
                "benign_ratios": [0.6, 0.2, 0.2],
            }
        }

    def tearDown(self):
        self.temp.cleanup()

    @contextmanager
    def mock_downloads(self):
        with (
            patch(
                "hb_steering.data.urlopen", return_value=io.BytesIO(self.harmful_bytes)
            ) as harmful,
            patch(
                "hb_steering.prepare.urllib.request.urlopen",
                return_value=io.BytesIO(self.benign_bytes),
            ) as benign,
        ):
            yield harmful, benign

    def test_prepare_and_read_back_preserve_dataset_provenance_and_reuse_cache(self):
        with self.mock_downloads() as (harmful_open, benign_open):
            summary = prepare_dataset(self.root, self.config)
            again = prepare_dataset(self.root, self.config)
            self.assertEqual(summary, again)
            harmful_open.assert_called_once()
            benign_open.assert_called_once()
        self.assertEqual(summary["harmbench_total"], 9)
        self.assertEqual(summary["level_counts"], {1: 3, 2: 3, 3: 3})
        self.assertEqual(summary["schema_version"], 2)
        self.assertEqual(summary["n_folds"], 3)
        self.assertEqual(summary["pooled_test_counts"], {1: 3, 2: 3, 3: 3})
        self.assertEqual(len(summary["harmbench_folds"]), 3)
        for fold in summary["harmbench_folds"]:
            self.assertEqual(fold["fit"], {1: 2, 2: 2, 3: 2})
            self.assertEqual(fold["test"], {1: 1, 2: 1, 3: 1})
        self.assertEqual(summary["benign_total"], 20)
        self.assertEqual(summary["benign_splits"], {"fit": 12, "dev": 4, "test": 4})
        records, labels, benign, saved = read_prepared(self.root, self.config)
        self.assertEqual(len(records), len(labels))
        self.assertEqual(len(benign), 20)
        self.assertEqual(saved["fingerprint"], summary["fingerprint"])
        self.assertEqual(saved["schema_version"], 2)
        self.assertEqual(saved["provenance"]["dataset_config"], self.config["dataset"])
        self.assertEqual(
            saved["provenance"]["labels_sha256"],
            hashlib.sha256((self.root / "data/annotations/severity.csv").read_bytes()).hexdigest(),
        )
        self.assertTrue((self.root / "data/processed/harmbench.jsonl").is_file())
        self.assertTrue((self.root / "data/processed/benign.jsonl").is_file())

    def test_label_changes_refuse_existing_split_until_explicit_resplit(self):
        with self.mock_downloads():
            before = prepare_dataset(self.root, self.config)
        labels_path = self.root / "data/annotations/severity.csv"
        labels_path.write_text(
            labels_path.read_text().replace("Fixture annotation", "Reviewed fixture rationale")
        )
        with self.assertRaisesRegex(ValueError, "changed.*resplit"):
            prepare_dataset(self.root, self.config)
        with self.assertRaisesRegex(ValueError, "differ.*resplit"):
            read_prepared(self.root, self.config)
        after = prepare_dataset(self.root, self.config, resplit=True)
        self.assertNotEqual(before["fingerprint"], after["fingerprint"])

    def test_stale_fingerprint_and_modified_split_settings_are_rejected(self):
        with self.mock_downloads():
            prepare_dataset(self.root, self.config)
        split_path = self.root / self.config["dataset"]["splits_path"]
        saved = json.loads(split_path.read_text())
        saved["fingerprint"] = "0" * 64
        split_path.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, "changed.*resplit"):
            prepare_dataset(self.root, self.config)
        prepare_dataset(self.root, self.config, resplit=True)
        self.config["dataset"]["seed"] = 123
        with self.assertRaisesRegex(ValueError, "changed.*resplit"):
            prepare_dataset(self.root, self.config)

    def test_schema_one_requires_explicit_resplit_even_when_fingerprint_matches(self):
        with self.mock_downloads():
            prepare_dataset(self.root, self.config)
        path = self.root / self.config["dataset"]["splits_path"]
        saved = json.loads(path.read_text())
        saved["schema_version"] = 1
        path.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, "schema changed.*resplit"):
            prepare_dataset(self.root, self.config)
        with self.assertRaisesRegex(ValueError, "differ.*resplit"):
            read_prepared(self.root, self.config)
        summary = prepare_dataset(self.root, self.config, resplit=True)
        self.assertEqual(summary["schema_version"], 2)

    def test_modified_fold_structure_rejected_with_unchanged_fingerprint(self):
        with self.mock_downloads():
            prepare_dataset(self.root, self.config)
        path = self.root / self.config["dataset"]["splits_path"]
        saved = json.loads(path.read_text())
        saved["harmbench"]["folds"][0]["fit"].pop()
        path.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, "exact complement"):
            prepare_dataset(self.root, self.config)
        with self.assertRaisesRegex(ValueError, "exact complement"):
            read_prepared(self.root, self.config)

    def test_valid_alternate_fold_count_cannot_bypass_config_with_unchanged_fingerprint(self):
        with self.mock_downloads():
            prepare_dataset(self.root, self.config)
        records, labels, _, saved = read_prepared(self.root, self.config)
        saved["harmbench"] = make_folds(records, labels, n_folds=2)
        path = self.root / self.config["dataset"]["splits_path"]
        path.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, "fold count differs.*resplit"):
            prepare_dataset(self.root, self.config)
        with self.assertRaisesRegex(ValueError, "fold count differs.*resplit"):
            read_prepared(self.root, self.config)

    def test_corrupt_raw_cache_and_wrong_source_revision_are_rejected(self):
        with self.mock_downloads():
            prepare_dataset(self.root, self.config)
        raw = self.root / "data/raw/harmbench_behaviors_text_all.csv"
        original = raw.read_bytes()
        raw.write_bytes(original + b"corrupt")
        with self.assertRaisesRegex(RuntimeError, "provenance/hash mismatch"):
            prepare_dataset(self.root, self.config)
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            read_prepared(self.root, self.config)
        raw.write_bytes(original)
        self.config["dataset"]["harmbench_revision"] = "c" * 40
        with self.assertRaisesRegex(RuntimeError, "provenance/hash mismatch"):
            prepare_dataset(self.root, self.config)

    def test_pinned_content_hash_is_enforced_before_preparing_splits(self):
        self.config["dataset"]["harmbench_sha256"] = "0" * 64
        with self.mock_downloads(), self.assertRaisesRegex(ValueError, "checksum differs"):
            prepare_dataset(self.root, self.config)
        self.assertFalse((self.root / self.config["dataset"]["splits_path"]).exists())

    def test_unknown_annotation_id_is_rejected_without_silent_filtering(self):
        labels_path = self.root / "data/annotations/severity.csv"
        with labels_path.open("a") as stream:
            stream.write("unknown-id,1,Unknown fixture\n")
        with self.mock_downloads(), self.assertRaisesRegex(ValueError, "Unknown behavior ID"):
            prepare_dataset(self.root, self.config)

    def test_subset_categories_with_unfiltered_full_annotations_fail_explicitly(self):
        # Selecting a category subset requires deliberate annotation handling;
        # the strict loader must never silently discard unmatched labels.
        self.config["dataset"]["categories"] = ["standard"]
        with self.mock_downloads(), self.assertRaisesRegex(ValueError, "Unknown behavior ID"):
            prepare_dataset(self.root, self.config)

    def test_cached_invalid_benign_split_is_rejected(self):
        with self.mock_downloads():
            prepare_dataset(self.root, self.config)
        path = self.root / self.config["dataset"]["splits_path"]
        saved = json.loads(path.read_text())
        saved["benign"]["test"].append(saved["benign"]["fit"][0])
        path.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, "overlap"):
            prepare_dataset(self.root, self.config)
        with self.assertRaisesRegex(ValueError, "overlap"):
            read_prepared(self.root, self.config)

    def test_benign_filtering_splits_and_coverage_are_deterministic(self):
        path = self.root / "benign.csv"
        path.write_bytes(self.benign_bytes)
        records = load_benign(path)
        self.assertEqual(len(records), 20)
        self.assertNotIn("20", {record["benign_id"] for record in records})
        splits = benign_splits(records, 42, [0.6, 0.2, 0.2])
        self.assertEqual(splits, benign_splits(list(reversed(records)), 42, [0.6, 0.2, 0.2]))
        self.assertNotEqual(splits, benign_splits(records, 100, [0.6, 0.2, 0.2]))
        validate_benign_splits(records, splits)
        splits["test"].append(splits["fit"][0])
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_benign_splits(records, splits)

    def test_benign_duplicate_prompt_and_unexpected_schema_are_rejected(self):
        path = self.root / "benign.csv"
        path.write_bytes(
            encode_csv(
                ("id", "prompt", "type", "label"),
                [
                    {"id": "a", "prompt": "Example request", "type": "fixture", "label": "safe"},
                    {
                        "id": "b",
                        "prompt": "  EXAMPLE\nrequest ",
                        "type": "fixture",
                        "label": "safe",
                    },
                ],
            )
        )
        with self.assertRaisesRegex(ValueError, "Duplicate benign prompts"):
            load_benign(path)
        path.write_text("id,prompt\na,Example request\n")
        with self.assertRaisesRegex(ValueError, "schema"):
            load_benign(path)

    @patch("hb_steering.prepare.urllib.request.urlopen")
    def test_checked_download_reuses_only_checksum_verified_cache(self, open_mock):
        path = self.root / "cached.csv"
        path.write_bytes(self.benign_bytes)
        digest = hashlib.sha256(self.benign_bytes).hexdigest()
        checked_download("https://example.invalid/data.csv", path, digest)
        open_mock.assert_not_called()
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            checked_download("https://example.invalid/data.csv", path, digest)
        open_mock.assert_not_called()

    def test_default_config_accepts_placeholders_and_rejects_unknown_categories(self):
        template = Path(__file__).resolve().parents[1] / "configs/experiment.toml"
        path = self.root / "config.toml"
        text = template.read_text()
        path.write_text(text)
        config, resolved = load_config(self.root, "config.toml")
        self.assertEqual(resolved, path)
        self.assertEqual(config["models"]["model_1"]["hf_id"], "")
        self.assertEqual(config["dataset"]["n_folds"], 5)
        self.assertEqual(config["dataset"]["benign_ratios"], [0.6, 0.2, 0.2])
        for value in ("true", "1", "11", "5.0"):
            path.write_text(text.replace("n_folds = 5", f"n_folds = {value}"))
            with self.subTest(n_folds=value), self.assertRaisesRegex(ValueError, "n_folds"):
                load_config(self.root, "config.toml")
        path.write_text(
            text.replace('categories = ["standard", "contextual"]', 'categories = ["multimodal"]')
        )
        with self.assertRaisesRegex(ValueError, "standard.*contextual"):
            load_config(self.root, "config.toml")
        path.write_text(
            text.replace('categories = ["standard", "contextual"]', 'categories = ["standard"]')
        )
        with self.assertRaisesRegex(ValueError, "standard.*contextual"):
            load_config(self.root, "config.toml")


if __name__ == "__main__":
    unittest.main()
