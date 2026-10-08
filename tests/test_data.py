"""Data integrity checks use harmless fixtures and never require the network."""

import csv
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from hb_steering.data import (
    OFFICIAL_COLUMNS,
    download_harmbench,
    load_behaviors,
    load_labels,
    make_splits,
    validate_splits,
    write_label_template,
)

REVISION = "a" * 40


def csv_bytes(rows):
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=OFFICIAL_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def source_row(behavior_id="example", category="standard", context=""):
    return dict(
        zip(
            OFFICIAL_COLUMNS,
            (
                "Harmless test request",
                category,
                "fixture",
                "test",
                context,
                behavior_id,
            ),
        )
    )


def split_fixture():
    records = []
    labels = {}
    for level in (1, 2, 3):
        for index in range(10):
            behavior_id = f"l{level}-{index}"
            records.append(
                {
                    "behavior_id": behavior_id,
                    "behavior": f"Harmless example level {level} item {index}",
                    "context": "",
                }
            )
            labels[behavior_id] = level
    records.append(
        {
            "behavior_id": "duplicate",
            "behavior": "  HARMLESS\nexample level 1 item 0  ",
            "context": "Different context must not permit leakage",
        }
    )
    labels["duplicate"] = 1
    return records, labels


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_load_preserves_context_and_excludes_unsupported_categories(self):
        path = self.root / "source.csv"
        path.write_bytes(
            csv_bytes(
                [
                    source_row("standard"),
                    source_row(
                        "contextual", "contextual", "A fixture context, with a comma\nand newline"
                    ),
                    source_row("copyright", "copyright"),
                    source_row("image", "multimodal"),
                ]
            )
        )
        records = load_behaviors(path)
        self.assertEqual([record["behavior_id"] for record in records], ["standard", "contextual"])
        self.assertEqual(records[1]["context"], "A fixture context, with a comma\nand newline")
        self.assertEqual(records[0]["tags"], "test")
        self.assertEqual(records[0]["semantic_category"], "fixture")

    def test_load_rejects_duplicate_ids_missing_context_and_bad_columns(self):
        path = self.root / "source.csv"
        for rows, message in (
            ([source_row(), source_row()], "Duplicate"),
            ([source_row(category="contextual")], "Missing context"),
        ):
            path.write_bytes(csv_bytes(rows))
            with self.assertRaisesRegex(ValueError, message):
                load_behaviors(path)
        path.write_text("BehaviorID\nexample\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing columns"):
            load_behaviors(path)

    def test_blank_label_template_never_overwrites(self):
        path = self.root / "labels.csv"
        records = [{"behavior_id": "second"}, {"behavior_id": "first"}]
        write_label_template(records, path)
        original = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            write_label_template(records, path)
        self.assertEqual(path.read_bytes(), original)
        with self.assertRaisesRegex(ValueError, "explicit level"):
            load_labels(path, records)

    def test_labels_require_exact_coverage_valid_levels_and_rationale(self):
        path = self.root / "labels.csv"
        records = [{"behavior_id": "a"}, {"behavior_id": "b"}]
        path.write_text(
            "behavior_id,level,rationale\na,1,limited\nb,3,critical\n", encoding="utf-8"
        )
        self.assertEqual(load_labels(path, records), {"a": 1, "b": 3})
        cases = (
            ("a,1,limited\n", "Missing severity"),
            ("a,1,limited\nb,3,critical\nc,2,extra\n", "Unknown behavior"),
            ("a,1,limited\na,2,duplicate\n", "Duplicate label"),
            ("a,4,invalid\nb,3,critical\n", "explicit level"),
            ("a,1,\nb,3,critical\n", "nonempty rationale"),
        )
        for rows, message in cases:
            with self.subTest(message=message):
                path.write_text("behavior_id,level,rationale\n" + rows, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_labels(path, records)

    def test_splits_deterministic_stratified_and_group_duplicate_text(self):
        records, labels = split_fixture()
        splits = make_splits(records, labels)
        self.assertEqual(splits, make_splits(list(reversed(records)), labels))
        self.assertNotEqual(splits, make_splits(records, labels, seed=100))
        owners = {item: split for split, ids in splits.items() for item in ids}
        self.assertEqual(owners["duplicate"], owners["l1-0"])
        self.assertEqual(len(owners), len(records))
        for ids in splits.values():
            self.assertEqual({labels[item] for item in ids}, {1, 2, 3})
        validate_splits(records, labels, splits)

    def test_insufficient_groups_and_conflicting_duplicate_labels_rejected(self):
        records, labels = split_fixture()
        insufficient = [
            record
            for record in records
            if record["behavior_id"]
            in {
                "l1-0",
                "l1-1",
                "l2-0",
                "l2-1",
                "l3-0",
                "l3-1",
            }
        ]
        with self.assertRaisesRegex(ValueError, "distinct behavior group"):
            make_splits(
                insufficient,
                {record["behavior_id"]: labels[record["behavior_id"]] for record in insufficient},
            )
        labels["duplicate"] = 2
        with self.assertRaisesRegex(ValueError, "conflicting severity"):
            make_splits(records, labels)

    def test_shared_context_components_support_multiple_levels_and_transitive_links(self):
        records, labels = split_fixture()
        by_id = {record["behavior_id"]: record for record in records}
        by_id["l1-0"]["context"] = "Shared context A"
        by_id["duplicate"]["context"] = "Shared context B"
        by_id["l2-0"]["context"] = "  SHARED context b\n"
        by_id["l3-0"]["context"] = "Shared context A"
        splits = make_splits(records, labels)
        owners = {item: split for split, ids in splits.items() for item in ids}
        self.assertEqual(len({owners[item] for item in ("l1-0", "duplicate", "l2-0", "l3-0")}), 1)
        self.assertEqual(splits, make_splits(list(reversed(records)), labels))
        origin = owners["l2-0"]
        target = next(name for name in splits if name != origin)
        splits[origin].remove("l2-0")
        splits[target].append("l2-0")
        with self.assertRaisesRegex(ValueError, "context leaks"):
            validate_splits(records, labels, splits)

    def test_context_grouping_reports_impossible_level_coverage(self):
        records, labels = split_fixture()
        for record in records:
            if labels[record["behavior_id"]] == 1:
                record["context"] = "All level-one fixtures share this context"
        with self.assertRaisesRegex(ValueError, "coverage impossible"):
            make_splits(records, labels)

    def test_zero_ratio_splits_are_empty_and_positive_splits_have_coverage(self):
        records, labels = split_fixture()
        splits = make_splits(records, labels, ratios=(0.8, 0.0, 0.2))
        self.assertEqual(splits["dev"], [])
        validate_splits(records, labels, splits, required_splits=("fit", "test"))

    def test_validate_rejects_duplicate_text_leakage_and_repeated_ids(self):
        records, labels = split_fixture()
        splits = make_splits(records, labels)
        origin = next(name for name, ids in splits.items() if "duplicate" in ids)
        target = next(name for name in splits if name != origin)
        splits[origin].remove("duplicate")
        splits[target].append("duplicate")
        with self.assertRaisesRegex(ValueError, "leaks between splits"):
            validate_splits(records, labels, splits)
        splits = make_splits(records, labels)
        splits["test"].append(splits["fit"][0])
        with self.assertRaisesRegex(ValueError, "more than once"):
            validate_splits(records, labels, splits)

    def test_split_ratios_and_label_types_validated(self):
        records, labels = split_fixture()
        for ratios in ((0.5, 0.2, 0.2), (-0.1, 0.5, 0.6), (float("nan"), 0.5, 0.5)):
            with self.subTest(ratios=ratios), self.assertRaises(ValueError):
                make_splits(records, labels, ratios=ratios)
        labels["l1-0"] = True
        with self.assertRaisesRegex(ValueError, "integers"):
            make_splits(records, labels)

    @patch("hb_steering.data.urlopen")
    def test_download_pinned_atomic_and_cached_with_hash_validation(self, open_mock):
        content = csv_bytes([source_row()])
        open_mock.return_value = io.BytesIO(content)
        manifest = download_harmbench(self.root, REVISION)
        self.assertEqual(manifest["sha256"], hashlib.sha256(content).hexdigest())
        self.assertIn("/" + REVISION + "/", manifest["source_url"])
        self.assertEqual(Path(manifest["path"]).read_bytes(), content)
        self.assertEqual(download_harmbench(self.root, REVISION), manifest)
        self.assertEqual(open_mock.call_count, 1)
        Path(manifest["path"]).write_bytes(b"corrupt")
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            download_harmbench(self.root, REVISION)

    @patch("hb_steering.data.urlopen")
    def test_download_rejects_floating_refs_and_invalid_csv(self, open_mock):
        for ref in ("main", "v1.0", "a" * 7, "../data"):
            with self.subTest(ref=ref), self.assertRaisesRegex(ValueError, "commit SHA"):
                download_harmbench(self.root, ref)
        open_mock.assert_not_called()
        open_mock.return_value = io.BytesIO(b"<html>Not CSV</html>")
        with self.assertRaisesRegex(RuntimeError, "CSV is invalid"):
            download_harmbench(self.root, REVISION)
        self.assertFalse((self.root / "data" / "raw" / "manifest.json").exists())
        self.assertFalse((self.root / "data" / "raw" / "harmbench_behaviors_text_all.csv").exists())

    @patch("hb_steering.data.time.sleep")
    @patch("hb_steering.data.urlopen")
    def test_download_retries_transient_errors_but_not_missing_revision(
        self, open_mock, sleep_mock
    ):
        open_mock.side_effect = URLError("fixture offline")
        with self.assertRaisesRegex(RuntimeError, "after retries"):
            download_harmbench(self.root, REVISION)
        self.assertEqual(open_mock.call_count, 3)
        self.assertEqual(sleep_mock.call_count, 2)
        open_mock.reset_mock()
        open_mock.side_effect = HTTPError("https://example.invalid", 404, "missing", {}, None)
        with self.assertRaisesRegex(RuntimeError, "HTTP 404"):
            download_harmbench(self.root, REVISION)
        self.assertEqual(open_mock.call_count, 1)

    @patch("hb_steering.data.urlopen")
    def test_download_rejects_incomplete_and_wrong_revision_cache(self, open_mock):
        content = csv_bytes([source_row()])
        open_mock.return_value = io.BytesIO(content)
        manifest = download_harmbench(self.root, REVISION)
        with self.assertRaisesRegex(RuntimeError, "provenance/hash mismatch"):
            download_harmbench(self.root, "b" * 40)
        (self.root / "data" / "raw" / "manifest.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            download_harmbench(self.root, REVISION)
        self.assertEqual(Path(manifest["path"]).read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
