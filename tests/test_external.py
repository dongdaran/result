"""Independent transfer data selection, leakage screening, and snapshot integrity."""

import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hb_steering.external import prepare_external, read_external


def _csv(rows, columns=None):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns or rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def _upstream(index, goal, source="Original"):
    return {"Index": str(index), "Goal": goal, "Target": "Fixture response",
            "Behavior": f"Fixture {index}", "Category": "fixture", "Source": source}


class ExternalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        # Exact duplicate candidate, close-wording candidate, and reviewed
        # semantic overlap are three different exclusions, distinct from source.
        self.harmful_bytes = _csv([
            _upstream(0, " FIRST\nREFERENCE "),
            _upstream(1, "Neutral fixture amber"),
            _upstream(2, "Neutral fixture birch"),
            _upstream(3, "Neutral fixture cobalt"),
            _upstream(4, "First reference rephrased", "TDC/HarmBench"),
            _upstream(5, "Another inherited instruction", "AdvBench"),
            _upstream(6, "Distinct fixture wording with reviewed similarity"),
            _upstream(7, "The long distinctive harmless screening fixture sentence?"),
        ])
        self.benign_bytes = _csv([
            _upstream(0, "Neutral safe zinnia"),
            _upstream(1, "ＳＡＦＥ　REFERENCE"),
            _upstream(2, "first reference"),
        ])
        self.labels = [
            {"behavior_id": f"jbb_harmful:{index}", "level": level,
             "rationale": "Harmless fixture annotation", "exclude": "false",
             "overlap_with": "", "overlap_rationale": ""}
            for index, level in ((0, 1), (1, 1), (2, 2), (3, 3), (6, 2), (7, 3))
        ]
        self.labels[-2].update(exclude="true", overlap_with="hb:0",
                               overlap_rationale="AI fixture review identifies same intention")
        self.labels_path = self.root / "data/annotations/external_severity.csv"
        self.labels_path.parent.mkdir(parents=True)
        self.labels_path.write_bytes(_csv(self.labels))
        self.config = {"external": {
            "enabled": True, "source": "jailbreakbench_original", "revision": "a" * 40,
            "harmful_sha256": hashlib.sha256(self.harmful_bytes).hexdigest(),
            "benign_sha256": hashlib.sha256(self.benign_bytes).hexdigest(),
            "labels_path": "data/annotations/external_severity.csv",
        }}
        self.hb = [{"behavior_id": "hb:0", "behavior": "First reference", "context": ""},
                   {"behavior_id": "hb:1", "behavior":
                    "The long distinctive harmless screening fixture sentence.", "context": ""}]
        self.benign_fit = [{"benign_id": "0", "prompt": "safe reference"}]

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self):
        with patch("hb_steering.external.urllib.request.urlopen",
                   side_effect=[io.BytesIO(self.harmful_bytes), io.BytesIO(self.benign_bytes)]):
            return prepare_external(self.root, self.config, self.hb, self.benign_fit)

    def test_selection_preserves_levels_and_independent_benign_with_exclusion_audit(self):
        summary = self.prepare()
        harmful, benign = read_external(self.root, self.config, self.hb, self.benign_fit)
        self.assertEqual(summary["harmful_total"], 3)
        self.assertEqual(summary["benign_total"], 1)
        self.assertEqual(summary["level_counts"], {1: 1, 2: 1, 3: 1})
        self.assertEqual(summary["group_count"], 3)
        self.assertEqual({row["behavior_id"] for row in harmful},
                         {"jbb_harmful:1", "jbb_harmful:2", "jbb_harmful:3"})
        self.assertTrue(all(row["dataset"] == "jailbreakbench_original" for row in harmful))
        self.assertEqual(benign[0]["level"], 0)
        self.assertEqual(benign[0]["dataset"], "jailbreakbench_benign")
        reasons = {row["behavior_id"]: row["reason"] for row in summary["exclusions"]}
        self.assertEqual(reasons["jbb_harmful:0"], "exact_normalized_overlap")
        self.assertEqual(reasons["jbb_harmful:4"], "inherited_source")
        self.assertEqual(reasons["jbb_harmful:5"], "inherited_source")
        self.assertEqual(reasons["jbb_harmful:6"], "reviewed_semantic_overlap")
        self.assertEqual(reasons["jbb_harmful:7"], "near_text_overlap")
        self.assertEqual(reasons["jbb_benign:1"], "exact_normalized_overlap")
        self.assertEqual(reasons["jbb_benign:2"], "exact_normalized_overlap")
        with patch("hb_steering.external.urllib.request.urlopen") as download:
            self.assertEqual(summary,
                             prepare_external(self.root, self.config, self.hb, self.benign_fit))
            download.assert_not_called()

    def test_raw_cache_corruption_and_repinning_are_rejected(self):
        self.prepare()
        raw = self.root / "data/raw/external/harmful-behaviors.csv"
        raw.write_bytes(raw.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            read_external(self.root, self.config, self.hb, self.benign_fit)
        raw.write_bytes(self.harmful_bytes)
        self.config["external"]["revision"] = "b" * 40
        with self.assertRaisesRegex(ValueError, "changed"):
            read_external(self.root, self.config, self.hb, self.benign_fit)

    def test_labels_and_fit_references_are_frozen(self):
        self.prepare()
        original = self.labels_path.read_bytes()
        self.labels_path.write_bytes(original.replace(b"Harmless fixture", b"New fixture"))
        with self.assertRaisesRegex(ValueError, "changed"):
            read_external(self.root, self.config, self.hb, self.benign_fit)
        self.labels_path.write_bytes(original)
        changed_fit = self.benign_fit + [{"benign_id": "1", "prompt": "different safe reference"}]
        with self.assertRaisesRegex(ValueError, "changed"):
            read_external(self.root, self.config, self.hb, changed_fit)

    def test_explicit_resplit_replaces_selection_snapshot_after_review_changes(self):
        before = self.prepare()
        self.labels_path.write_bytes(self.labels_path.read_bytes().replace(
            b"Harmless fixture annotation", b"Reviewed fixture annotation"))
        with self.assertRaisesRegex(ValueError, "changed"):
            prepare_external(self.root, self.config, self.hb, self.benign_fit)
        after = prepare_external(self.root, self.config, self.hb, self.benign_fit, resplit=True)
        self.assertNotEqual(before["fingerprint"], after["fingerprint"])
        harmful, benign = read_external(self.root, self.config, self.hb, self.benign_fit)
        self.assertEqual((len(harmful), len(benign)), (3, 1))

    def test_processed_content_and_manifest_tampering_are_rejected(self):
        self.prepare()
        directory = self.root / "data/processed/external"
        artifact = directory / "harmful.jsonl"
        original = artifact.read_bytes()
        artifact.write_bytes(original + b"{}\n")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            read_external(self.root, self.config, self.hb, self.benign_fit)
        artifact.write_bytes(original)
        path = directory / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["harmful_total"] = 99
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "changed"):
            read_external(self.root, self.config, self.hb, self.benign_fit)

    def test_incomplete_labels_and_lost_severity_coverage_are_rejected(self):
        self.labels_path.write_bytes(_csv(self.labels[:-1]))
        with self.assertRaisesRegex(ValueError, "Missing severity"):
            self.prepare()
        self.labels_path.write_bytes(_csv(self.labels))
        new_hb = self.hb + [{"behavior_id": "hb:2", "behavior": "Neutral fixture cobalt"}]
        with self.assertRaisesRegex(ValueError, "all three"):
            prepare_external(self.root, self.config, new_hb, self.benign_fit)

    def test_unknown_source_or_unpinned_revision_is_rejected(self):
        self.config["external"]["source"] = "made_up"
        with self.assertRaisesRegex(ValueError, "source"):
            self.prepare()
        self.config["external"]["source"] = "jailbreakbench_original"
        self.config["external"]["revision"] = "main"
        with self.assertRaisesRegex(ValueError, "commit SHA"):
            self.prepare()

    def test_disabled_external_is_backward_compatible_without_downloads(self):
        self.assertEqual(prepare_external(self.root, {}, self.hb, self.benign_fit),
                         {"enabled": False})
        self.assertEqual(read_external(self.root, {}, self.hb, self.benign_fit), ([], []))


if __name__ == "__main__":
    unittest.main()
