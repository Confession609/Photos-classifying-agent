import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import train_v2_ablation as experiment


class AblationAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []
        for category, folder in experiment.FOLDERS.items():
            directory = self.root / "data/dataset/label layer data" / folder
            directory.mkdir(parents=True)
            annotations = []
            for split in ("train", "validation", "test"):
                image = directory / f"{split}.jpg"
                image.write_bytes(f"{folder}-{split}".encode())
                identity = f"{folder}-{split}"
                row = {"image_id": identity, "category": category, "split": split,
                       "source_path": str(image), "crop_path": str(image),
                       "sha256": experiment.digest(image), "group_id": identity,
                       "primary_subject_box_xywh_norm": [0, 0, 1, 1]}
                self.rows.append(row)
                annotations.append(row | {"status": "success", "subject_crop_path": str(image)})
            (directory / "subject_annotations.jsonl").write_text("\n".join(json.dumps(r) for r in annotations), encoding="utf8")
        self.manifest = self.root / "manifest.jsonl"

    def run_audit(self):
        self.manifest.write_text("\n".join(json.dumps(r) for r in self.rows), encoding="utf8")
        with patch.object(experiment, "ROOT", self.root):
            return experiment.audit(self.manifest)

    def test_preserves_splits_and_excludes_test_from_feature_hashes(self):
        rows, hashes, summary = self.run_audit()
        self.assertEqual(rows, self.rows)
        self.assertEqual(summary["total"], 12)
        self.assertEqual(set(hashes), {r["image_id"] for r in rows if r["split"] != "test"})

    def test_rejects_group_leakage(self):
        self.rows[1]["group_id"] = self.rows[0]["group_id"]
        with self.assertRaisesRegex(AssertionError, "Group leakage"):
            self.run_audit()

    def test_rejects_stale_original(self):
        Path(self.rows[0]["source_path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(AssertionError, "Original changed"):
            self.run_audit()

    def test_rejects_stale_box(self):
        self.rows[0]["primary_subject_box_xywh_norm"] = [0, 0, 0.5, 0.5]
        with self.assertRaises(AssertionError):
            self.run_audit()

    def test_rejects_missing_manifest_record(self):
        self.rows.pop()
        with self.assertRaisesRegex(AssertionError, "exactly"):
            self.run_audit()
