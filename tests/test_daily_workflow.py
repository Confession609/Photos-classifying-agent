import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from photo_classifier_agent.daily_workflow import (
    check_layout, copy_photo, digest, validate_score, run_folder, safe_directory, scan_images,
)
from photo_classifier_agent.hierarchy_v2 import V2_LEAF_LABELS


class DailyWorkflowTests(unittest.TestCase):
    def test_sibling_output_allowed_but_input_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "待分类文件夹"
            source.mkdir()
            check_layout(source, root)
            for output in (source, source / "output", root.parent):
                with self.assertRaises(ValueError):
                    check_layout(source, output)
            bad_input = root / "人像"
            bad_input.mkdir()
            with self.assertRaises(ValueError):
                check_layout(bad_input, root)

    def test_unlabeled_scan_does_not_read_sidecars_or_sibling_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input"
            source.mkdir()
            (source / "a.JPG").write_bytes(b"a")
            (source / "labels.json").write_text("INVALID", encoding="utf-8")
            child = source / "nested"
            child.mkdir()
            (child / "b.png").write_bytes(b"b")
            (root / "already_classified.jpg").write_bytes(b"output")
            recursive, _ = scan_images(source)
            top, _ = scan_images(source, False)
            self.assertEqual(len(recursive), 2)
            self.assertEqual([p.name for p in top], ["a.JPG"])

    def test_copy_collision_and_repeat_are_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "first", root / "second"
            first.mkdir(); second.mkdir()
            a, b = first / "same.jpg", second / "same.jpg"
            a.write_bytes(b"first"); b.write_bytes(b"second")
            output = root / "output"
            x = copy_photo(a, output / "活动事件摄影", digest(a), output)
            y = copy_photo(b, output / "活动事件摄影", digest(b), output)
            repeated = copy_photo(b, output / "活动事件摄影", digest(b), output)
            self.assertEqual(x["copy_status"], "copied")
            self.assertEqual(y["copy_status"], "copied_collision_renamed")
            self.assertEqual(repeated["copy_status"], "skipped_identical")
            self.assertEqual(y["destination_path"], repeated["destination_path"])
            self.assertEqual(Path(x["destination_path"]).read_bytes(), b"first")
            self.assertEqual(a.read_bytes(), b"first")
            self.assertEqual(b.read_bytes(), b"second")

    def test_changed_source_not_copied(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "a.jpg"
            source.write_bytes(b"original")
            sha = digest(source)
            source.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                copy_photo(source, root / "output/人像", sha, root / "output")
            self.assertFalse((root / "output/人像/a.jpg").exists())

    def test_redirected_category_directory_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("photo_classifier_agent.daily_workflow.linked", side_effect=lambda p: p.name == "人像"):
                with self.assertRaises(ValueError):
                    safe_directory(root / "人像", root)

    def test_low_scores_are_valid_but_nonfinite_invalid_scores_rejected(self):
        self.assertEqual(validate_score(.01), .01)
        self.assertEqual(validate_score(.75), .75)
        for score in (math.nan, math.inf, -.1, 1.1):
            with self.assertRaises(ValueError):
                validate_score(score)

    def test_pipeline_archives_low_scores_and_isolates_failure_without_touching_input(self):
        class FakeModels:
            def __init__(self, *args):
                pass

            def predict(self, source, run_dir, identity):
                if source.name == "broken.jpg":
                    raise ValueError("decode failed")
                return {"predicted_category": "活动事件摄影", "confidence_uncalibrated": .4 if source.name == "uncertain.jpg" else .9}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input"
            source.mkdir()
            for filename in ("clear.jpg", "uncertain.jpg", "broken.jpg"):
                (source / filename).write_bytes(filename.encode())
            before = {p.name: digest(p) for p in source.iterdir()}
            with patch("photo_classifier_agent.daily_workflow.load_assets", return_value=({}, {}, {})):
                result = run_folder(source, root, root / "unused.json", model_factory=FakeModels)
            self.assertEqual(result["statuses"], {"failed": 1, "classified": 2})
            self.assertTrue((root / "活动事件摄影/clear.jpg").exists())
            self.assertTrue((root / "活动事件摄影/uncertain.jpg").exists())
            self.assertFalse((root / "_待复核").exists())
            self.assertFalse(result["confidence_threshold_enabled"])
            self.assertFalse((root / "活动事件摄影/broken.jpg").exists())
            self.assertEqual(before, {p.name: digest(p) for p in source.iterdir()})
            run_dir = Path(result["run_directory"])
            saved = [json.loads(line) for line in (run_dir / "classification_results.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(saved), 3)
            self.assertTrue((run_dir / "report.html").exists())
            self.assertIn("活动事件摄影", (run_dir / "分类结果.txt").read_text(encoding="utf-8"))

    def test_all_four_predicted_categories_archive_even_at_low_score(self):
        class FakeModels:
            def __init__(self, *args):
                pass

            def predict(self, source, run_dir, identity):
                return {"predicted_category": source.stem, "confidence_uncalibrated": .01}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input"
            source.mkdir()
            for category in V2_LEAF_LABELS:
                (source / f"{category}.jpg").write_bytes(category.encode("utf-8"))
            with patch("photo_classifier_agent.daily_workflow.load_assets", return_value=({}, {}, {})):
                summary = run_folder(source, root, root / "unused.json", model_factory=FakeModels)
            self.assertEqual(summary["statuses"], {"classified": 4})
            for category in V2_LEAF_LABELS:
                self.assertTrue((root / category / f"{category}.jpg").exists())
            self.assertFalse((root / "_待复核").exists())
            self.assertFalse(summary["confidence_threshold_enabled"])


if __name__ == "__main__":
    unittest.main()
