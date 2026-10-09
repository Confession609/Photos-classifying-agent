import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from photo_classifier_agent.categories import CATEGORY_NAMES
from photo_classifier_agent.classifier import StaticClassifier
from photo_classifier_agent.dataset import scan_dataset, stratified_split, write_split_manifest
from photo_classifier_agent.decision import decide
from photo_classifier_agent.evaluation import evaluate
from photo_classifier_agent.file_ops import apply_decisions
from photo_classifier_agent.hierarchy import HIERARCHICAL_LEAF_PATHS, HIERARCHY_NODES, build_hierarchical_manifest, leaf_probabilities, make_hierarchical_decision, project_manifest_to_nodes, route_category, routed_leaf_scores, target_for
from photo_classifier_agent.hierarchy_cv import _inner_split, _stratified_buckets
from photo_classifier_agent.hierarchy_v2 import V2_LEAF_LABELS, build_v2_manifest, project_v2_node_manifests, read_v2_manifest
from photo_classifier_agent.reports import read_decisions, write_decisions, write_review_html
from photo_classifier_agent.schemas import ContextSignals, VisionReport
from photo_classifier_agent.training import read_training_manifest


class CorePipelineTests(unittest.TestCase):
    def make_dataset(self, root: Path) -> None:
        for category in CATEGORY_NAMES:
            category_dir = root / category
            category_dir.mkdir(parents=True)
            (category_dir / f"{category}.jpg").write_bytes(category.encode("utf-8"))
            (category_dir / "ignored.txt").write_text("ignored", encoding="utf-8")

    def test_scan_and_split_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            self.make_dataset(root)
            samples = scan_dataset(root)
            first = stratified_split(samples, seed=42)
            second = stratified_split(samples, seed=42)
            self.assertEqual(first, second)
            self.assertEqual(len(samples), 4)
            self.assertEqual(sum(map(len, (first.train, first.validation, first.test))), 4)

    def test_manifest_round_trip_and_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            self.make_dataset(root)
            split = stratified_split(scan_dataset(root), seed=1)
            manifest = write_split_manifest(split, Path(temporary) / "artifacts")
            truth = {}
            for line in manifest.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                truth[record["image_id"]] = record["category"]
            decisions = []
            for image_id, category in truth.items():
                source = next(item.path for item in scan_dataset(root) if item.image_id == image_id)
                prediction = StaticClassifier({name: 1.0 if name == category else 0.0 for name in CATEGORY_NAMES}).classify(source)
                decisions.append(decide(prediction))
            output = Path(temporary) / "decisions.jsonl"
            write_decisions(decisions, output)
            self.assertEqual(len(read_decisions(output)), len(decisions))
            metrics = evaluate(decisions, truth)
            self.assertEqual(metrics.accuracy, 1.0)
            self.assertEqual(metrics.macro_f1, 1.0)

    def test_person_priority_and_review_gate(self):
        classifier = StaticClassifier({"人像": 0.45, "风光摄影": 0.5, "星空": 0.04, "静物摄影": 0.01})
        prediction = classifier.classify("person-night.jpg")
        report = VisionReport(
            image_id=prediction.image_id,
            source_path=prediction.source_path,
            context=ContextSignals(people_present=True, people_is_primary=True, night_sky_present=True),
            report_complete=True,
        )
        decision = decide(prediction, report)
        self.assertEqual(decision.final_category, "人像")
        self.assertTrue(decision.review_required)

    def test_apply_copies_without_touching_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.jpg"
            source.write_bytes(b"photo")
            prediction = StaticClassifier({"人像": 1.0, "风光摄影": 0.0, "星空": 0.0, "静物摄影": 0.0}).classify(source)
            decision = decide(prediction)
            results = apply_decisions([decision], Path(temporary) / "output")
            self.assertEqual(results[0].status, "copied")
            self.assertTrue(source.exists())
            self.assertTrue((Path(temporary) / "output" / "人像" / "source.jpg").exists())

    def test_training_manifest_is_authoritative_and_rejects_cross_split_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.jsonl"
            records = []
            for split in ("train", "validation", "test"):
                image = root / f"{split}.jpg"
                image.write_bytes(split.encode("utf-8"))
                from photo_classifier_agent.classifier import image_id_for
                records.append({
                    "image_id": image_id_for(image),
                    "path": str(image),
                    "category": "人像",
                    "content_hash": split,
                    "split": split,
                })
            manifest_path.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                encoding="utf-8",
            )
            loaded, digest = read_training_manifest(manifest_path)
            self.assertEqual(len(loaded), 3)
            self.assertEqual(len(digest), 64)

            records[2]["content_hash"] = records[0]["content_hash"]
            manifest_path.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate content_hash"):
                read_training_manifest(manifest_path)

    def test_html_review_report_keeps_css_braces_and_renders_rows(self):
        classifier = StaticClassifier({"人像": 0.9, "风光摄影": 0.05, "星空": 0.03, "静物摄影": 0.02})
        decision = decide(classifier.classify("sample.jpg"))
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "review.html"
            write_review_html([decision], report)
            content = report.read_text(encoding="utf-8")
            self.assertIn("body { font-family: sans-serif;", content)
            self.assertIn("<td>人像</td>", content)
            self.assertNotIn("__PHOTO_CLASSIFIER_ROWS__", content)

    def test_hierarchy_routes_and_leaf_probabilities(self):
        self.assertEqual(route_category(0.8, 0.99, 0.99), "人像")
        self.assertEqual(route_category(0.2, 0.8, 0.9), "星空")
        self.assertEqual(route_category(0.2, 0.2, 0.8), "静物摄影")
        self.assertEqual(route_category(0.2, 0.2, 0.1), "风光摄影")
        probabilities = leaf_probabilities(0.2, 0.4, 0.7)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        routed = route_category(0.49, 0.51, 0.9)
        scores = routed_leaf_scores(routed, 0.49, 0.51, 0.9)
        self.assertEqual(max(scores, key=scores.get), routed)
        self.assertAlmostEqual(sum(scores.values()), 1.0)
        decision = make_hierarchical_decision("x", "x.jpg", 0.82, 0.99, 0.99, classifier_version="cascade")
        self.assertEqual(decision.final_category, "人像")
        self.assertFalse(decision.review_required)

    def test_hierarchy_manifests_filter_samples_and_preserve_split(self):
        records = [
            {"image_id": category, "path": f"{category}.jpg", "category": category,
             "content_hash": category, "split": "train"}
            for category in CATEGORY_NAMES
        ]
        with tempfile.TemporaryDirectory() as temporary:
            summaries = project_manifest_to_nodes(records, temporary)
            self.assertEqual(summaries["person_gate"]["samples"], 4)
            self.assertEqual(summaries["sky_gate"]["samples"], 3)
            self.assertEqual(summaries["still_vs_landscape"]["samples"], 2)
            person_rows = [json.loads(line) for line in (Path(temporary) / "person_gate.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual({row["split"] for row in person_rows}, {"train"})
            self.assertEqual(target_for(HIERARCHY_NODES[0], "静物摄影"), "非人像")
            self.assertIsNone(target_for(HIERARCHY_NODES[2], "星空"))

    def test_nested_hierarchy_scan_preserves_old_split_by_content_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            output = Path(temporary) / "output"
            samples = []
            known_payload = b"existing landscape sample"
            for relative_parts, category in HIERARCHICAL_LEAF_PATHS:
                folder = root.joinpath(*relative_parts)
                folder.mkdir(parents=True)
                payload = known_payload if category == "风光摄影" else f"new-{category}".encode("utf-8")
                photo = folder / "sample.jpg"
                photo.write_bytes(payload)
                samples.append((category, photo, hashlib.sha256(payload).hexdigest()))
            previous = Path(temporary) / "previous.jsonl"
            old = next(row for row in samples if row[0] == "风光摄影")
            previous.write_text(json.dumps({"content_hash": old[2], "split": "test"}) + "\n", encoding="utf-8")

            records, summary = build_hierarchical_manifest(root, output, previous_manifest=previous)
            self.assertEqual(summary["preserved_prior_splits"], 1)
            self.assertEqual(summary["new_samples"], 3)
            self.assertEqual(next(row["split"] for row in records if row["category"] == "风光摄影"), "test")
            self.assertEqual({row["split"] for row in records if row["category"] != "风光摄影"}, {"train"})

    def test_stratified_kfold_assigns_each_sample_to_one_fold(self):
        records = [
            {"image_id": f"{category}-{index}", "category": category}
            for category in CATEGORY_NAMES
            for index in range(11)
        ]
        folds = _stratified_buckets(records, 5, 42)
        flattened = [row["image_id"] for fold in folds for row in fold]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(flattened), {row["image_id"] for row in records})
        for fold in folds:
            self.assertTrue(all(sum(row["category"] == category for row in fold) >= 2 for category in CATEGORY_NAMES))

    def test_inner_stratification_supports_conditional_node_subsets(self):
        records = [
            {"image_id": f"{category}-{index}", "category": category}
            for category in ("风光摄影", "静物摄影")
            for index in range(20)
        ]
        train, validation = _inner_split(records, 7)
        self.assertEqual(len(train) + len(validation), len(records))
        self.assertEqual({row["category"] for row in validation}, {"风光摄影", "静物摄影"})

    def test_v2_manifest_has_one_authoritative_split_and_two_nodes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            (root / "人像").mkdir(parents=True)
            for category in ("风光摄影", "静物摄影", "活动事件摄影"):
                (root / "非人像" / category).mkdir(parents=True)
            for category_dir in (
                root / "人像",
                root / "非人像" / "风光摄影",
                root / "非人像" / "静物摄影",
                root / "非人像" / "活动事件摄影",
            ):
                for index in range(4):
                    (category_dir / f"{index}.jpg").write_bytes(f"{category_dir.name}-{index}".encode())
            output = Path(temporary) / "artifacts"
            summary = build_v2_manifest(root, output, seed=11)
            records, digest = read_v2_manifest(summary["manifest"])
            nodes = project_v2_node_manifests(records, output / "manifests")
            self.assertEqual(set(row["category"] for row in records), set(V2_LEAF_LABELS))
            self.assertEqual(len({row["content_hash"] for row in records}), len(records))
            self.assertEqual(nodes["person_gate"]["samples"], 16)
            self.assertEqual(nodes["non_person_classifier"]["samples"], 12)
            self.assertEqual(len(digest), 64)


if __name__ == "__main__":
    unittest.main()
