"""Training and inference for the v2 person/non-person cascade."""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .classifier import image_id_for
from .evaluation import write_evaluation
from .hierarchy_v2 import V2_LEAF_LABELS, V2_NON_PERSON_LABELS, V2_PERSON_LABELS, read_v2_manifest
from .reports import write_decisions, write_predictions
from .schemas import CandidateScore, ClassificationDecision, Prediction
from .subject import load_subject_reports


def _cache_env() -> None:
    root = Path(__file__).resolve().parents[2] / "algorithms" / "models" / "huggingface"
    os.environ.setdefault("HF_HOME", str(root))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(root / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(root / "transformers"))
    os.environ.setdefault("HF_MODULES_CACHE", str(root / "modules"))


def _report_features(report: dict[str, Any] | None) -> list[float]:
    if not report:
        return [0.0, 0.0, 0.0, 0.0]
    regions = report.get("subject_regions", []) or []
    person_regions = [row for row in regions if "person" in str(row.get("label", "")).lower()]
    max_person_score = max((float(row.get("confidence") or 0.0) for row in person_regions), default=0.0)
    max_person_area = max((float(row.get("width", 0.0)) * float(row.get("height", 0.0)) for row in person_regions), default=0.0)
    context = report.get("context", {}) or {}
    people_present = 1.0 if context.get("people_present") else 0.0
    people_primary = 1.0 if context.get("people_is_primary") else 0.0
    return [max_person_score, max_person_area, people_present, people_primary]


def _open_clip(model_name: str, pretrained: str, device: str | None):
    _cache_env()
    try:
        import open_clip
        import torch
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("v2 training requires the vision dependencies") from exc
    chosen = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=chosen)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return torch, Image, model, preprocess, chosen


def _load_image(path: str, image_cls: Any, preprocess: Any):
    with image_cls.open(path) as image:
        return preprocess(image.convert("RGB"))


def _extract_features(
    records: list[dict[str, Any]],
    reports: dict[str, dict[str, Any]],
    model: Any,
    preprocess: Any,
    image_cls: Any,
    torch: Any,
    device: str,
    batch_size: int,
) -> Any:
    vectors = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        original = torch.stack([_load_image(row["path"], image_cls, preprocess) for row in batch]).to(device)
        crops = []
        scalars = []
        for row in batch:
            report = reports.get(row["image_id"])
            crop_path = report.get("subject_crop_path") if report else None
            if crop_path and Path(crop_path).is_file():
                crops.append(_load_image(crop_path, image_cls, preprocess))
            else:
                crops.append(_load_image(row["path"], image_cls, preprocess))
            scalars.append(_report_features(report))
        crop_tensor = torch.stack(crops).to(device)
        with torch.inference_mode():
            original_features = model.encode_image(original)
            crop_features = model.encode_image(crop_tensor)
            original_features = original_features / original_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            crop_features = crop_features / crop_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        scalar_tensor = torch.tensor(scalars, dtype=torch.float32, device=device)
        vectors.append(torch.cat((original_features, crop_features, scalar_tensor), dim=1).float().cpu())
        print(f"v2 features {min(start + len(batch), len(records))}/{len(records)}", flush=True)
    return torch.cat(vectors) if vectors else torch.empty((0, 0), dtype=torch.float32)


def _metrics(logits: Any, labels: Any, class_names: tuple[str, ...]) -> dict[str, Any]:
    import torch

    predicted = logits.argmax(dim=1).cpu().tolist()
    actual = labels.cpu().tolist()
    confusion = [[0 for _ in class_names] for _ in class_names]
    for truth, guess in zip(actual, predicted):
        confusion[truth][guess] += 1
    recalls = []
    f1s = []
    for index in range(len(class_names)):
        tp = confusion[index][index]
        fn = sum(confusion[index]) - tp
        fp = sum(row[index] for row in confusion) - tp
        recalls.append(tp / (tp + fn) if tp + fn else 0.0)
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1s.append(2 * precision * recalls[-1] / (precision + recalls[-1]) if precision + recalls[-1] else 0.0)
    return {
        "count": len(actual),
        "accuracy": sum(a == p for a, p in zip(actual, predicted)) / len(actual) if actual else 0.0,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "recall": dict(zip(class_names, recalls)),
        "confusion_matrix": confusion,
        "class_names": list(class_names),
    }


def _fit_head(
    torch: Any,
    train_x: Any,
    train_y: Any,
    validation_x: Any,
    validation_y: Any,
    class_names: tuple[str, ...],
    *,
    epochs: int,
    patience: int,
    learning_rate: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch.manual_seed(seed)
    head = torch.nn.Linear(train_x.shape[1], len(class_names))
    counts = Counter(train_y.tolist())
    weights = torch.tensor(
        [len(train_y) / (len(class_names) * max(1, counts[index])) for index in range(len(class_names))],
        dtype=torch.float32,
    )
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=0.01)
    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        head.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(train_x), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(head(train_x[order]), train_y[order])
        loss.backward()
        optimizer.step()
        head.eval()
        with torch.inference_mode():
            validation_logits = head(validation_x)
        metric = _metrics(validation_logits, validation_y, class_names)
        row = {"epoch": epoch, "loss": float(loss.item()), **metric}
        history.append(row)
        if metric["macro_f1"] > best_f1:
            best_f1 = metric["macro_f1"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("v2 classifier head did not produce a validation checkpoint")
    return {
        "state_dict": best_state,
        "class_names": class_names,
        "best_epoch": best_epoch,
        "validation_macro_f1": best_f1,
        "train_counts": dict(Counter(int(value) for value in train_y.tolist())),
    }, {"history": history}


def _node_rows(records: list[dict[str, Any]], node: str) -> list[dict[str, Any]]:
    if node == "person_gate":
        return [row | {"target_label": "人像" if row["category"] == "人像" else "非人像"} for row in records]
    return [row | {"target_label": row["category"]} for row in records if row["category"] != "人像"]


def _decision_from_probs(record: dict[str, Any], person_probs: list[float], non_person_probs: list[float], version: str):
    person = float(person_probs[0])
    if person >= 0.5:
        category = "人像"
        confidence = person
        evidence = (f"人像门控选择人像（{person:.3f}）",)
        selected = person
    else:
        index = max(range(len(non_person_probs)), key=non_person_probs.__getitem__)
        category = V2_NON_PERSON_LABELS[index]
        selected = float(non_person_probs[index])
        confidence = (1.0 - person) * selected
        evidence = (f"人像门控选择非人像（{1.0 - person:.3f}）", f"非人像分类选择{category}（{selected:.3f}）")
    reasons = []
    if selected < 0.75:
        reasons.append("级联路径置信度低于0.75")
    if person >= 0.4 and person < 0.6:
        reasons.append("人像门控接近决策边界")
    if not person >= 0.5 and selected < 0.5:
        reasons.append("非人像三分类置信度低")
    return ClassificationDecision(
        image_id=record["image_id"], source_path=record["path"], final_category=category,
        confidence=confidence, evidence=evidence, review_required=bool(reasons),
        review_reason="；".join(reasons) if reasons else None, classifier_version=version,
    )


def train_v2(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    reports_path: str | Path | None = None,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: str | None = None,
    batch_size: int = 16,
    epochs: int = 60,
    patience: int = 10,
    learning_rate: float = 0.01,
    seed: int = 1337,
) -> dict[str, Any]:
    records, manifest_sha = read_v2_manifest(manifest_path)
    reports = load_subject_reports(reports_path) if reports_path else {}
    if reports_path:
        missing_reports = [row["image_id"] for row in records if row["image_id"] not in reports]
        if missing_reports:
            raise ValueError(
                f"subject report coverage is incomplete: {len(missing_reports)} of {len(records)} manifest images are missing; "
                "finish generate-subject-reports before training"
            )
    torch, image_cls, model, preprocess, chosen_device = _open_clip(model_name, pretrained, device)
    random.seed(seed)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    by_split = {split: [row for row in records if row["split"] == split] for split in ("train", "validation", "test")}
    features = {split: _extract_features(rows, reports, model, preprocess, image_cls, torch, chosen_device, batch_size) for split, rows in by_split.items()}
    heads = {}
    histories = {}
    for node, names in (("person_gate", V2_PERSON_LABELS), ("non_person_classifier", V2_NON_PERSON_LABELS)):
        train_rows = _node_rows(by_split["train"], node)
        validation_rows = _node_rows(by_split["validation"], node)
        positions_train = {row["image_id"]: index for index, row in enumerate(by_split["train"])}
        positions_validation = {row["image_id"]: index for index, row in enumerate(by_split["validation"])}
        train_indices = torch.tensor([positions_train[row["image_id"]] for row in train_rows], dtype=torch.long)
        validation_indices = torch.tensor([positions_validation[row["image_id"]] for row in validation_rows], dtype=torch.long)
        train_x = features["train"].index_select(0, train_indices)
        validation_x = features["validation"].index_select(0, validation_indices)
        train_y = torch.tensor([names.index(row["target_label"]) for row in train_rows], dtype=torch.long)
        validation_y = torch.tensor([names.index(row["target_label"]) for row in validation_rows], dtype=torch.long)
        if set(train_y.tolist()) != set(range(len(names))) or set(validation_y.tolist()) != set(range(len(names))):
            raise ValueError(f"v2 node {node} needs every class in train and validation")
        head, history = _fit_head(
            torch, train_x, train_y, validation_x, validation_y, names,
            epochs=epochs, patience=patience, learning_rate=learning_rate,
            seed=seed + (0 if node == "person_gate" else 100),
        )
        heads[node] = head
        histories[node] = history

    checkpoint_path = output / "hierarchical_v2_openclip_probe.pt"
    checkpoint = {
        "heads": heads,
        "model_name": model_name,
        "pretrained": pretrained,
        "feature_dim": int(features["train"].shape[1]),
        "manifest_sha256": manifest_sha,
        "reports_path": str(Path(reports_path).resolve()) if reports_path else None,
        "variant": "subject_crop_plus_original_v2",
        "leaf_labels": V2_LEAF_LABELS,
    }
    torch.save(checkpoint, checkpoint_path)
    (output / "training_history.json").write_text(json.dumps(histories, ensure_ascii=False, indent=2), encoding="utf-8")

    test_predictions = []
    test_decisions = []
    person_head = torch.nn.Linear(features["test"].shape[1], 2)
    person_head.load_state_dict(heads["person_gate"]["state_dict"])
    non_person_head = torch.nn.Linear(features["test"].shape[1], 3)
    non_person_head.load_state_dict(heads["non_person_classifier"]["state_dict"])
    with torch.inference_mode():
        all_person = torch.softmax(person_head(features["test"]), dim=1)
        all_non_person = torch.softmax(non_person_head(features["test"]), dim=1)
    for index, record in enumerate(by_split["test"]):
        person_probs = all_person[index].tolist()
        non_person_probs = all_non_person[index].tolist()
        decision = _decision_from_probs(record, person_probs, non_person_probs, "hierarchical_v2_openclip")
        scores = {label: 0.0 for label in V2_LEAF_LABELS}
        scores["人像"] = person_probs[0]
        for label, score in zip(V2_NON_PERSON_LABELS, non_person_probs, strict=True):
            scores[label] = (1.0 - person_probs[0]) * score
        test_predictions.append(Prediction(
            image_id=record["image_id"], source_path=record["path"],
            scores=tuple(CandidateScore(label, float(scores[label])) for label in V2_LEAF_LABELS),
            model_name="open_clip_hierarchical_v2", model_version="subject_crop_plus_original_v2",
        ))
        test_decisions.append(decision)
    predictions_path = write_predictions(test_predictions, output / "test_predictions.jsonl")
    decisions_path = write_decisions(test_decisions, output / "test_decisions.jsonl")
    truth = {row["image_id"]: row["category"] for row in by_split["test"]}
    metrics = {
        "cascade": {
            "accuracy": sum(decision.final_category == truth[decision.image_id] for decision in test_decisions) / len(test_decisions),
            "count": len(test_decisions),
            "review_rate": sum(decision.review_required for decision in test_decisions) / len(test_decisions),
        },
        "person_gate_validation": heads["person_gate"]["validation_macro_f1"],
        "non_person_validation": heads["non_person_classifier"]["validation_macro_f1"],
        "manifest_sha256": manifest_sha,
        "split_counts": {split: len(rows) for split, rows in by_split.items()},
        "checkpoint": str(checkpoint_path),
        "predictions": str(predictions_path),
        "decisions": str(decisions_path),
    }
    (output / "test_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


class V2CascadeClassifier:
    """Load a v2 checkpoint and classify using cached subject reports when available."""

    def __init__(self, checkpoint_path: str | Path, reports_path: str | Path | None = None, device: str | None = None):
        torch, image_cls, model, preprocess, chosen_device = _open_clip("ViT-B-32", "openai", device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if tuple(checkpoint["leaf_labels"]) != V2_LEAF_LABELS:
            raise ValueError("checkpoint does not match v2 labels")
        if checkpoint["model_name"] != "ViT-B-32" or checkpoint["pretrained"] != "openai":
            raise ValueError("v2 inference currently expects the project's cached ViT-B-32/openai backbone")
        self.torch, self.image_cls, self.model, self.preprocess, self.device = torch, image_cls, model, preprocess, chosen_device
        self.reports = load_subject_reports(reports_path) if reports_path else {}
        self.person_head = torch.nn.Linear(checkpoint["feature_dim"], 2)
        self.person_head.load_state_dict(checkpoint["heads"]["person_gate"]["state_dict"])
        self.non_person_head = torch.nn.Linear(checkpoint["feature_dim"], 3)
        self.non_person_head.load_state_dict(checkpoint["heads"]["non_person_classifier"]["state_dict"])
        self.person_head.eval()
        self.non_person_head.eval()
        self.model_version = "hierarchical_v2_openclip:subject_crop_plus_original_v2"

    def _features(self, path: Path):
        report = self.reports.get(image_id_for(path))
        original = _load_image(str(path), self.image_cls, self.preprocess).unsqueeze(0).to(self.device)
        crop_path = report.get("subject_crop_path") if report else None
        crop = _load_image(crop_path, self.image_cls, self.preprocess).unsqueeze(0).to(self.device) if crop_path and Path(crop_path).is_file() else original
        with self.torch.inference_mode():
            first = self.model.encode_image(original)
            second = self.model.encode_image(crop)
            first = first / first.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            second = second / second.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        scalar = self.torch.tensor([_report_features(report)], dtype=self.torch.float32, device=self.device)
        return self.torch.cat((first, second, scalar), dim=1).float()

    def classify_with_decision(self, image_path: str | Path) -> tuple[Prediction, ClassificationDecision]:
        path = Path(image_path).expanduser().resolve()
        features = self._features(path)
        with self.torch.inference_mode():
            person_probs = self.torch.softmax(self.person_head(features), dim=1)[0].tolist()
            non_person_probs = self.torch.softmax(self.non_person_head(features), dim=1)[0].tolist()
        record = {"image_id": image_id_for(path), "path": str(path)}
        decision = _decision_from_probs(record, person_probs, non_person_probs, self.model_version)
        scores = {label: 0.0 for label in V2_LEAF_LABELS}
        scores["人像"] = person_probs[0]
        for label, score in zip(V2_NON_PERSON_LABELS, non_person_probs, strict=True):
            scores[label] = (1.0 - person_probs[0]) * score
        prediction = Prediction(
            image_id=record["image_id"], source_path=str(path),
            scores=tuple(CandidateScore(label, float(scores[label])) for label in V2_LEAF_LABELS),
            model_name="open_clip_hierarchical_v2", model_version=self.model_version,
        )
        return prediction, decision

    def classify(self, image_path: str | Path) -> Prediction:
        return self.classify_with_decision(image_path)[0]
