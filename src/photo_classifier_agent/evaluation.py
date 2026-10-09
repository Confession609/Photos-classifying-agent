"""Dependency-free classification metrics for the first baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .categories import CATEGORY_NAMES
from .schemas import ClassificationDecision


@dataclass(frozen=True)
class EvaluationResult:
    count: int
    accuracy: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]
    confusion_matrix: dict[str, dict[str, int]]
    review_rate: float
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix,
            "review_rate": self.review_rate,
            "metadata": self.metadata,
        }


def evaluate(
    decisions: Iterable[ClassificationDecision],
    truth_by_image_id: dict[str, str],
) -> EvaluationResult:
    usable = [
        decision for decision in decisions
        if decision.image_id in truth_by_image_id
        and truth_by_image_id[decision.image_id] in CATEGORY_NAMES
    ]
    confusion = {actual: {predicted: 0 for predicted in CATEGORY_NAMES} for actual in CATEGORY_NAMES}
    for decision in usable:
        confusion[truth_by_image_id[decision.image_id]][decision.final_category] += 1

    per_class: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for category in CATEGORY_NAMES:
        true_positive = confusion[category][category]
        false_positive = sum(confusion[other][category] for other in CATEGORY_NAMES if other != category)
        false_negative = sum(confusion[category][other] for other in CATEGORY_NAMES if other != category)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[category] = {"precision": precision, "recall": recall, "f1": f1}

    correct = sum(confusion[category][category] for category in CATEGORY_NAMES)
    review_count = sum(decision.review_required for decision in usable)
    return EvaluationResult(
        count=len(usable),
        accuracy=correct / len(usable) if usable else 0.0,
        macro_f1=sum(f1_values) / len(f1_values),
        per_class=per_class,
        confusion_matrix=confusion,
        review_rate=review_count / len(usable) if usable else 0.0,
    )


def read_truth_manifest(path: str | Path) -> dict[str, str]:
    truth: dict[str, str] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                truth[record["image_id"]] = record["category"]
    return truth


def write_evaluation(result: EvaluationResult, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return target

