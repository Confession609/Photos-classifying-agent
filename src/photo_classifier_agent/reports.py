"""JSONL, CSV, and HTML report persistence."""

from __future__ import annotations

import csv
import html
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .schemas import ClassificationDecision, Prediction, VisionReport


def write_predictions(predictions: Iterable[Prediction], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(asdict(prediction), ensure_ascii=False) + "\n")
    return target


def read_predictions(path: str | Path) -> list[Prediction]:
    results: list[Prediction] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            from .schemas import CandidateScore
            results.append(
                Prediction(
                    image_id=record["image_id"],
                    source_path=record["source_path"],
                    scores=tuple(CandidateScore(**item) for item in record["scores"]),
                    model_name=record["model_name"],
                    model_version=record["model_version"],
                )
            )
    return results


def write_decisions(decisions: Iterable[ClassificationDecision], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for decision in decisions:
            handle.write(json.dumps(asdict(decision), ensure_ascii=False) + "\n")
    return target


def read_decisions(path: str | Path) -> list[ClassificationDecision]:
    decisions: list[ClassificationDecision] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                decisions.append(ClassificationDecision(**json.loads(line)))
    return decisions


def write_review_csv(decisions: Iterable[ClassificationDecision], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image_id", "source_path", "final_category", "confidence",
        "review_required", "review_reason", "evidence", "classifier_version",
    ]
    with target.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for decision in decisions:
            row = asdict(decision)
            row["evidence"] = "；".join(decision.evidence)
            writer.writerow(row)
    return target


def write_review_html(decisions: Iterable[ClassificationDecision], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    for decision in decisions:
        review_class = "review" if decision.review_required else "accepted"
        rows.append(
            "<tr class='{}'><td>{}</td><td>{}</td><td>{:.3f}</td>"
            "<td>{}</td><td>{}</td></tr>".format(
                review_class,
                html.escape(decision.image_id),
                html.escape(decision.final_category),
                decision.confidence,
                "是" if decision.review_required else "否",
                html.escape(decision.review_reason or ""),
            )
        )
    content = """<!doctype html>
<meta charset="utf-8">
<title>照片分类审核报告</title>
<style>
body { font-family: sans-serif; margin: 2rem; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #ccc; padding: .45rem; text-align: left; }
tr.review { background: #fff3cd; }
tr.accepted { background: #f3fff3; }
</style>
<h1>照片分类审核报告</h1>
<table><thead><tr><th>图片 ID</th><th>类别</th><th>置信度</th><th>需审核</th><th>原因</th></tr></thead>
<tbody>__PHOTO_CLASSIFIER_ROWS__</tbody></table>
""".replace("__PHOTO_CLASSIFIER_ROWS__", "\n".join(rows))
    target.write_text(content, encoding="utf-8")
    return target
