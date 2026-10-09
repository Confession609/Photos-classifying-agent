"""Create per-category subject captions, boxes, and crop files in-place.

Only the four explicit curated class directories are scanned.  Staging folders
are siblings and are therefore never traversed.  Output checkpoints are
rewritten atomically so a cancelled batch can resume without duplicate rows.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .dataset import SUPPORTED_EXTENSIONS
from .subject import SubjectAnalyzer, _image_id

CATEGORY_DIRECTORIES = {
    "portraits": "人像",
    "landscapes": "风光摄影",
    "still_life": "静物摄影",
    "events": "活动事件摄影",
}
ANNOTATION_FILENAME = "subject_annotations.jsonl"
CROP_DIRECTORY = "subject_crops"
REPORT_MODEL_VERSION = "Florence-2-base-ft+grounding-dino-tiny:category-guided-v4"


def scan_curated_categories(dataset_root: str | Path) -> dict[str, list[Path]]:
    """List supported originals under exact curated folder names only."""
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"label-layer dataset root does not exist: {root}")
    result: dict[str, list[Path]] = {}
    for directory_name in CATEGORY_DIRECTORIES:
        category_root = root / directory_name
        if not category_root.is_dir():
            raise FileNotFoundError(f"required curated category directory is missing: {category_root}")
        paths = []
        for path in sorted(category_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            relative_parts = path.relative_to(category_root).parts
            if CROP_DIRECTORY in relative_parts or any(part.endswith("_staging") for part in relative_parts):
                continue
            if path.name == ANNOTATION_FILENAME:
                continue
            paths.append(path.resolve())
        result[directory_name] = paths
    return result


def _read_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    existing: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid annotation JSONL at {path}:{line_number}; refusing to overwrite") from exc
            if row.get("image_id"):
                existing[row["image_id"]] = row
    return existing


def _is_completed(row: dict[str, Any] | None) -> bool:
    if row is None or row.get("model_version") != REPORT_MODEL_VERSION or row.get("status") not in {"success", "needs_review"}:
        return False
    crop_path = row.get("subject_crop_path")
    if crop_path:
        return Path(crop_path).is_file()
    return row.get("status") == "needs_review"


def _atomic_write_jsonl(path: Path, records: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for image_id in sorted(records):
                handle.write(json.dumps(records[image_id], ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _region_row(region: Any) -> dict[str, Any]:
    return {
        "label": region.label,
        "box_xywh_norm": [region.x, region.y, region.width, region.height],
        "confidence": region.confidence,
    }


def _make_record(path: Path, directory_name: str, analyzer: SubjectAnalyzer) -> dict[str, Any]:
    category_root = path.parents[0]
    while category_root.name != directory_name and category_root.parent != category_root:
        category_root = category_root.parent
    crop_dir = category_root / CROP_DIRECTORY
    report = analyzer.analyze(path, crop_dir, category=CATEGORY_DIRECTORIES[directory_name])
    from PIL import Image

    with Image.open(path) as source_image:
        image_size = [source_image.width, source_image.height]
    regions = list(report.subject_regions)
    selected = analyzer._choose_region(report.subject_regions)
    selected_row = _region_row(selected) if selected else None
    caption = report.main_subject_description
    crop_exists = bool(report.subject_crop_path and Path(report.subject_crop_path).is_file())
    if not crop_exists:
        stale_crop = crop_dir / f"{report.image_id}.jpg"
        if stale_crop.is_file():
            stale_crop.unlink()
    status = "success" if selected_row and crop_exists and caption else "needs_review"
    reasons = []
    if not caption:
        reasons.append("主体描述为空")
    if not selected_row:
        reasons.append("未找到可靠主体框")
    elif not crop_exists:
        reasons.append("主体裁剪图未生成")
    elif float(selected.confidence or 0.0) < 0.35:
        reasons.append("主体框置信度低于0.35")
    if reasons:
        status = "needs_review"
    return {
        "image_id": report.image_id,
        "category": CATEGORY_DIRECTORIES[directory_name],
        "source_path": str(path.resolve()),
        "source_relative_path": str(path.resolve().relative_to(category_root.resolve())),
        "image_size": image_size,
        "main_subject_description": caption,
        "primary_subject_label": selected.label if selected else None,
        "primary_subject_box_xywh_norm": selected_row["box_xywh_norm"] if selected_row else None,
        "primary_subject_confidence": selected.confidence if selected else None,
        "candidate_boxes": [_region_row(region) for region in regions],
        "subject_crop_path": report.subject_crop_path if crop_exists else None,
        "status": status,
        "review_reasons": reasons,
        "model_name": report.model_name,
        "model_version": report.model_version,
    }


def generate_category_subject_data(
    dataset_root: str | Path,
    *,
    device: str | None = None,
    limit_per_category: int | None = None,
    checkpoint_every: int = 10,
) -> dict[str, Any]:
    """Generate/resume all per-category JSONL and subject crop outputs."""
    if limit_per_category is not None and limit_per_category < 1:
        raise ValueError("limit_per_category must be positive")
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    scanned = scan_curated_categories(dataset_root)
    analyzer = SubjectAnalyzer(device=device)
    summary: dict[str, Any] = {"categories": {}, "staging_directories_included": False}

    for directory_name, paths in scanned.items():
        if limit_per_category is not None:
            paths = paths[:limit_per_category]
        class_root = Path(dataset_root).expanduser().resolve() / directory_name
        annotation_path = class_root / ANNOTATION_FILENAME
        records = _read_existing(annotation_path)
        before = len(records)
        pending = [path for path in paths if not _is_completed(records.get(_image_id(path)))]
        print(
            f"[{directory_name}] images={len(paths)} completed={len(paths) - len(pending)} pending={len(pending)}",
            flush=True,
        )
        processed_since_checkpoint = 0
        for index, path in enumerate(pending, 1):
            image_id = _image_id(path)
            try:
                record = _make_record(path, directory_name, analyzer)
            except Exception as exc:
                record = {
                    "image_id": image_id,
                    "category": CATEGORY_DIRECTORIES[directory_name],
                    "source_path": str(path),
                    "source_relative_path": str(path.relative_to(class_root)),
                    "main_subject_description": None,
                    "primary_subject_label": None,
                    "primary_subject_box_xywh_norm": None,
                    "primary_subject_confidence": None,
                    "candidate_boxes": [],
                    "subject_crop_path": None,
                    "status": "failed",
                    "review_reasons": [repr(exc)],
                    "model_name": "Florence-2+GroundingDINO",
                    "model_version": REPORT_MODEL_VERSION,
                }
            records[image_id] = record
            processed_since_checkpoint += 1
            if processed_since_checkpoint >= checkpoint_every or index == len(pending):
                _atomic_write_jsonl(annotation_path, records)
                processed_since_checkpoint = 0
            if index == 1 or index % 10 == 0 or index == len(pending):
                print(f"[{directory_name}] processed {index}/{len(pending)}", flush=True)

        # Ensure a report exists for empty/already-complete classes and compact
        # any prior duplicate rows to one latest record per image_id.
        _atomic_write_jsonl(annotation_path, records)
        class_rows = [records.get(_image_id(path)) for path in paths]
        status_counts = Counter(row["status"] for row in class_rows if row)
        summary["categories"][directory_name] = {
            "label": CATEGORY_DIRECTORIES[directory_name],
            "images": len(paths),
            "annotations": sum(row is not None for row in class_rows),
            "new_or_retried": len(pending),
            "prior_annotation_rows": before,
            "status_counts": dict(status_counts),
            "annotation_file": str(annotation_path.resolve()),
            "crop_directory": str((class_root / CROP_DIRECTORY).resolve()),
        }

    return summary
