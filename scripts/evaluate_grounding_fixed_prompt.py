"""Evaluate a fixed-prompt GroundingDINO checkpoint on one reserved split."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_grounding_dino_subject_boxes import _batch, _box_iou, _read_manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _xywh(cxcywh: list[float]) -> list[float]:
    cx, cy, width, height = map(float, cxcywh)
    return [cx - width / 2, cy - height / 2, width, height]


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ious = [float(row["iou"]) for row in rows]
    return {
        "count": len(rows),
        "mean_loss": sum(float(row["loss"]) for row in rows) / len(rows),
        "mean_iou": sum(ious) / len(ious),
        "median_iou": statistics.median(ious),
        "iou_at_0_3": sum(value >= 0.3 for value in ious) / len(ious),
        "iou_at_0_5": sum(value >= 0.5 for value in ious) / len(ious),
        "iou_at_0_75": sum(value >= 0.75 for value in ious) / len(ious),
        "hits_at_0_5": sum(value >= 0.5 for value in ious),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--expected-count", type=int, default=None,
                        help="optionally require an exact number of rows in the selected split")
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoContrastiveEmbedding

    if args.split == "test":
        raise ValueError("Test evaluation is intentionally locked for this experiment")
    torch.set_num_threads(args.cpu_threads)
    manifest = args.manifest.resolve()
    model_path = args.model.resolve()
    output = args.output_dir.resolve()
    weights = model_path / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    rows = [row for row in _read_manifest(manifest) if row["split"] == args.split]
    if args.expected_count is not None and len(rows) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} {args.split} rows, found {len(rows)}")
    if any(row.get("prompt_mode") != "fixed_main_subject" for row in rows):
        raise ValueError("Every row must use fixed_main_subject")
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate evaluation IDs")

    fingerprint_payload = {
        "manifest": str(manifest),
        "model": str(model_path),
        "weights_sha256": _sha256(weights),
        "split": args.split,
        "prompt": "main subject.",
        "ids": [row["image_id"] for row in rows],
        "selection": "highest sigmoid score at the main-subject token position",
    }
    fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "run_config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old.get("run_fingerprint") != fingerprint:
            raise RuntimeError("Existing output belongs to a different evaluation run")
    else:
        config_path.write_text(json.dumps({**fingerprint_payload, "run_fingerprint": fingerprint,
            "true_category_provided_to_model": False, "test_split_used": False}, ensure_ascii=False, indent=2), encoding="utf-8")
    records_dir = output / "records"
    records_dir.mkdir(exist_ok=True)

    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(str(model_path), local_files_only=True)
    model.config.num_labels = int(model.config.max_text_len)
    def finite_padding(module: Any, inputs: Any, result: Any) -> Any:
        if torch.isnan(result).any() or torch.isposinf(result).any():
            raise FloatingPointError("Invalid non-padding contrastive logits")
        return result.masked_fill(torch.isneginf(result), -100.0)
    for module in model.modules():
        if isinstance(module, GroundingDinoContrastiveEmbedding):
            module.register_forward_hook(finite_padding)
    model.to("cpu").eval()

    completed: dict[str, dict[str, Any]] = {}
    for path in records_dir.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("run_fingerprint") == fingerprint:
            completed[record["image_id"]] = record
    started = time.perf_counter()
    with torch.inference_mode():
        for index, row in enumerate(rows, 1):
            ident = row["image_id"]
            if ident not in completed:
                item_started = time.perf_counter()
                with Image.open(row["source_path"]) as image:
                    image = image.convert("RGB")
                    inputs, target, token_index = _batch(model, processor, image, row, torch, "cpu")
                    result = model(**inputs, labels=[target])
                scores = result.logits[0, :, token_index].sigmoid()
                best = int(scores.argmax().item())
                predicted = result.pred_boxes[0, best].detach().cpu().tolist()
                expected = target["boxes"][0].detach().cpu().tolist()
                record = {
                    "image_id": ident,
                    "category": row["category"],
                    "split": row["split"],
                    "prompt": "main subject.",
                    "category_provided_to_model": False,
                    "prediction_cxcywh_norm": predicted,
                    "prediction_xywh_norm": _xywh(predicted),
                    "target_cxcywh_norm": expected,
                    "target_xywh_norm": row["primary_subject_box_xywh_norm"],
                    "score": float(scores[best].cpu()),
                    "loss": float(result.loss.cpu()),
                    "iou": _box_iou(predicted, expected),
                    "processing_seconds": time.perf_counter() - item_started,
                    "run_fingerprint": fingerprint,
                }
                (records_dir / f"{ident}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                completed[ident] = record
            if index % 10 == 0 or index == len(rows):
                progress = {"phase": "evaluating", "processed": index, "total": len(rows),
                            "elapsed_seconds": time.perf_counter() - started, "run_fingerprint": fingerprint}
                (output / "progress.json").write_text(json.dumps(progress, ensure_ascii=False), encoding="utf-8")
                print(f"evaluation {index}/{len(rows)}", flush=True)

    ordered = [completed[row["image_id"]] for row in rows]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in ordered:
        groups[record["category"]].append(record)
    summary = {
        "phase": "completed",
        "model": str(model_path),
        "weights_sha256": fingerprint_payload["weights_sha256"],
        "split": args.split,
        "prompt": "main subject.",
        "test_split_used": False,
        "overall": _metrics(ordered),
        "by_category": {category: _metrics(items) for category, items in sorted(groups.items())},
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "run_fingerprint": fingerprint,
    }
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for record in ordered:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "progress.json").write_text(json.dumps({"phase": "completed", "processed": len(rows),
        "total": len(rows), "run_fingerprint": fingerprint}, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
