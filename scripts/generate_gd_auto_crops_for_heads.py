"""Generate fixed-prompt GroundingDINO crops for downstream-head training.

The model receives only the image and the same ``main subject.`` prompt for
every image. Category, captions, existing boxes and existing crops are used
only to select the requested split and are never passed to the model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def token_start(input_ids: Any, phrase_ids: list[int]) -> int:
    ids = input_ids[0].tolist()
    for start in range(0, len(ids) - len(phrase_ids) + 1):
        if ids[start:start + len(phrase_ids)] == phrase_ids:
            return start
    raise ValueError("fixed prompt tokens were not found")


def xywh(cxcywh: list[float]) -> list[float]:
    cx, cy, width, height = map(float, cxcywh)
    return [cx - width / 2, cy - height / 2, width, height]


def crop_image(Image, source: Path, box: list[float], destination: Path) -> list[int]:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
        width, height = image.size
        x, y, w, h = box
        left = max(0, min(width - 1, math.floor(x * width)))
        top = max(0, min(height - 1, math.floor(y * height)))
        right = max(left + 1, min(width, math.ceil((x + w) * width)))
        bottom = max(top + 1, min(height, math.ceil((y + h) * height)))
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.crop((left, top, right, bottom)).save(destination, format="JPEG", quality=95)
    return [left, top, right, bottom]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "examination", "inter", "test"),
        default="train",
        help="manifest split to process; test is supported for explicit experiments only",
    )
    parser.add_argument("--role", choices=("category", "withoutlabel"), default="category",
                        help="manifest role used as model input; category is the default")
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0, help="optional prefix limit for a smoke test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--resume", action="store_true",
                        help="reuse valid records already present in output-dir")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Output is not empty: {args.output_dir}")
    # Unified manifests keep a mirrored ``withoutlabel`` row for every
    # category row.  Inference and downstream training must process one image
    # record only, so restrict this utility to the category role.  Older
    # experiment manifests have no role field and are treated as category
    # records for backward compatibility.
    rows = [row for row in read_jsonl(args.manifest)
            if row.get("split") == args.split and row.get("role", "category") == args.role]
    # The live unified manifest names the stable content ID
    # ``source_content_hash``; older manifests use ``image_id``.  Normalize
    # both forms for crop filenames and downstream joins.
    for row in rows:
        if not row.get("image_id"):
            row["image_id"] = row.get("source_content_hash")
        if not row.get("image_id"):
            raise ValueError("Manifest row is missing image_id/source_content_hash")
    if args.limit:
        rows = rows[:args.limit]
    if args.expected_count is not None and len(rows) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} {args.split} rows, found {len(rows)}")
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate input IDs")
    for row in rows:
        source_path = row.get("source_path", row.get("path"))
        if not source_path or not Path(source_path).is_file():
            raise FileNotFoundError(source_path or row["image_id"])

    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoContrastiveEmbedding

    torch.set_num_threads(args.cpu_threads)
    model_path = args.model.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    crops = output / "crops"
    records_dir = output / "records"
    crops.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)
    model_hash = sha(model_path / "model.safetensors")

    row_ids = {row["image_id"] for row in rows}
    existing: dict[str, dict[str, Any]] = {}
    if args.resume:
        for record_path in sorted(records_dir.glob("*.json")):
            record = json.loads(record_path.read_text(encoding="utf-8"))
            identity = record.get("image_id")
            if identity not in row_ids:
                raise ValueError(f"Existing record is not part of this input split: {identity}")
            if identity in existing:
                raise ValueError(f"Duplicate existing record: {identity}")
            crop_path = Path(record.get("crop_path", ""))
            if record.get("run_fingerprint") != model_hash or not crop_path.is_file():
                continue
            with Image.open(crop_path) as check:
                check.verify()
            existing[identity] = record
    remaining = [row for row in rows if row["image_id"] not in existing]
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
    prompt = "main subject."
    prompt_ids = processor.tokenizer("main subject", add_special_tokens=False)["input_ids"]
    started = time.monotonic()
    results_by_id: dict[str, dict[str, Any]] = dict(existing)
    with torch.inference_mode():
        for start in range(0, len(remaining), args.batch_size):
            batch = remaining[start:start + args.batch_size]
            images = []
            for row in batch:
                source = Path(row.get("source_path", row.get("path")))
                expected_hash = row.get("sha256", row.get("content_hash"))
                if expected_hash and sha(source) != expected_hash:
                    raise ValueError(f"Source changed: {row['image_id']}")
                with Image.open(source) as opened:
                    images.append(opened.convert("RGB"))
            inputs = processor(images=images, text=[prompt] * len(batch), return_tensors="pt", padding=True)
            inputs = {key: value.to("cpu") for key, value in inputs.items() if hasattr(value, "to")}
            index_token = token_start(inputs["input_ids"], prompt_ids)
            if index_token >= int(model.config.max_text_len):
                raise ValueError("prompt token exceeds model limit")
            result_model = model(**inputs)
            scores_batch = result_model.logits[:, :, index_token].sigmoid()
            best_batch = scores_batch.argmax(dim=1).tolist()
            for local_index, row in enumerate(batch):
                image_id = row["image_id"]
                source = Path(row.get("source_path", row.get("path")))
                best = int(best_batch[local_index])
                predicted_cxcywh = result_model.pred_boxes[local_index, best].detach().cpu().tolist()
                predicted_xywh = xywh(predicted_cxcywh)
                if predicted_xywh[2] <= 0 or predicted_xywh[3] <= 0:
                    raise ValueError(f"invalid prediction box: {image_id}")
                crop_path = crops / f"{image_id}.jpg"
                pixel_box = crop_image(Image, source, predicted_xywh, crop_path)
                with Image.open(crop_path) as check:
                    check.verify()
                result = {
                    "image_id": image_id,
                    "source_path": str(source.resolve()),
                    "split": args.split,
                    "prompt": prompt,
                    "category_provided_to_model": False,
                    "caption_provided_to_model": False,
                    "existing_box_provided_to_model": False,
                    "prediction_xywh_norm": predicted_xywh,
                    "score": float(scores_batch[local_index, best].cpu()),
                    "crop_path": str(crop_path.resolve()),
                    "crop_box_xyxy_pixels": pixel_box,
                    "crop_sha256": sha(crop_path),
                    "run_fingerprint": model_hash,
                }
                write_json(records_dir / f"{image_id}.json", result)
                results_by_id[image_id] = result
                images[local_index].close()
            processed = len(results_by_id)
            progress = {"phase": "running", "processed": processed, "total": len(rows), "statuses": {"success": processed}, "elapsed_seconds": time.monotonic() - started}
            write_json(output / "progress.json", progress)
            print(json.dumps(progress, ensure_ascii=False), flush=True)
    results = [results_by_id[row["image_id"]] for row in rows]
    with (output / "auto_crop_manifest.jsonl").open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    summary = {"phase": "completed", "split": args.split, "role": args.role, "processed": len(results), "available": len(rows), "resumed_records": len(existing), "test_used": args.split == "test", "prompt": prompt, "model": str(model_path), "model_sha256": model_hash, "batch_size": args.batch_size, "elapsed_seconds": time.monotonic() - started}
    write_json(output / "summary.json", summary)
    write_json(output / "progress.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
