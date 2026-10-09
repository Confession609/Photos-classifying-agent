"""Retrain the two frozen-OpenCLIP cascade heads on GD-generated crops.

The detector crop manifest is produced separately with a fixed ``main subject.``
prompt. Only train and validation rows are used; the test split is sealed.
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

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from photo_classifier_agent.hierarchy_v2 import V2_LEAF_LABELS, V2_NON_PERSON_LABELS, V2_PERSON_LABELS
from photo_classifier_agent.hierarchy_v2_training import _fit_head, _metrics, _open_clip
from scripts.train_v2_ablation import audit, digest, read_jsonl, write_json


def load_auto_manifest(path: Path) -> dict[str, dict]:
    rows = read_jsonl(path)
    by_id = {}
    for row in rows:
        identity = row["image_id"]
        if identity in by_id:
            raise ValueError(f"Duplicate auto-crop ID: {identity}")
        if row.get("category_provided_to_model") or row.get("caption_provided_to_model") or row.get("existing_box_provided_to_model"):
            raise ValueError(f"Auto-crop record indicates forbidden model input: {identity}")
        if not Path(row["crop_path"]).is_file():
            raise FileNotFoundError(row["crop_path"])
        by_id[identity] = row
    return by_id


def load_feature_cache(torch, path: Path) -> tuple[dict[str, object], dict[str, object], str]:
    original, automatic, keys = {}, {}, set()
    for shard in sorted(path.glob("*.pt")):
        values = torch.load(shard, map_location="cpu", weights_only=True)
        keys.add(values["key"])
        for index, identity in enumerate(values["ids"]):
            if identity in original:
                raise ValueError(f"Duplicate feature ID: {identity}")
            original[identity] = values["original"][index]
            automatic[identity] = values["auto_crop"][index]
    if len(keys) != 1:
        raise ValueError(f"Expected one feature-cache key, found {len(keys)}")
    return original, automatic, keys.pop()


def audit_manifest_only(path: Path) -> tuple[list[dict], dict, dict]:
    """Audit a self-contained manifest without requiring label-layer JSONL.

    The combined GD manifest can intentionally contain difficult/self-domain
    images whose source annotations do not live under ``label layer data``.
    Labels, source paths, split isolation and fixed-prompt provenance are still
    checked here; the automatic crop generator has already verified source
    hashes while producing its crop manifest.
    """
    rows = read_jsonl(path)
    ids = set()
    group_splits: dict[str, set[str]] = {}
    hash_splits: dict[str, set[str]] = {}
    for row in rows:
        identity = row["image_id"]
        if identity in ids:
            raise ValueError(f"Duplicate manifest ID: {identity}")
        ids.add(identity)
        if row.get("split") not in {"train", "validation", "test"}:
            raise ValueError(f"Unsupported split for {identity}: {row.get('split')}")
        if row.get("category") not in V2_LEAF_LABELS:
            raise ValueError(f"Unknown category for {identity}: {row.get('category')}")
        source = Path(row.get("source_path", ""))
        if not source.is_file():
            raise FileNotFoundError(source)
        if row.get("prompt_mode") != "fixed_main_subject" or row.get("primary_subject_label") != "main subject":
            raise ValueError(f"Manifest is not fixed-main-subject data: {identity}")
        if not row.get("sha256"):
            raise ValueError(f"Manifest is missing source hash: {identity}")
        group = row.get("group_id", identity)
        group_splits.setdefault(group, set()).add(row["split"])
        hash_splits.setdefault(row["sha256"], set()).add(row["split"])
    if any(len(splits) > 1 for splits in group_splits.values()):
        raise ValueError("Group leakage in manifest")
    if any(len(splits) > 1 for splits in hash_splits.values()):
        raise ValueError("Exact-image leakage in manifest")
    if not any(row["split"] == "train" for row in rows) or not any(row["split"] == "validation" for row in rows):
        raise ValueError("Manifest must contain train and validation rows")
    summary = {
        "manifest": str(path.resolve()),
        "manifest_sha256": digest(path),
        "counts": {split: dict(Counter(row["category"] for row in rows if row["split"] == split))
                   for split in ("train", "validation", "test")},
        "total": len(rows),
        "group_split_leaks": 0,
        "exact_duplicate_split_leaks": 0,
        "audit_mode": "manifest_only",
    }
    return rows, {}, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--auto-crop-manifest", type=Path, required=True)
    parser.add_argument("--validation-auto-crop-manifest", type=Path, default=None,
                        help="auto-crop manifest for validation rows generated with the same GD checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--manifest-only", action="store_true",
                        help="audit the supplied self-contained manifest without label-layer annotations")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output is not empty: {args.output}")
    if args.batch_size < 1 or args.threads < 1:
        raise ValueError("batch-size and threads must be positive")
    rows, _, audit_summary = (audit_manifest_only(args.manifest.resolve())
                              if args.manifest_only else audit(args.manifest.resolve()))
    selected = [row for row in rows if row["split"] != "test"]
    auto = load_auto_manifest(args.auto_crop_manifest.resolve())
    if args.validation_auto_crop_manifest is not None:
        validation_auto = load_auto_manifest(args.validation_auto_crop_manifest.resolve())
        auto.update(validation_auto)
    selected_ids = {row["image_id"] for row in selected}
    if set(auto) != selected_ids:
        missing, extra = sorted(selected_ids - set(auto)), sorted(set(auto) - selected_ids)
        raise ValueError(f"Auto-crop IDs mismatch; missing={missing[:5]}, extra={extra[:5]}")

    import torch
    from PIL import Image
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_num_threads(args.threads)
    weights = list((ROOT / "algorithms/models/huggingface/hub/models--timm--vit_base_patch32_clip_224.openai/snapshots").glob("*/open_clip_model.safetensors"))
    if len(weights) != 1:
        raise FileNotFoundError("Expected exactly one local OpenCLIP checkpoint")
    torch, Image, model, preprocess, device = _open_clip("ViT-B-32-quickgelu", str(weights[0]), "cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "data_audit.json", {**audit_summary, "auto_crop_manifest": str(args.auto_crop_manifest.resolve()), "validation_auto_crop_manifest": str(args.validation_auto_crop_manifest.resolve()) if args.validation_auto_crop_manifest else None, "auto_crop_model_input": "image + fixed main subject prompt only", "test_used": False})
    cache_dir = args.output / "feature_cache_quickgelu"
    cache_dir.mkdir()
    cache_key = hashlib.sha256(json.dumps({"ids": [r["image_id"] for r in selected], "openclip_weights": digest(weights[0]), "model": "ViT-B-32-quickgelu", "preprocess": repr(preprocess), "torch": torch.__version__, "auto_crop_manifest": digest(args.auto_crop_manifest)}, sort_keys=True).encode()).hexdigest()
    original_shards, auto_shards = [], []
    started = time.monotonic()
    for start in range(0, len(selected), args.batch_size):
        batch = selected[start:start + args.batch_size]
        tensors_original, tensors_auto = [], []
        for row in batch:
            with Image.open(row["source_path"]) as image:
                tensors_original.append(preprocess(image.convert("RGB")))
            with Image.open(auto[row["image_id"]]["crop_path"]) as image:
                tensors_auto.append(preprocess(image.convert("RGB")))
        with torch.inference_mode():
            original = model.encode_image(torch.stack(tensors_original)).float().cpu()
            automatic = model.encode_image(torch.stack(tensors_auto)).float().cpu()
            original = original / original.norm(dim=1, keepdim=True).clamp_min(1e-12)
            automatic = automatic / automatic.norm(dim=1, keepdim=True).clamp_min(1e-12)
        assert torch.isfinite(original).all() and torch.isfinite(automatic).all()
        shard = {"key": cache_key, "ids": [r["image_id"] for r in batch], "original": original, "auto_crop": automatic}
        torch.save(shard, cache_dir / f"{start:06d}.pt")
        original_shards.append(original)
        auto_shards.append(automatic)
        progress = {"phase": "extract_features", "done": start + len(batch), "total": len(selected), "elapsed_seconds": time.monotonic() - started}
        write_json(args.output / "progress.json", progress)
        print(json.dumps(progress, ensure_ascii=False), flush=True)
    del model
    original = torch.cat(original_shards)
    automatic = torch.cat(auto_shards)
    variants = {"original": original, "original_auto_crop": torch.cat((original, automatic), dim=1) / math.sqrt(2)}
    results = {}
    for variant, features in variants.items():
        folder = args.output / variant
        folder.mkdir()
        heads, histories, metrics = {}, {}, {}
        val_idx = [i for i, row in enumerate(selected) if row["split"] == "validation"]
        val_rows = [selected[i] for i in val_idx]
        probabilities = {}
        for node, names in (("person_gate", V2_PERSON_LABELS), ("non_person_classifier", V2_NON_PERSON_LABELS)):
            indices = {split: [i for i, row in enumerate(selected) if row["split"] == split and (node == "person_gate" or row["category"] != "人像")] for split in ("train", "validation")}
            targets = {}
            for split, indexes in indices.items():
                labels = [("人像" if selected[i]["category"] == "人像" else "非人像") if node == "person_gate" else selected[i]["category"] for i in indexes]
                targets[split] = torch.tensor([names.index(label) for label in labels])
                if set(targets[split].tolist()) != set(range(len(names))):
                    raise ValueError(f"Missing class in {variant}/{node}/{split}")
            head, history = _fit_head(torch, features[indices["train"]], targets["train"], features[indices["validation"]], targets["validation"], names, epochs=args.epochs, patience=args.patience, learning_rate=0.01, seed=1337 + (100 if node != "person_gate" else 0))
            heads[node], histories[node] = head, history
            layer = torch.nn.Linear(features.shape[1], len(names))
            layer.load_state_dict(head["state_dict"])
            with torch.inference_mode():
                metrics[node] = _metrics(layer(features[indices["validation"]]), targets["validation"], names)
                probabilities[node] = layer(features[val_idx]).softmax(dim=1)
        gate, nonperson = probabilities["person_gate"], probabilities["non_person_classifier"]
        predicted = torch.where(gate[:, 0] >= 0.5, 0, nonperson.argmax(dim=1) + 1)
        truth = torch.tensor([V2_LEAF_LABELS.index(row["category"]) for row in val_rows])
        metrics["cascade"] = _metrics(torch.nn.functional.one_hot(predicted, 4).float(), truth, V2_LEAF_LABELS)
        confidence = torch.where(predicted == 0, gate[:, 0], gate[:, 1] * nonperson.max(dim=1).values)
        metrics["cascade"]["review_rate_at_0.75_uncalibrated"] = float((confidence < 0.75).float().mean())
        checkpoint = {"format": "v2_ablation_v1", "variant": variant, "heads": heads, "feature_dim": features.shape[1], "model_name": "ViT-B-32-quickgelu", "pretrained_path": str(weights[0]), "feature_normalization": "L2 each; concatenate original and GD auto crop then divide sqrt(2)", "cache_key": cache_key, "manifest_sha256": audit_summary["manifest_sha256"], "leaf_labels": V2_LEAF_LABELS, "auto_crop_manifest": str(args.auto_crop_manifest.resolve()), "test_evaluated": False}
        torch.save(checkpoint, folder / "heads.pt")
        torch.load(folder / "heads.pt", weights_only=True)
        write_json(folder / "training_history.json", histories)
        write_json(folder / "validation_metrics.json", metrics)
        with (folder / "validation_predictions.jsonl").open("w", encoding="utf-8") as stream:
            for i, row in enumerate(val_rows):
                stream.write(json.dumps({"image_id": row["image_id"], "truth": row["category"], "predicted": V2_LEAF_LABELS[int(predicted[i])], "correct": int(predicted[i]) == int(truth[i]), "confidence_uncalibrated": float(confidence[i])}, ensure_ascii=False) + "\n")
        results[variant] = metrics
        print(variant, json.dumps(metrics, ensure_ascii=False), flush=True)
    winner = max(results, key=lambda name: results[name]["cascade"]["macro_f1"])
    write_json(args.output / "comparison.json", {"validation": results, "provisional_choice": winner, "selection_metric": "cascade validation macro_f1", "test_evaluated": False, "heads_retrained": True, "epochs_max": args.epochs, "patience": args.patience, "learning_rate": 0.01, "auto_crop_manifest": str(args.auto_crop_manifest.resolve()), "audit": audit_summary})
    write_json(args.output / "progress.json", {"phase": "completed", "test_evaluated": False, "provisional_choice": winner})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
