"""Frozen OpenCLIP original/crop ablation using the existing subject split.

Only train and validation images are encoded. Test remains sealed. Checkpoints
are experiment-specific (512/1024 inputs), not legacy V2CascadeClassifier files.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from photo_classifier_agent.hierarchy_v2 import V2_LEAF_LABELS, V2_NON_PERSON_LABELS, V2_PERSON_LABELS
from photo_classifier_agent.hierarchy_v2_training import _fit_head, _metrics, _open_clip

FOLDERS = dict(zip(V2_LEAF_LABELS, ("portraits", "landscapes", "still_life", "events")))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, obj):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def audit(manifest):
    rows = read_jsonl(manifest)
    annotations = {}
    for category, folder in FOLDERS.items():
        for record in read_jsonl(ROOT / "data/dataset/label layer data" / folder / "subject_annotations.jsonl"):
            if not record.get("image_id"):
                continue
            if record["image_id"] in annotations:
                raise ValueError("Duplicate annotation ID")
            annotations[record["image_id"]] = record
    groups, hashes = defaultdict(set), defaultdict(set)
    ids = set()
    image_hashes = {}
    for row in rows:
        identity = row["image_id"]
        if identity in ids:
            raise ValueError(f"Duplicate manifest ID: {identity}")
        ids.add(identity)
        assert row["split"] in ("train", "validation", "test")
        assert row["category"] in V2_LEAF_LABELS
        ann = annotations[identity]
        assert ann["status"] == "success" and ann["category"] == row["category"], identity
        assert ann.get("human_review", {}).get("outcome") != "discarded", identity
        assert ann["primary_subject_box_xywh_norm"] == row["primary_subject_box_xywh_norm"], identity
        assert Path(ann["source_path"]).resolve() == Path(row["source_path"]).resolve(), identity
        assert Path(ann["subject_crop_path"]).resolve() == Path(row["crop_path"]).resolve(), identity
        x, y, w, h = row["primary_subject_box_xywh_norm"]
        assert min(x, y) >= 0 and min(w, h) > 0 and x+w <= 1.00001 and y+h <= 1.00001, identity
        actual_hash = digest(row["source_path"])
        assert actual_hash == row["sha256"], f"Original changed: {identity}"
        assert Path(row["crop_path"]).is_file(), identity
        groups[row["group_id"]].add(row["split"])
        hashes[actual_hash].add(row["split"])
        if row["split"] != "test":
            image_hashes[identity] = [actual_hash, digest(row["crop_path"])]
    active_ids = {key for key, value in annotations.items() if value.get("status") == "success" and value.get("human_review", {}).get("outcome") != "discarded"}
    assert ids == active_ids, "Manifest does not cover exactly the current successful annotations"
    assert all(len(v) == 1 for v in groups.values()), "Group leakage"
    assert all(len(v) == 1 for v in hashes.values()), "Identical-image leakage"
    summary = {
        "manifest": str(manifest), "manifest_sha256": digest(manifest),
        "counts": {s: dict(Counter(r["category"] for r in rows if r["split"] == s)) for s in ("train", "validation", "test")},
        "total": len(rows), "group_split_leaks": 0, "exact_duplicate_split_leaks": 0,
        "limitation": "Existing group IDs checked; shooting-event grouping is not independently certified. Crops may reflect category-guided detection and human review, so this is a curated-input experiment, not end-to-end deployment performance.",
    }
    return rows, image_hashes, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/subject_layer_florence_caption_ft/subject_caption_manifest.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/v2_original_crop_ablation_2026-09-28")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    assert args.batch_size > 0 and args.threads > 0
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "comparison.json").exists():
        raise FileExistsError("Completed run already exists; choose a fresh output directory")
    rows, hashes, summary = audit(args.manifest)
    write_json(output / "data_audit.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if args.audit_only:
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    weights = list((ROOT / "algorithms/models/huggingface/hub/models--timm--vit_base_patch32_clip_224.openai/snapshots").glob("*/open_clip_model.safetensors"))
    if len(weights) != 1:
        raise FileNotFoundError("Expected exactly one local OpenCLIP checkpoint")
    import torch
    torch.set_num_threads(args.threads)
    torch, Image, model, preprocess, device = _open_clip("ViT-B-32-quickgelu", str(weights[0]), "cpu")
    selected = [r for r in rows if r["split"] != "test"]
    cache = output / "feature_cache_quickgelu"
    cache.mkdir(exist_ok=True)
    cache_key = hashlib.sha256(json.dumps({"hashes": hashes, "weights": digest(weights[0]), "model_name": "ViT-B-32-quickgelu", "preprocess": repr(preprocess), "torch": torch.__version__}, sort_keys=True).encode()).hexdigest()
    original, cropped = [], []
    started = time.monotonic()
    for start in range(0, len(selected), args.batch_size):
        batch = selected[start:start + args.batch_size]
        batch_ids = [r["image_id"] for r in batch]
        path = cache / f"{start:06d}.pt"
        if path.exists():
            values = torch.load(path, weights_only=True)
            if values["key"] != cache_key or values["ids"] != batch_ids:
                raise ValueError("Stale cache; use a fresh output directory")
        else:
            features = []
            for field in ("source_path", "crop_path"):
                tensors = []
                for row in batch:
                    with Image.open(row[field]) as im:
                        tensors.append(preprocess(im.convert("RGB")))
                with torch.inference_mode():
                    vec = model.encode_image(torch.stack(tensors).to(device)).float().cpu()
                    vec = vec / vec.norm(dim=1, keepdim=True).clamp_min(1e-12)
                assert torch.isfinite(vec).all()
                features.append(vec)
            values = {"key": cache_key, "ids": batch_ids, "original": features[0], "crop": features[1]}
            temp = path.with_suffix(".tmp")
            torch.save(values, temp)
            os.replace(temp, path)
        original.append(values["original"])
        cropped.append(values["crop"])
        progress = {"phase": "extract_features", "done": start+len(batch), "total": len(selected), "elapsed_seconds": time.monotonic()-started}
        write_json(output / "progress.json", progress)
        print(json.dumps(progress), flush=True)
    del model
    original, cropped = torch.cat(original), torch.cat(cropped)
    results = {}
    for variant, features in (("original", original), ("original_crop", torch.cat((original, cropped), dim=1) / 2**0.5)):
        folder = output / variant
        folder.mkdir(exist_ok=True)
        heads, history, metrics = {}, {}, {}
        val_idx = [i for i, r in enumerate(selected) if r["split"] == "validation"]
        val_rows = [selected[i] for i in val_idx]
        probabilities = {}
        for node, names in (("person_gate", V2_PERSON_LABELS), ("non_person_classifier", V2_NON_PERSON_LABELS)):
            indices = {s: [i for i,r in enumerate(selected) if r["split"] == s and (node == "person_gate" or r["category"] != "人像")] for s in ("train", "validation")}
            targets = {}
            for s, ix in indices.items():
                labels = [("人像" if selected[i]["category"] == "人像" else "非人像") if node == "person_gate" else selected[i]["category"] for i in ix]
                targets[s] = torch.tensor([names.index(label) for label in labels])
                assert set(targets[s].tolist()) == set(range(len(names)))
            head, hist = _fit_head(torch, features[indices["train"]], targets["train"], features[indices["validation"]], targets["validation"], names, epochs=args.epochs, patience=args.patience, learning_rate=0.01, seed=1337 + (100 if node != "person_gate" else 0))
            heads[node], history[node] = head, hist
            layer = torch.nn.Linear(features.shape[1], len(names))
            layer.load_state_dict(head["state_dict"])
            with torch.inference_mode():
                metrics[node] = _metrics(layer(features[indices["validation"]]), targets["validation"], names)
                probabilities[node] = layer(features[val_idx]).softmax(dim=1)
        gate, nonperson = probabilities["person_gate"], probabilities["non_person_classifier"]
        predicted = torch.where(gate[:,0] >= 0.5, 0, nonperson.argmax(dim=1)+1)
        truth = torch.tensor([V2_LEAF_LABELS.index(r["category"]) for r in val_rows])
        metrics["cascade"] = _metrics(torch.nn.functional.one_hot(predicted, 4).float(), truth, V2_LEAF_LABELS)
        confidence = torch.where(predicted == 0, gate[:,0], gate[:,1]*nonperson.max(dim=1).values)
        metrics["cascade"]["review_rate_at_0.75_uncalibrated"] = float((confidence < 0.75).float().mean())
        checkpoint = {"format": "v2_ablation_v1", "variant": variant, "heads": heads, "feature_dim": features.shape[1], "model_name": "ViT-B-32-quickgelu", "pretrained_path": str(weights[0]), "feature_normalization": "L2 each; concatenate and divide sqrt(2) for original_crop", "cache_key": cache_key, "manifest_sha256": summary["manifest_sha256"], "leaf_labels": V2_LEAF_LABELS}
        torch.save(checkpoint, folder / "heads.pt")
        torch.load(folder / "heads.pt", weights_only=True)
        write_json(folder / "training_history.json", history)
        write_json(folder / "validation_metrics.json", metrics)
        with (folder / "validation_predictions.jsonl").open("w", encoding="utf-8") as stream:
            for i,r in enumerate(val_rows):
                stream.write(json.dumps({"image_id": r["image_id"], "truth": r["category"], "predicted": V2_LEAF_LABELS[int(predicted[i])], "confidence_uncalibrated": float(confidence[i]), "person_probs": gate[i].tolist(), "non_person_probs": nonperson[i].tolist()}, ensure_ascii=False)+"\n")
        results[variant] = metrics
        print(variant, json.dumps(metrics, ensure_ascii=False), flush=True)
    winner = max(results, key=lambda v: results[v]["cascade"]["macro_f1"])
    write_json(output / "comparison.json", {"validation": results, "provisional_choice": winner, "selection_metric": "cascade validation macro_f1", "test_evaluated": False, "epochs_max": args.epochs, "patience": args.patience, "seed": 1337, "learning_rate": 0.01, "loss": "independent class-balanced cross entropy for each head", "input_scope": "curated originals/crops; no captions or category-derived scalar features", "audit": summary})
    write_json(output / "progress.json", {"phase": "completed", "test_evaluated": False, "provisional_choice": winner})


if __name__ == "__main__":
    main()
