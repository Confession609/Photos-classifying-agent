"""Prepare and fine-tune Florence-2 on verified subject/scene captions.

Each source photo creates two paired examples:
  * original image -> scene caption
  * verified subject crop -> subject caption

The manifest is built from the four curated annotation JSONL files, excludes
discarded rows, and keeps perceptually near-duplicate photos in the same split.
GroundingDINO box fine-tuning is implemented separately in
``train_grounding_dino_subject_boxes.py`` and consumes the same split manifest.

Run ``--prepare-manifest --dry-run`` first. Training requires ``--train`` and,
on CPU-only installations, the explicit ``--allow-cpu`` acknowledgement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "dataset" / "label layer data"
MODEL_ROOT = ROOT / "algorithms" / "models" / "microsoft__Florence-2-base-ft"
DEFAULT_OUTPUT = ROOT / "artifacts" / "subject_layer_florence_caption_sampled_ft"
DEFAULT_INIT_MODEL = ROOT / "artifacts" / "subject_layer_florence_caption_ft" / "best_model"
DEFAULT_MANIFEST = ROOT / "artifacts" / "subject_layer_florence_caption_ft" / "subject_caption_manifest.jsonl"
CATEGORIES = {
    "portraits": "人像",
    "landscapes": "风光摄影",
    "still_life": "静物摄影",
    "events": "活动事件摄影",
}
PROMPT = "<MORE_DETAILED_CAPTION>"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL {path}:{line_number}: {exc}") from exc
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _difference_hash(path: Path) -> int:
    from PIL import Image

    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).getdata())
    value = 0
    for row in range(8):
        offset = row * 9
        for col in range(8):
            value = (value << 1) | int(pixels[offset + col] > pixels[offset + col + 1])
    return value


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _split_groups(rows: list[dict[str, Any]], seed: int, hash_distance: int) -> int:
    """Assign 70/15/15 splits by exact/perceptual groups, never by single file."""
    union = _UnionFind(len(rows))
    exact_owner: dict[str, int] = {}
    for index, row in enumerate(rows):
        previous = exact_owner.setdefault(row["sha256"], index)
        if rows[previous]["category"] != row["category"]:
            raise ValueError(
                "Identical image content has conflicting labels: "
                f"{rows[previous]['image_id']} vs {row['image_id']}"
            )
        union.union(index, previous)

    # The corpus is small enough for pairwise 64-bit dHash comparison (~2M pairs).
    hashes = [row["dhash"] for row in rows]
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            if (hashes[left] ^ hashes[right]).bit_count() <= hash_distance:
                union.union(left, right)

    members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        members[union.find(index)].append(index)
    groups = list(members.values())
    group_counts: dict[int, Counter[str]] = {}
    total_by_category = Counter(row["category"] for row in rows)
    for group_index, indices in enumerate(groups):
        counts = Counter(rows[index]["category"] for index in indices)
        group_counts[group_index] = counts

    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=lambda indices: len(indices), reverse=True)
    target_fractions = {"train": 0.70, "validation": 0.15, "test": 0.15}
    assigned: dict[str, Counter[str]] = {split: Counter() for split in target_fractions}
    group_splits: dict[int, str] = {}
    for group_index, indices in enumerate(groups):
        counts = Counter(rows[index]["category"] for index in indices)
        scores = []
        for candidate in target_fractions:
            # Score the complete three-way allocation, not only the candidate
            # bucket; otherwise one split can be overfilled while others lag.
            score = 0.0
            for split, fraction in target_fractions.items():
                for category, total in total_by_category.items():
                    target = total * fraction
                    proposed = assigned[split][category]
                    if split == candidate:
                        proposed += counts[category]
                    score += ((proposed - target) / max(target, 1.0)) ** 2
            scores.append((score, rng.random(), candidate))
        selected = min(scores)[2]
        group_splits[group_index] = selected
        assigned[selected].update(counts)

    for group_index, indices in enumerate(groups):
        split = group_splits[group_index]
        group_id = hashlib.sha1("|".join(sorted(rows[i]["sha256"] for i in indices)).encode()).hexdigest()[:16]
        for index in indices:
            rows[index]["split"] = split
            rows[index]["group_id"] = group_id

    per_category_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        per_category_splits[row["category"]].add(row["split"])
    missing = {category: {"train", "validation", "test"} - splits for category, splits in per_category_splits.items()}
    missing = {category: splits for category, splits in missing.items() if splits}
    if missing:
        raise ValueError(f"Group split failed to place every class in all splits: {missing}")
    return len(groups)


def prepare_manifest(
    data_root: Path,
    manifest_path: Path,
    *,
    seed: int = 1337,
    hash_distance: int = 4,
    dry_run: bool = False,
) -> dict[str, Any]:
    if not 0 <= hash_distance <= 16:
        raise ValueError("--hash-distance must be between 0 and 16")
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    category_counts: Counter[str] = Counter()

    for folder, expected_category in CATEGORIES.items():
        source = data_root / folder / "subject_annotations.jsonl"
        if not source.is_file():
            raise FileNotFoundError(f"Missing category annotation file: {source}")
        for record in _load_jsonl(source):
            if not record.get("image_id"):
                continue  # summary rows
            review = record.get("human_review") or {}
            if record.get("status") == "discarded" or review.get("outcome") == "discarded":
                continue
            if record.get("status") != "success":
                raise ValueError(f"Non-success annotation must be resolved before training: {record['image_id']}")
            image_id = record["image_id"]
            if image_id in seen_ids:
                raise ValueError(f"Duplicate image_id across category JSONL files: {image_id}")
            seen_ids.add(image_id)
            if record.get("category") != expected_category:
                raise ValueError(f"Unexpected label for {image_id}: {record.get('category')!r}")

            source_path = Path(record.get("source_path", ""))
            crop_path = Path(record.get("subject_crop_path", ""))
            update = record.get("description_update") or {}
            subject_caption = (update.get("subject_crop_description") or "").strip()
            scene_caption = (
                update.get("original_image_description")
                or record.get("main_subject_description_before_update")
                or ""
            ).strip()
            if not source_path.is_file() or not crop_path.is_file():
                raise FileNotFoundError(f"Missing original/crop for {image_id}: {source_path} | {crop_path}")
            if not subject_caption or not scene_caption:
                raise ValueError(f"Missing verified subject/scene caption for {image_id}")
            subject_box = record.get("primary_subject_box_xywh_norm")
            subject_label = (record.get("primary_subject_label") or "").strip()
            if not subject_label or not isinstance(subject_box, list) or len(subject_box) != 4:
                raise ValueError(f"Missing primary subject label/box for {image_id}")
            if any(not isinstance(value, (int, float)) for value in subject_box):
                raise ValueError(f"Non-numeric primary subject box for {image_id}: {subject_box}")
            x, y, width, height = map(float, subject_box)
            if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1.00001 or y + height > 1.00001:
                raise ValueError(f"Out-of-range primary subject box for {image_id}: {subject_box}")

            rows.append({
                "image_id": image_id,
                "category": expected_category,
                "source_path": str(source_path.resolve()),
                "crop_path": str(crop_path.resolve()),
                "sha256": _sha256(source_path),
                "dhash": _difference_hash(source_path),
                "subject_caption": subject_caption,
                "scene_caption": scene_caption,
                "primary_subject_label": subject_label,
                "primary_subject_box_xywh_norm": [x, y, width, height],
                "human_reviewed": bool(review),
            })
            category_counts[expected_category] += 1

    group_count = _split_groups(rows, seed, hash_distance)
    split_counts = Counter(row["split"] for row in rows)
    category_split_counts = {
        category: dict(Counter(row["split"] for row in rows if row["category"] == category))
        for category in CATEGORIES.values()
    }
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_splits[row["group_id"]].add(row["split"])
    leakage = [group_id for group_id, splits in group_splits.items() if len(splits) != 1]
    if leakage:
        raise RuntimeError(f"Near-duplicate leakage across splits: {leakage[:5]}")

    summary = {
        "manifest": str(manifest_path.resolve()),
        "photos": len(rows),
        "caption_pairs": len(rows) * 2,
        "groups": group_count,
        "category_counts": dict(category_counts),
        "split_counts": dict(split_counts),
        "category_split_counts": category_split_counts,
        "near_duplicate_distance": hash_distance,
        "group_split_leaks": len(leakage),
        "dry_run": dry_run,
    }
    if not dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{manifest_path.name}.", suffix=".tmp", dir=manifest_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                for row in sorted(rows, key=lambda item: item["image_id"]):
                    row["dhash"] = f"{row['dhash']:016x}"
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, manifest_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return summary


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = _load_jsonl(path)
    required = {"image_id", "category", "source_path", "crop_path", "split", "group_id", "subject_caption", "scene_caption"}
    if not rows or any(not required.issubset(row) for row in rows):
        raise ValueError(f"Invalid or empty subject caption manifest: {path}")
    seen = set()
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["image_id"] in seen:
            raise ValueError(f"Duplicate image_id in manifest: {row['image_id']}")
        seen.add(row["image_id"])
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid split for {row['image_id']}")
        if not Path(row["source_path"]).is_file() or not Path(row["crop_path"]).is_file():
            raise FileNotFoundError(f"Image/crop missing from manifest for {row['image_id']}")
        group_splits[row["group_id"]].add(row["split"])
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise ValueError("Near-duplicate group crosses dataset splits")
    return rows


def make_examples(rows: list[dict[str, Any]], split: str) -> list[dict[str, str]]:
    examples = []
    for row in rows:
        if row["split"] != split:
            continue
        examples.append({"image_path": row["source_path"], "caption": row["scene_caption"], "kind": "scene", "category": row["category"], "image_id": row["image_id"]})
        examples.append({"image_path": row["crop_path"], "caption": row["subject_caption"], "kind": "subject", "category": row["category"], "image_id": row["image_id"]})
    return examples


def _sample_quotas(rows: list[dict[str, Any]], photos_per_mini_epoch: int) -> dict[str, int]:
    counts = Counter(row["category"] for row in rows)
    total = sum(counts.values())
    exact = {category: photos_per_mini_epoch * counts[category] / total for category in CATEGORIES.values()}
    quotas = {category: int(exact[category]) for category in CATEGORIES.values()}
    remainder = photos_per_mini_epoch - sum(quotas.values())
    order = sorted(CATEGORIES.values(), key=lambda category: exact[category] - quotas[category], reverse=True)
    for category in order[:remainder]:
        quotas[category] += 1
    if any(quotas[category] > counts[category] for category in quotas):
        raise ValueError("A mini-epoch requests more unique photos from a category than the training split contains")
    return quotas


def _sample_mini_epoch(
    rows: list[dict[str, Any]],
    quotas: dict[str, int],
    mini_epoch: int,
    seed: int,
) -> list[dict[str, str]]:
    selected_rows = []
    for category_index, category in enumerate(CATEGORIES.values()):
        category_rows = sorted((row for row in rows if row["category"] == category), key=lambda row: row["image_id"])
        random.Random(seed + category_index * 1009).shuffle(category_rows)
        quota = quotas[category]
        start = (mini_epoch - 1) * quota
        selected_rows.extend(category_rows[index % len(category_rows)] for index in range(start, start + quota))
    pairs = []
    for row in selected_rows:
        pairs.extend((
            {"image_path": row["source_path"], "caption": row["scene_caption"], "kind": "scene", "category": row["category"], "image_id": row["image_id"]},
            {"image_path": row["crop_path"], "caption": row["subject_caption"], "kind": "subject", "category": row["category"], "image_id": row["image_id"]},
        ))
    random.Random(seed + mini_epoch * 7919).shuffle(pairs)
    return pairs


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _save_model_checkpoint(model: Any, processor: Any, checkpoint_dir: Path) -> None:
    """Atomically refresh a weight-only checkpoint, leaving the previous file intact on interruption."""
    import shutil

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if not (checkpoint_dir / "config.json").exists():
        model.config.save_pretrained(checkpoint_dir)
    if not (checkpoint_dir / "processor_config.json").exists():
        processor.save_pretrained(checkpoint_dir)
    for module_name in ("modeling_florence2.py", "configuration_florence2.py"):
        source_module = MODEL_ROOT / module_name
        if source_module.is_file() and not (checkpoint_dir / module_name).exists():
            shutil.copy2(source_module, checkpoint_dir / module_name)
    with tempfile.TemporaryDirectory(prefix="florence_checkpoint_", dir=checkpoint_dir.parent) as temporary:
        temporary_dir = Path(temporary)
        model.save_pretrained(temporary_dir, safe_serialization=True)
        generated = temporary_dir / "model.safetensors"
        os.replace(generated, checkpoint_dir / "model.safetensors")
        for path in temporary_dir.iterdir():
            if path.name != "model.safetensors" and path.is_file() and not (checkpoint_dir / path.name).exists():
                shutil.copy2(path, checkpoint_dir / path.name)


def _load_weights(model: Any, checkpoint_dir: Path) -> tuple[list[str], list[str]]:
    from safetensors.torch import load_model

    weights = checkpoint_dir / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"Missing model weights in initialization checkpoint: {weights}")
    missing, unexpected = load_model(model, weights, strict=False, device="cpu")
    if unexpected:
        raise RuntimeError(f"Checkpoint contains unexpected Florence-2 tensors: {unexpected[:10]}")
    return sorted(missing), sorted(unexpected)


def _train(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable. Training on CPU is disabled by default; pass --allow-cpu to opt in.")
    if not args.init_model.is_dir():
        raise FileNotFoundError(f"Initialization checkpoint is missing: {args.init_model}")
    rows = load_manifest(args.manifest)
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    if not train_rows or not validation_rows:
        raise ValueError("Manifest must contain train and validation samples")
    if args.steps_per_mini_epoch * args.batch_size % (2 * len(CATEGORIES)) != 0:
        raise ValueError("steps-per-mini-epoch × batch-size must be divisible by 2 × category-count")
    photos_per_mini_epoch = args.steps_per_mini_epoch * args.batch_size // 2
    train_quotas = _sample_quotas(train_rows, photos_per_mini_epoch)
    validation_quotas = {category: args.validation_photos_per_category for category in CATEGORIES.values()}
    sample_validation_examples = _sample_mini_epoch(
        validation_rows, validation_quotas, mini_epoch=1, seed=args.seed + 900_001
    )
    full_validation_examples = make_examples(rows, "validation")
    if args.limit_final_validation_examples:
        full_validation_examples = full_validation_examples[:args.limit_final_validation_examples]

    output = args.output_dir.resolve()
    resume_state_path = output / "latest_checkpoint" / "training_state.json"
    if args.resume:
        if not resume_state_path.is_file():
            raise FileNotFoundError(f"No resumable mini-epoch checkpoint found: {resume_state_path}")
        init_model = output / "latest_checkpoint"
    else:
        init_model = args.init_model.resolve()
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Output directory is not empty; choose another path or pass --resume: {output}")

    # Load architecture/code from the original local model, then overlay checkpoint
    # weights. Older checkpoints may not contain modeling_florence2.py themselves.
    processor = AutoProcessor.from_pretrained(str(MODEL_ROOT), trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_ROOT), trust_remote_code=True, local_files_only=True)
    if init_model.resolve() != MODEL_ROOT.resolve():
        missing, unexpected = _load_weights(model, init_model.resolve())
        if missing:
            print(json.dumps({"warning": "checkpoint_missing_tensors", "count": len(missing), "sample": missing[:10]}), flush=True)
    if args.freeze_vision:
        vision_tower = getattr(model, "vision_tower", None)
        if vision_tower is None and getattr(model, "model", None) is not None:
            vision_tower = getattr(model.model, "vision_tower", None)
        if vision_tower is None:
            raise RuntimeError("Could not locate Florence-2 vision_tower for --freeze-vision")
        for parameter in vision_tower.parameters():
            parameter.requires_grad_(False)
    model.to(device)
    model.train()

    def collate(examples: list[dict[str, str]]) -> dict[str, Any]:
        images = []
        prompts = []
        targets = []
        for example in examples:
            with Image.open(example["image_path"]) as image:
                images.append(image.convert("RGB"))
            prompts.append(PROMPT)
            targets.append(example["caption"])
        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
        target_tokens = processor.tokenizer(
            targets,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_target_tokens,
            add_special_tokens=True,
        )["input_ids"]
        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            target_tokens[target_tokens == pad_token_id] = -100
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
            "pixel_values": inputs["pixel_values"],
            "labels": target_tokens,
        }

    output.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if args.resume:
        state = json.loads(resume_state_path.read_text(encoding="utf-8"))
        history_path = output / "training_history.json"
        history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
        completed_mini_epochs = int(state["completed_mini_epochs"])
        global_step = int(state["global_step"])
        best_validation = float(state.get("best_sample_validation_loss", float("inf")))
        best_mini_epoch = int(state.get("best_mini_epoch", 0))
        # Optimizer state is intentionally restarted; checkpoints are at mini-epoch boundaries.
    else:
        history = []
        completed_mini_epochs = 0
        global_step = 0
        best_validation = float("inf")
        best_mini_epoch = 0

    started_at = time.time()

    def evaluate(examples: list[dict[str, str]]) -> float:
        model.eval()
        losses = []
        with torch.inference_mode():
            for start in range(0, len(examples), args.batch_size):
                batch = examples[start:start + args.batch_size]
                tensors = {key: value.to(device) for key, value in collate(batch).items() if value is not None}
                losses.append(float(model(**tensors).loss.cpu()))
        model.train()
        return sum(losses) / max(1, len(losses))

    for mini_epoch in range(completed_mini_epochs + 1, args.mini_epochs + 1):
        mini_examples = _sample_mini_epoch(train_rows, train_quotas, mini_epoch, args.seed)
        random.Random(args.seed + mini_epoch * 104729).shuffle(mini_examples)
        model.train()
        train_losses = []
        mini_started_at = time.time()
        batch_count = (len(mini_examples) + args.batch_size - 1) // args.batch_size
        for batch_index, start in enumerate(range(0, len(mini_examples), args.batch_size), 1):
            batch = mini_examples[start:start + args.batch_size]
            tensors = {key: value.to(device) for key, value in collate(batch).items() if value is not None}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**tensors).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at mini-epoch {mini_epoch}, step {global_step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            global_step += 1
            if batch_index % args.log_every_steps == 0 or batch_index == batch_count:
                progress = {
                    "mini_epoch": mini_epoch,
                    "mini_epochs_total": args.mini_epochs,
                    "batch": batch_index,
                    "batches_in_mini_epoch": batch_count,
                    "global_step": global_step,
                    "train_loss_recent": sum(train_losses[-args.log_every_steps:]) / min(len(train_losses), args.log_every_steps),
                    "elapsed_seconds": round(time.time() - started_at, 1),
                }
                print(json.dumps(progress, ensure_ascii=False), flush=True)
                _write_json_atomic(output / "progress.json", progress)
            if args.max_steps and global_step >= args.max_steps:
                break

        sample_validation_loss = evaluate(sample_validation_examples)
        epoch_result = {
            "mini_epoch": mini_epoch,
            "train_examples": len(train_losses),
            "train_loss": sum(train_losses) / max(1, len(train_losses)),
            "sample_validation_examples": len(sample_validation_examples),
            "sample_validation_loss": sample_validation_loss,
            "elapsed_seconds": round(time.time() - mini_started_at, 1),
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result, ensure_ascii=False), flush=True)
        _write_json_atomic(output / "training_history.json", history)

        if sample_validation_loss < best_validation:
            best_validation = sample_validation_loss
            best_mini_epoch = mini_epoch
            _save_model_checkpoint(model, processor, output / "best_model")
        _save_model_checkpoint(model, processor, output / "latest_checkpoint")
        completed_mini_epochs = mini_epoch
        state = {
            "completed_mini_epochs": completed_mini_epochs,
            "global_step": global_step,
            "best_sample_validation_loss": best_validation,
            "best_mini_epoch": best_mini_epoch,
            "mini_epochs_total": args.mini_epochs,
            "steps_per_mini_epoch": args.steps_per_mini_epoch,
            "optimizer_state_saved": False,
        }
        _write_json_atomic(output / "latest_checkpoint" / "training_state.json", state)
        _write_json_atomic(output / "training_state.json", state)
        if args.max_steps and global_step >= args.max_steps:
            break

    final_validation_loss = None
    if not args.skip_final_full_validation:
        final_validation_loss = evaluate(full_validation_examples)
        print(json.dumps({"final_full_validation_loss": final_validation_loss, "examples": len(full_validation_examples)}, ensure_ascii=False), flush=True)

    result = {
        "model_base": str(MODEL_ROOT),
        "initialized_from": str(init_model.resolve()),
        "manifest": str(args.manifest.resolve()),
        "device": device,
        "train_photo_count": sum(row["split"] == "train" for row in rows),
        "validation_photo_count": sum(row["split"] == "validation" for row in rows),
        "test_photo_count": sum(row["split"] == "test" for row in rows),
        "mini_epochs_requested": args.mini_epochs,
        "mini_epochs_completed": completed_mini_epochs,
        "steps_per_mini_epoch": args.steps_per_mini_epoch,
        "global_step": global_step,
        "sampled_training_photo_count": sum(
            min(sum(row["category"] == category for row in train_rows), train_quotas[category] * completed_mini_epochs)
            for category in CATEGORIES.values()
        ),
        "best_mini_epoch": best_mini_epoch,
        "best_sample_validation_loss": best_validation,
        "final_full_validation_loss": final_validation_loss,
        "best_model_path": str((output / "best_model").resolve()),
        "latest_checkpoint_path": str((output / "latest_checkpoint").resolve()),
        "optimizer_state_saved": False,
        "test_split_used_for_optimization": False,
        "freeze_vision": args.freeze_vision,
    }
    _write_json_atomic(output / "training_summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--init-model", type=Path, default=DEFAULT_INIT_MODEL, help="local model/checkpoint to initialize from")
    parser.add_argument("--prepare-manifest", action="store_true", help="build a split-safe manifest from category JSONL files")
    parser.add_argument("--dry-run", action="store_true", help="validate/summarize manifest preparation without writing")
    parser.add_argument("--plan-only", action="store_true", help="validate and print the sampled training plan without loading a model")
    parser.add_argument("--train", action="store_true", help="start fine-tuning; omitted by default")
    parser.add_argument("--resume", action="store_true", help="resume model weights from this output directory's latest mini-epoch checkpoint")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--hash-distance", type=int, default=4, help="dHash distance threshold for grouping near-duplicates")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mini-epochs", type=int, default=5, help="number of fixed-size sampled rounds")
    parser.add_argument("--steps-per-mini-epoch", type=int, default=200, help="optimizer batches per sampled round; not a full data pass")
    parser.add_argument("--validation-photos-per-category", type=int, default=8)
    parser.add_argument("--log-every-steps", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-target-tokens", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=0, help="optional total optimizer-step cap for a short trial")
    parser.add_argument("--limit-final-validation-examples", type=int, default=0, help="optional smoke-test cap; otherwise validate the full validation split once at the end")
    parser.add_argument("--skip-final-full-validation", action="store_true", help="skip full validation; intended only for smoke tests")
    parser.add_argument("--allow-cpu", action="store_true", help="explicitly allow slow CPU fine-tuning")
    parser.add_argument("--train-vision", dest="freeze_vision", action="store_false", help="also train the Florence-2 vision tower")
    parser.add_argument("--freeze-vision", dest="freeze_vision", action="store_true", help="freeze vision tower (default)")
    parser.set_defaults(freeze_vision=True)
    args = parser.parse_args()

    if args.prepare_manifest:
        summary = prepare_manifest(
            args.data_root.resolve(), args.manifest.resolve(), seed=args.seed,
            hash_distance=args.hash_distance, dry_run=args.dry_run,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.plan_only:
        if not args.manifest.is_file():
            parser.error(f"Manifest not found: {args.manifest}; run --prepare-manifest first")
        rows = load_manifest(args.manifest)
        train_rows = [row for row in rows if row["split"] == "train"]
        if args.batch_size < 1 or args.steps_per_mini_epoch < 1:
            parser.error("batch size and steps per mini-epoch must be positive")
        examples_per_mini = args.steps_per_mini_epoch * args.batch_size
        if examples_per_mini % (2 * len(CATEGORIES)):
            parser.error("steps-per-mini-epoch × batch-size must be divisible by 2 × category-count")
        quotas = _sample_quotas(train_rows, examples_per_mini // 2)
        plan = {
            "photos": len(rows),
            "train_photos": len(train_rows),
            "category_photo_quotas_per_mini_epoch": quotas,
            "steps_per_mini_epoch": args.steps_per_mini_epoch,
            "mini_epochs": args.mini_epochs,
            "total_optimizer_steps": args.steps_per_mini_epoch * args.mini_epochs,
            "examples_per_photo": 2,
            "init_model": str(args.init_model.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "resume_optimizer_state": False,
        }
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.train:
        if args.dry_run or args.plan_only:
            parser.error("--dry-run/--plan-only cannot be combined with --train")
        if not args.manifest.is_file():
            parser.error(f"Manifest not found: {args.manifest}; run --prepare-manifest first")
        if args.batch_size < 1 or args.mini_epochs < 1 or args.steps_per_mini_epoch < 1 or args.validation_photos_per_category < 1 or args.log_every_steps < 1:
            parser.error("batch size, mini-epochs, steps per mini-epoch, validation sample size, and log interval must be positive")
        if args.max_steps < 0 or args.limit_final_validation_examples < 0:
            parser.error("step and validation limits cannot be negative")
        summary = _train(args)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not args.prepare_manifest and not args.plan_only and not args.train:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
