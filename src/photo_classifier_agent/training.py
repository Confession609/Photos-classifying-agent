"""Manifest-bound OpenCLIP linear probing for the four project classes."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .categories import CATEGORY_NAMES
from .classifier import image_id_for
from .decision import decide
from .evaluation import evaluate, write_evaluation
from .reports import write_decisions
from .schemas import CandidateScore, ClassificationDecision, Prediction


def read_training_manifest(path: str | Path) -> tuple[list[dict[str, Any]], str]:
    """Read and validate the sole source of sample membership and labels."""
    manifest_path = Path(path).expanduser().resolve()
    raw = manifest_path.read_bytes()
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_hashes: dict[str, str] = {}
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid manifest JSON on line {line_number}: {exc}") from exc
        required = {"image_id", "path", "category", "content_hash", "split"}
        missing = required - record.keys()
        if missing:
            raise ValueError(f"manifest line {line_number} is missing fields: {sorted(missing)}")
        if record["category"] not in CATEGORY_NAMES:
            raise ValueError(f"unknown category on manifest line {line_number}: {record['category']!r}")
        if record["split"] not in {"train", "validation", "test"}:
            raise ValueError(f"unknown split on manifest line {line_number}: {record['split']!r}")
        if record["image_id"] in seen_ids:
            raise ValueError(f"duplicate image_id in manifest: {record['image_id']}")
        seen_ids.add(record["image_id"])
        if image_id_for(record["path"]) != record["image_id"]:
            raise ValueError(f"image_id does not match manifest path on line {line_number}")
        sample_path = Path(record["path"])
        if not sample_path.is_file():
            raise FileNotFoundError(f"manifest image does not exist: {sample_path}")
        content_hash = record["content_hash"]
        if content_hash in seen_hashes:
            raise ValueError(
                "duplicate content_hash in manifest could leak across splits: "
                f"{seen_hashes[content_hash]} and {record['image_id']}"
            )
        seen_hashes[content_hash] = record["image_id"]
        records.append(record)

    if not records:
        raise ValueError("manifest contains no samples")
    for split in ("train", "validation", "test"):
        if not any(record["split"] == split for record in records):
            raise ValueError(f"manifest has no {split} samples")
    return records, hashlib.sha256(raw).hexdigest()


def _metrics_from_logits(logits, records: list[dict[str, Any]], class_names: tuple[str, ...]) -> dict[str, Any]:
    predicted = logits.argmax(dim=1).tolist()
    truth = [class_names.index(record["category"]) for record in records]
    matrix = [[0 for _ in class_names] for _ in class_names]
    for actual, guess in zip(truth, predicted, strict=True):
        matrix[actual][guess] += 1
    recalls = []
    f1s = []
    for index in range(len(class_names)):
        tp = matrix[index][index]
        fp = sum(matrix[row][index] for row in range(len(class_names)) if row != index)
        fn = sum(matrix[index][column] for column in range(len(class_names)) if column != index)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        recalls.append(recall)
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    correct = sum(matrix[index][index] for index in range(len(class_names)))
    return {
        "accuracy": correct / len(records),
        "macro_f1": sum(f1s) / len(f1s),
        "per_class_recall": dict(zip(class_names, recalls, strict=True)),
        "confusion_matrix": {
            actual: dict(zip(class_names, row, strict=True))
            for actual, row in zip(class_names, matrix, strict=True)
        },
    }


def train_linear_probe(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: str | None = None,
    batch_size: int = 16,
    epochs: int = 60,
    patience: int = 10,
    learning_rate: float = 0.01,
    seed: int = 1337,
) -> dict[str, Any]:
    """Train a linear head on frozen pretrained CLIP features.

    Image features are extracted only for manifest ``train`` and ``validation``
    rows before model selection. Test images are opened only after the best
    validation checkpoint has been selected.
    """
    import open_clip
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    if batch_size < 1 or epochs < 1 or patience < 1:
        raise ValueError("batch_size, epochs, and patience must be positive")
    records, manifest_sha256 = read_training_manifest(manifest_path)
    train_records = [record for record in records if record["split"] == "train"]
    validation_records = [record for record in records if record["split"] == "validation"]
    test_records = [record for record in records if record["split"] == "test"]
    class_names = tuple(CATEGORY_NAMES)
    class_to_index = {name: index for index, name in enumerate(class_names)}

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    chosen_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).resolve().parents[2]
    project_cache = project_root / "algorithms" / "models" / "huggingface"
    import os
    os.environ.setdefault("HF_HOME", str(project_cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(project_cache / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(project_cache / "transformers"))
    os.environ.setdefault("HF_MODULES_CACHE", str(project_cache / "modules"))

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=chosen_device
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    class ManifestImages(Dataset):
        def __init__(self, items: list[dict[str, Any]]):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            record = self.items[index]
            with Image.open(record["path"]) as image:
                tensor = preprocess(image.convert("RGB"))
            return tensor, class_to_index[record["category"]], index

    def extract_features(items: list[dict[str, Any]], split_name: str):
        loader = DataLoader(ManifestImages(items), batch_size=batch_size, shuffle=False, num_workers=0)
        outputs = []
        labels = []
        model.eval()
        with torch.inference_mode():
            for batch_index, (images, batch_labels, _) in enumerate(loader, 1):
                embeddings = model.encode_image(images.to(chosen_device))
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                outputs.append(embeddings.float().cpu())
                labels.append(batch_labels.long())
                processed = min(batch_index * batch_size, len(items))
                print(f"features {split_name}: {processed}/{len(items)}", flush=True)
        return torch.cat(outputs), torch.cat(labels)

    print(f"Manifest SHA-256: {manifest_sha256}", flush=True)
    print(
        "Manifest split counts: "
        + json.dumps({"train": len(train_records), "validation": len(validation_records), "test": len(test_records)}),
        flush=True,
    )
    train_x, train_y = extract_features(train_records, "train")
    validation_x, validation_y = extract_features(validation_records, "validation")

    head = torch.nn.Linear(train_x.shape[1], len(class_names)).to(chosen_device)
    counts = Counter(record["category"] for record in train_records)
    class_weights = torch.tensor(
        [len(train_records) / (len(class_names) * counts[name]) for name in class_names],
        dtype=torch.float32,
        device=chosen_device,
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=0.01)
    train_x, train_y = train_x.to(chosen_device), train_y.to(chosen_device)
    validation_x, validation_y = validation_x.to(chosen_device), validation_y.to(chosen_device)
    generator = torch.Generator().manual_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_epoch = 0
    best_state = None
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        head.train()
        permutation = torch.randperm(len(train_x), generator=generator)
        loss_total = 0.0
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start:start + batch_size].to(chosen_device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(train_x[indices])
            loss = criterion(logits, train_y[indices])
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * len(indices)

        head.eval()
        with torch.inference_mode():
            val_logits = head(validation_x)
        epoch_metrics = _metrics_from_logits(val_logits, validation_records, class_names)
        epoch_record = {
            "epoch": epoch,
            "train_loss": loss_total / len(train_x),
            **epoch_metrics,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, ensure_ascii=True), flush=True)
        if epoch_metrics["macro_f1"] > best_f1:
            best_f1 = epoch_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            print(f"Early stopping at epoch {epoch}; best validation epoch was {best_epoch}.", flush=True)
            break

    if best_state is None:
        raise RuntimeError("training failed to produce a validation checkpoint")
    head.load_state_dict(best_state)
    checkpoint_path = output_path / "openclip_linear_probe.pt"
    checkpoint = {
        "head_state_dict": best_state,
        "class_names": class_names,
        "input_dim": int(train_x.shape[1]),
        "model_name": model_name,
        "pretrained": pretrained,
        "manifest_sha256": manifest_sha256,
        "best_epoch": best_epoch,
        "validation_macro_f1": best_f1,
        "seed": seed,
        "training_method": "frozen OpenCLIP image encoder + weighted linear probe",
    }
    torch.save(checkpoint, checkpoint_path)
    (output_path / "training_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Final test evaluation only after validation-based checkpoint selection.
    test_x, _ = extract_features(test_records, "test")
    head.eval()
    with torch.inference_mode():
        test_logits = head(test_x.to(chosen_device)).cpu()
        test_probabilities = torch.softmax(test_logits, dim=1)
    decisions: list[ClassificationDecision] = []
    for record, probabilities in zip(test_records, test_probabilities.tolist(), strict=True):
        prediction = Prediction(
            image_id=record["image_id"],
            source_path=record["path"],
            scores=tuple(
                CandidateScore(name, float(score))
                for name, score in zip(class_names, probabilities, strict=True)
            ),
            model_name=f"open_clip_linear_probe:{model_name}",
            model_version=f"{model_name}:{pretrained}:epoch-{best_epoch}",
        )
        decisions.append(decide(prediction))
    decisions_path = output_path / "test_decisions.jsonl"
    write_decisions(decisions, decisions_path)
    truth = {record["image_id"]: record["category"] for record in test_records}
    result = evaluate(decisions, truth)
    metrics_path = write_evaluation(result, output_path / "test_metrics.json")
    summary = {
        "manifest_sha256": manifest_sha256,
        "split_counts": {"train": len(train_records), "validation": len(validation_records), "test": len(test_records)},
        "class_counts_train": dict(counts),
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        "test_metrics": result.to_dict(),
        "checkpoint": str(checkpoint_path),
        "test_decisions": str(decisions_path),
        "test_metrics_path": str(metrics_path),
    }
    (output_path / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
