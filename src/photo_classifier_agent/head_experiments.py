"""Validation-only comparison of classifier heads for the third hierarchy node."""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .hierarchy import HIERARCHY_NODES, target_for
from .training import _metrics_from_logits, read_training_manifest


def compare_third_layer_heads(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: str | None = None,
    batch_size: int = 16,
    epochs: int = 80,
    patience: int = 12,
    learning_rate: float = 0.01,
    seed: int = 1337,
) -> dict[str, Any]:
    """Compare five heads using only third-node train/validation images.

    Test records are validated as part of the manifest but their images and
    labels are never used for feature extraction, model selection, or ranking.
    """
    import open_clip
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    if batch_size < 1 or epochs < 1 or patience < 1:
        raise ValueError("batch_size, epochs, and patience must be positive")

    records, manifest_sha256 = read_training_manifest(manifest_path)
    node = next(node for node in HIERARCHY_NODES if node.name == "still_vs_landscape")
    class_names = (node.positive_label, node.negative_label)
    train_records = [
        row | {"category": target_for(node, row["category"])}
        for row in records
        if row["split"] == "train" and target_for(node, row["category"]) is not None
    ]
    validation_records = [
        row | {"category": target_for(node, row["category"])}
        for row in records
        if row["split"] == "validation" and target_for(node, row["category"]) is not None
    ]
    if not train_records or not validation_records:
        raise ValueError("third-layer train and validation samples are required")
    for split_name, rows in (("train", train_records), ("validation", validation_records)):
        counts = Counter(row["category"] for row in rows)
        if any(counts[label] == 0 for label in class_names):
            raise ValueError(f"third-layer {split_name} split must contain both classes")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    chosen_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).resolve().parents[2]
    project_cache = project_root / "algorithms" / "models" / "huggingface"
    os.environ.setdefault("HF_HOME", str(project_cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(project_cache / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(project_cache / "transformers"))
    os.environ.setdefault("HF_MODULES_CACHE", str(project_cache / "modules"))
    backbone, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=chosen_device
    )
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)

    class ManifestImages(Dataset):
        def __init__(self, rows: list[dict[str, Any]]):
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int):
            with Image.open(self.rows[index]["path"]) as image:
                return preprocess(image.convert("RGB"))

    def extract(rows: list[dict[str, Any]], split_name: str):
        loader = DataLoader(
            ManifestImages(rows), batch_size=batch_size, shuffle=False, num_workers=0
        )
        features = []
        with torch.inference_mode():
            for batch_index, images in enumerate(loader, 1):
                vectors = backbone.encode_image(images.to(chosen_device))
                vectors = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                features.append(vectors.float().cpu())
                processed = min(batch_index * batch_size, len(rows))
                print(f"third-layer features {split_name}: {processed}/{len(rows)}", flush=True)
        return torch.cat(features, dim=0)

    print(f"Manifest SHA-256: {manifest_sha256}", flush=True)
    print(
        "Extracting only third-layer train/validation images; test images remain unopened.",
        flush=True,
    )
    train_x = extract(train_records, "train")
    validation_x = extract(validation_records, "validation")
    train_y = torch.tensor(
        [class_names.index(row["category"]) for row in train_records], dtype=torch.long
    )
    validation_y = torch.tensor(
        [class_names.index(row["category"]) for row in validation_records], dtype=torch.long
    )
    train_x = train_x.to(chosen_device)
    validation_x = validation_x.to(chosen_device)
    train_y_device = train_y.to(chosen_device)
    validation_y_device = validation_y.to(chosen_device)
    train_counts = Counter(train_y.tolist())
    class_weights = torch.tensor(
        [len(train_y) / (len(class_names) * train_counts[i]) for i in range(len(class_names))],
        dtype=torch.float32,
        device=chosen_device,
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    experiments: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, Any]]] = {}
    validation_logits: dict[str, torch.Tensor] = {}

    def metric_rows(logits: torch.Tensor) -> dict[str, Any]:
        return _metrics_from_logits(logits.detach().cpu(), validation_records, class_names)

    def fit_torch_head(name: str, head: torch.nn.Module, *, hinge: bool = False):
        head = head.to(chosen_device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=0.01)
        generator = torch.Generator().manual_seed(seed + len(experiments))
        best_f1 = -1.0
        best_state = None
        best_logits = None
        best_metrics = None
        best_epoch = 0
        stale_epochs = 0
        history = []
        for epoch in range(1, epochs + 1):
            head.train()
            permutation = torch.randperm(len(train_x), generator=generator)
            loss_total = 0.0
            for start in range(0, len(permutation), 64):
                indexes = permutation[start : start + 64].to(chosen_device)
                optimizer.zero_grad(set_to_none=True)
                if hinge:
                    scores = head(train_x[indexes]).squeeze(-1)
                    signed = torch.where(
                        train_y_device[indexes] == 0,
                        torch.ones_like(scores),
                        -torch.ones_like(scores),
                    )
                    losses = torch.relu(1.0 - signed * scores).square()
                    loss = (losses * class_weights[train_y_device[indexes]]).mean()
                    logits = torch.stack((scores, -scores), dim=1)
                else:
                    logits = head(train_x[indexes])
                    loss = torch.nn.functional.cross_entropy(
                        logits, train_y_device[indexes], weight=class_weights
                    )
                loss.backward()
                optimizer.step()
                loss_total += float(loss.detach()) * len(indexes)
            head.eval()
            with torch.inference_mode():
                logits = head(validation_x)
                if hinge:
                    scores = logits.squeeze(-1)
                    logits = torch.stack((scores, -scores), dim=1)
                metrics = metric_rows(logits)
            row = {"epoch": epoch, "train_loss": loss_total / len(train_x), **metrics}
            history.append(row)
            if metrics["macro_f1"] > best_f1:
                best_f1 = metrics["macro_f1"]
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
                best_logits = logits.detach().cpu().clone()
                best_metrics = metrics
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= patience:
                break
        if best_state is None or best_logits is None or best_metrics is None:
            raise RuntimeError(f"head {name} failed to produce a validation checkpoint")
        histories[name] = history
        validation_logits[name] = best_logits
        checkpoint = {
            "method": name,
            "state_dict": best_state,
            "input_dim": int(train_x.shape[1]),
            "class_names": class_names,
            "best_epoch": best_epoch,
        }
        torch.save(checkpoint, output / f"{name}.pt")
        experiments.append({"name": name, "best_epoch": best_epoch, **best_metrics})

    linear = torch.nn.Linear(train_x.shape[1], len(class_names))
    fit_torch_head("linear_logistic", linear)

    svm = torch.nn.Linear(train_x.shape[1], 1)
    fit_torch_head("linear_svm_squared_hinge", svm, hinge=True)

    mlp = torch.nn.Sequential(
        torch.nn.Linear(train_x.shape[1], 256),
        torch.nn.GELU(),
        torch.nn.Dropout(0.30),
        torch.nn.Linear(256, len(class_names)),
    )
    fit_torch_head("mlp_256_dropout", mlp)

    # Binary RBF kernel ridge classifier: static life is the positive score.
    train_cpu = train_x.detach().cpu()
    val_cpu = validation_x.detach().cpu()
    squared_distances = (2.0 - 2.0 * (train_cpu @ train_cpu.T)).clamp_min(0.0)
    gamma = 4.0
    regularization = 0.10
    kernel = torch.exp(-gamma * squared_distances)
    binary_targets = torch.where(train_y == 0, 1.0, -1.0)
    sample_weights = torch.tensor(
        [len(train_y) / (2 * train_counts[int(label)]) for label in train_y.tolist()],
        dtype=torch.float32,
    )
    system = kernel + torch.diag(regularization / sample_weights)
    alpha = torch.linalg.solve(system, binary_targets)
    val_distances = (2.0 - 2.0 * (val_cpu @ train_cpu.T)).clamp_min(0.0)
    val_kernel = torch.exp(-gamma * val_distances)
    rbf_scores = val_kernel @ alpha
    rbf_logits = torch.stack((rbf_scores, -rbf_scores), dim=1)
    rbf_metrics = metric_rows(rbf_logits)
    validation_logits["rbf_kernel_ridge"] = rbf_logits
    torch.save(
        {
            "method": "rbf_kernel_ridge",
            "train_features": train_cpu,
            "alpha": alpha,
            "gamma": gamma,
            "regularization": regularization,
            "class_names": class_names,
            "manifest_sha256": manifest_sha256,
        },
        output / "rbf_kernel_ridge.pt",
    )
    experiments.append({"name": "rbf_kernel_ridge", **rbf_metrics})

    # Cosine 7-nearest-neighbour head over the same frozen train embeddings.
    k = min(7, len(train_y))
    similarities = val_cpu @ train_cpu.T
    neighbor_scores, neighbor_indices = similarities.topk(k, dim=1)
    neighbor_weights = torch.softmax(neighbor_scores * 20.0, dim=1)
    neighbor_labels = train_y[neighbor_indices]
    knn_logits = torch.zeros((len(validation_records), len(class_names)), dtype=torch.float32)
    knn_logits.scatter_add_(1, neighbor_labels, neighbor_weights)
    knn_metrics = metric_rows(knn_logits)
    validation_logits["cosine_knn_7"] = knn_logits
    torch.save(
        {
            "method": "cosine_knn_7",
            "train_features": train_cpu,
            "train_labels": train_y,
            "k": k,
            "temperature": 20.0,
            "class_names": class_names,
            "manifest_sha256": manifest_sha256,
        },
        output / "cosine_knn_7.pt",
    )
    experiments.append({"name": "cosine_knn_7", **knn_metrics})

    ranking = sorted(
        experiments,
        key=lambda row: (row["macro_f1"], row["accuracy"]),
        reverse=True,
    )
    for index, row in enumerate(ranking, 1):
        row["rank"] = index
    prediction_path = output / "validation_predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(validation_records):
            row = {
                "image_id": record["image_id"],
                "source_path": record["path"],
                "ground_truth": record["category"],
                "split": "validation",
                "predictions": {
                    name: class_names[int(logits[index].argmax())]
                    for name, logits in validation_logits.items()
                },
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    result = {
        "manifest_sha256": manifest_sha256,
        "node": node.name,
        "class_names": class_names,
        "train_samples": len(train_records),
        "validation_samples": len(validation_records),
        "test_images_opened": False,
        "model_name": model_name,
        "pretrained": pretrained,
        "device": chosen_device,
        "selection_metric": "validation macro_f1 (accuracy as tie-breaker)",
        "ranking": ranking,
        "training_history": histories,
        "validation_predictions": str(prediction_path),
        "checkpoints": [str(output / f"{row['name']}.pt") for row in experiments],
    }
    (output / "ranking.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def evaluate_third_layer_heads(
    manifest_path: str | Path,
    experiment_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    batch_size: int = 16,
    device: str | None = None,
) -> dict[str, Any]:
    """Evaluate all saved third-layer heads once on the manifest test subset."""
    import open_clip
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    experiment = Path(experiment_dir).expanduser().resolve()
    ranking_path = experiment / "ranking.json"
    if not ranking_path.is_file():
        raise FileNotFoundError(f"head comparison metadata not found: {ranking_path}")
    selection = json.loads(ranking_path.read_text(encoding="utf-8"))
    records, manifest_sha256 = read_training_manifest(manifest_path)
    if selection["manifest_sha256"] != manifest_sha256:
        raise ValueError("test manifest differs from the one used to train the five heads")

    node = next(node for node in HIERARCHY_NODES if node.name == "still_vs_landscape")
    class_names = tuple(selection["class_names"])
    test_records = [
        row | {"category": target_for(node, row["category"])}
        for row in records
        if row["split"] == "test" and target_for(node, row["category"]) is not None
    ]
    if not test_records:
        raise ValueError("manifest contains no third-layer test samples")

    chosen_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).resolve().parents[2]
    project_cache = project_root / "algorithms" / "models" / "huggingface"
    os.environ.setdefault("HF_HOME", str(project_cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(project_cache / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(project_cache / "transformers"))
    os.environ.setdefault("HF_MODULES_CACHE", str(project_cache / "modules"))
    backbone, _, preprocess = open_clip.create_model_and_transforms(
        selection["model_name"], pretrained=selection["pretrained"], device=chosen_device
    )
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)

    class TestImages(Dataset):
        def __len__(self) -> int:
            return len(test_records)

        def __getitem__(self, index: int):
            with Image.open(test_records[index]["path"]) as image:
                return preprocess(image.convert("RGB"))

    loader = DataLoader(TestImages(), batch_size=batch_size, shuffle=False, num_workers=0)
    features = []
    with torch.inference_mode():
        for batch_index, images in enumerate(loader, 1):
            vectors = backbone.encode_image(images.to(chosen_device))
            vectors = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            features.append(vectors.float().cpu())
            processed = min(batch_index * batch_size, len(test_records))
            print(f"third-layer features test: {processed}/{len(test_records)}", flush=True)
    test_x = torch.cat(features, dim=0)
    test_y = torch.tensor(
        [class_names.index(row["category"]) for row in test_records], dtype=torch.long
    )
    all_logits: dict[str, torch.Tensor] = {}

    for item in selection["ranking"]:
        name = item["name"]
        checkpoint_path = experiment / f"{name}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"candidate checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if tuple(checkpoint["class_names"]) != class_names:
            raise ValueError(f"candidate {name} uses different class order")

        if name == "linear_logistic":
            head = torch.nn.Linear(test_x.shape[1], len(class_names))
            head.load_state_dict(checkpoint["state_dict"])
            head.eval()
            with torch.inference_mode():
                logits = head(test_x.to(chosen_device)).cpu()
        elif name == "linear_svm_squared_hinge":
            head = torch.nn.Linear(test_x.shape[1], 1)
            head.load_state_dict(checkpoint["state_dict"])
            head.eval()
            with torch.inference_mode():
                scores = head(test_x.to(chosen_device)).squeeze(-1).cpu()
                logits = torch.stack((scores, -scores), dim=1)
        elif name == "mlp_256_dropout":
            head = torch.nn.Sequential(
                torch.nn.Linear(test_x.shape[1], 256),
                torch.nn.GELU(),
                torch.nn.Dropout(0.30),
                torch.nn.Linear(256, len(class_names)),
            )
            head.load_state_dict(checkpoint["state_dict"])
            head.eval()
            with torch.inference_mode():
                logits = head(test_x.to(chosen_device)).cpu()
        elif name == "rbf_kernel_ridge":
            train_x = checkpoint["train_features"].float()
            distances = (2.0 - 2.0 * (test_x @ train_x.T)).clamp_min(0.0)
            kernel = torch.exp(-float(checkpoint["gamma"]) * distances)
            scores = kernel @ checkpoint["alpha"].float()
            logits = torch.stack((scores, -scores), dim=1)
        elif name == "cosine_knn_7":
            train_x = checkpoint["train_features"].float()
            labels = checkpoint["train_labels"].long()
            similarities = test_x @ train_x.T
            neighbor_scores, neighbor_indices = similarities.topk(int(checkpoint["k"]), dim=1)
            weights = torch.softmax(neighbor_scores * float(checkpoint["temperature"]), dim=1)
            neighbor_labels = labels[neighbor_indices]
            logits = torch.zeros((len(test_records), len(class_names)), dtype=torch.float32)
            logits.scatter_add_(1, neighbor_labels, weights)
        else:
            raise ValueError(f"unknown third-layer candidate: {name}")
        all_logits[name] = logits

    from .training import _metrics_from_logits

    metrics = []
    for name, logits in all_logits.items():
        metrics.append({"name": name, **_metrics_from_logits(logits, test_records, class_names)})
    test_ranking = sorted(metrics, key=lambda row: (row["macro_f1"], row["accuracy"]), reverse=True)
    for index, row in enumerate(test_ranking, 1):
        row["rank"] = index

    output = Path(output_dir).expanduser().resolve() if output_dir else experiment
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "test_metrics.json"
    predictions_path = output / "test_predictions.jsonl"
    if metrics_path.exists() or predictions_path.exists():
        raise FileExistsError("test evaluation outputs already exist; refusing to overwrite them")
    with predictions_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(test_records):
            row = {
                "image_id": record["image_id"],
                "source_path": record["path"],
                "ground_truth": record["category"],
                "split": "test",
                "predictions": {
                    name: class_names[int(logits[index].argmax())]
                    for name, logits in all_logits.items()
                },
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = {
        "manifest_sha256": manifest_sha256,
        "node": node.name,
        "class_names": class_names,
        "test_samples": len(test_records),
        "scope": "conditional third-layer binary test subset; not end-to-end cascade",
        "test_used_for_model_selection": False,
        "ranking": test_ranking,
        "predictions": str(predictions_path),
    }
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def build_rbf_cascade_checkpoint(
    cascade_checkpoint: str | Path,
    rbf_head_checkpoint: str | Path,
    output_checkpoint: str | Path,
) -> dict[str, Any]:
    """Create a new cascade checkpoint by swapping only its third-stage head."""
    import torch

    source_path = Path(cascade_checkpoint).expanduser().resolve()
    head_path = Path(rbf_head_checkpoint).expanduser().resolve()
    output_path = Path(output_checkpoint).expanduser().resolve()
    if output_path in {source_path, head_path}:
        raise ValueError("output checkpoint must not overwrite either source checkpoint")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing checkpoint: {output_path}")

    cascade = torch.load(source_path, map_location="cpu", weights_only=True)
    rbf = torch.load(head_path, map_location="cpu", weights_only=True)
    if rbf.get("method") != "rbf_kernel_ridge":
        raise ValueError("candidate checkpoint is not an RBF kernel ridge head")
    if cascade.get("manifest_sha256") != rbf.get("manifest_sha256"):
        raise ValueError("RBF head and cascade were trained from different manifests")
    node_data = cascade["heads"].get("still_vs_landscape")
    expected_labels = ("静物摄影", "风光摄影")
    if node_data is None or tuple(node_data.get("class_names", ())) != expected_labels:
        raise ValueError("cascade third-stage labels do not match the RBF candidate")
    if tuple(rbf.get("class_names", ())) != expected_labels:
        raise ValueError("RBF candidate class order does not match the cascade")
    train_features = rbf["train_features"].float()
    alpha = rbf["alpha"].float().reshape(-1)
    if train_features.ndim != 2 or train_features.shape[1] != int(cascade["input_dim"]):
        raise ValueError("RBF features do not match cascade embedding dimension")
    if train_features.shape[0] != alpha.numel():
        raise ValueError("RBF coefficient count differs from its training sample count")

    cascade["heads"]["still_vs_landscape"] = {
        "head_type": "rbf_kernel_ridge",
        "class_names": expected_labels,
        "train_features": train_features,
        "alpha": alpha,
        "gamma": float(rbf["gamma"]),
        "regularization": float(rbf["regularization"]),
        "source_method": rbf["method"],
    }
    cascade["variant"] = "third_stage_rbf_kernel_ridge"
    cascade["training_method"] = (
        str(cascade.get("training_method", "shared frozen OpenCLIP encoder"))
        + "; third stage replaced with RBF kernel ridge head"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cascade, output_path)
    return {
        "checkpoint": str(output_path),
        "variant": cascade["variant"],
        "manifest_sha256": cascade["manifest_sha256"],
        "retained_linear_nodes": ["person_gate", "sky_gate"],
        "third_layer_head": rbf["method"],
        "rbf_training_vectors": int(train_features.shape[0]),
        "source_checkpoint_preserved": True,
    }
