"""Train independent binary heads for each conditional hierarchy node."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .categories import CATEGORY_NAMES
from .evaluation import evaluate, write_evaluation
from .hierarchy import (
    HIERARCHY_NODES,
    make_hierarchical_decision,
    project_manifest_to_nodes,
    route_category,
    routed_leaf_scores,
    target_for,
)
from .reports import write_decisions
from .schemas import CandidateScore, ClassificationDecision, Prediction
from .training import _metrics_from_logits, read_training_manifest


def train_hierarchical_probe(
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
    third_head: str = "linear",
) -> dict[str, Any]:
    """Fit a three-node cascade, optionally using RBF kernel ridge at stage three."""
    import os

    import open_clip
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    if batch_size < 1 or epochs < 1 or patience < 1:
        raise ValueError("batch_size, epochs, and patience must be positive")
    if third_head not in {"linear", "rbf_kernel_ridge"}:
        raise ValueError("third_head must be linear or rbf_kernel_ridge")
    records, manifest_sha256 = read_training_manifest(manifest_path)
    split_records = {
        split: [record for record in records if record["split"] == split]
        for split in ("train", "validation", "test")
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    node_manifests = project_manifest_to_nodes(records, output / "manifests")

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
        def __init__(self, items: list[dict[str, Any]]):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            with Image.open(self.items[index]["path"]) as image:
                return preprocess(image.convert("RGB"))

    def extract(items: list[dict[str, Any]], split: str):
        loader = DataLoader(ManifestImages(items), batch_size=batch_size, shuffle=False, num_workers=0)
        vectors = []
        with torch.inference_mode():
            for batch_index, images in enumerate(loader, 1):
                features = backbone.encode_image(images.to(chosen_device))
                features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                vectors.append(features.float().cpu())
                print(f"hierarchy features {split}: {min(batch_index * batch_size, len(items))}/{len(items)}", flush=True)
        return torch.cat(vectors)

    print(f"Manifest SHA-256: {manifest_sha256}", flush=True)
    print("Extracting train/validation features only; test remains unopened until all heads are selected.", flush=True)
    train_x = extract(split_records["train"], "train")
    validation_x = extract(split_records["validation"], "validation")
    train_positions = {record["image_id"]: index for index, record in enumerate(split_records["train"])}
    validation_positions = {record["image_id"]: index for index, record in enumerate(split_records["validation"])}

    trained_heads: dict[str, dict[str, Any]] = {}
    best_epochs: dict[str, int] = {}
    histories: dict[str, list[dict[str, Any]]] = {}
    for node_index, node in enumerate(HIERARCHY_NODES):
        class_names = (node.positive_label, node.negative_label)
        node_train = [record for record in split_records["train"] if target_for(node, record["category"]) is not None]
        node_validation = [record for record in split_records["validation"] if target_for(node, record["category"]) is not None]
        train_indices = torch.tensor([train_positions[row["image_id"]] for row in node_train], dtype=torch.long)
        validation_indices = torch.tensor([validation_positions[row["image_id"]] for row in node_validation], dtype=torch.long)
        train_labels = torch.tensor([class_names.index(target_for(node, row["category"])) for row in node_train])
        validation_labels = torch.tensor([class_names.index(target_for(node, row["category"])) for row in node_validation])
        train_counts = Counter(target_for(node, row["category"]) for row in node_train)
        validation_counts = Counter(target_for(node, row["category"]) for row in node_validation)
        if any(train_counts[label] == 0 or validation_counts[label] == 0 for label in class_names):
            raise ValueError(f"node {node.name} needs both labels in train and validation")

        if node.name == "still_vs_landscape" and third_head == "rbf_kernel_ridge":
            node_train_x = train_x[train_indices].float().cpu()
            node_validation_x = validation_x[validation_indices].float().cpu()
            train_binary_labels = train_labels
            gamma = 4.0
            regularization = 0.10
            distances = (2.0 - 2.0 * (node_train_x @ node_train_x.T)).clamp_min(0.0)
            kernel = torch.exp(-gamma * distances)
            binary_targets = torch.where(train_binary_labels == 0, 1.0, -1.0)
            sample_weights = torch.tensor(
                [len(train_binary_labels) / (2 * int(train_counts[class_names[int(label)]]))
                 for label in train_binary_labels.tolist()],
                dtype=torch.float32,
            )
            alpha = torch.linalg.solve(
                kernel + torch.diag(regularization / sample_weights), binary_targets
            )
            val_distances = (2.0 - 2.0 * (node_validation_x @ node_train_x.T)).clamp_min(0.0)
            val_scores = torch.exp(-gamma * val_distances) @ alpha
            val_logits = torch.stack((val_scores, -val_scores), dim=1)
            validation_metric_records = [
                row | {"category": target_for(node, row["category"])}
                for row in node_validation
            ]
            metrics = _metrics_from_logits(val_logits, validation_metric_records, class_names)
            trained_heads[node.name] = {
                "head_type": "rbf_kernel_ridge",
                "class_names": class_names,
                "train_features": node_train_x,
                "alpha": alpha,
                "gamma": gamma,
                "regularization": regularization,
                "best_epoch": None,
                "validation_macro_f1": metrics["macro_f1"],
                "train_counts": dict(train_counts),
                "validation_counts": dict(validation_counts),
            }
            histories[node.name] = [{"method": "rbf_kernel_ridge", **metrics}]
            print(json.dumps({"node": node.name, "method": "rbf_kernel_ridge", **metrics}, ensure_ascii=True), flush=True)
            continue

        head = torch.nn.Linear(train_x.shape[1], len(class_names)).to(chosen_device)
        weights = torch.tensor(
            [len(node_train) / (len(class_names) * train_counts[label]) for label in class_names],
            dtype=torch.float32,
            device=chosen_device,
        )
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=0.01)
        node_train_x = train_x[train_indices].to(chosen_device)
        node_train_y = train_labels.to(chosen_device)
        node_validation_x = validation_x[validation_indices].to(chosen_device)
        node_validation_y = validation_labels.to(chosen_device)
        validation_metric_records = [row | {"category": target_for(node, row["category"])} for row in node_validation]
        generator = torch.Generator().manual_seed(seed + node_index)
        best_f1 = -1.0
        best_state = None
        best_epoch = 0
        stale_epochs = 0
        history: list[dict[str, Any]] = []

        print(f"Training independent node {node.name}; train={len(node_train)}, validation={len(node_validation)}", flush=True)
        for epoch in range(1, epochs + 1):
            head.train()
            permutation = torch.randperm(len(node_train_x), generator=generator)
            loss_total = 0.0
            for start in range(0, len(permutation), batch_size):
                indices = permutation[start:start + batch_size].to(chosen_device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(head(node_train_x[indices]), node_train_y[indices])
                loss.backward()
                optimizer.step()
                loss_total += loss.item() * len(indices)
            head.eval()
            with torch.inference_mode():
                metrics = _metrics_from_logits(head(node_validation_x), validation_metric_records, class_names)
            row = {"epoch": epoch, "train_loss": loss_total / len(node_train_x), **metrics}
            history.append(row)
            print(json.dumps({"node": node.name, **row}, ensure_ascii=True), flush=True)
            if metrics["macro_f1"] > best_f1:
                best_f1 = metrics["macro_f1"]
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Node {node.name} early-stopped at epoch {epoch}; best={best_epoch}.", flush=True)
                break
        if best_state is None:
            raise RuntimeError(f"node {node.name} did not yield a validation checkpoint")
        trained_heads[node.name] = {
            "head_type": "linear",
            "state_dict": best_state,
            "class_names": class_names,
            "best_epoch": best_epoch,
            "validation_macro_f1": best_f1,
            "train_counts": dict(train_counts),
            "validation_counts": dict(validation_counts),
        }
        best_epochs[node.name] = best_epoch
        histories[node.name] = history

    checkpoint_path = output / "hierarchical_openclip_probe.pt"
    checkpoint = {
        "heads": trained_heads,
        "node_order": tuple(node.name for node in HIERARCHY_NODES),
        "leaf_categories": tuple(CATEGORY_NAMES),
        "input_dim": int(train_x.shape[1]),
        "model_name": model_name,
        "pretrained": pretrained,
        "manifest_sha256": manifest_sha256,
        "seed": seed,
        "variant": f"third_stage_{third_head}" if third_head != "linear" else "all_linear_heads",
        "training_method": (
            "shared frozen OpenCLIP image encoder + two independent binary linear heads "
            + ("+ third-stage RBF kernel ridge head" if third_head == "rbf_kernel_ridge" else "+ third binary linear head")
        ),
    }
    torch.save(checkpoint, checkpoint_path)
    (output / "training_history.json").write_text(json.dumps(histories, ensure_ascii=False, indent=2), encoding="utf-8")

    # Do not open test images until all three node checkpoints are fixed.
    test_x = extract(split_records["test"], "test")
    head_modules: dict[str, Any] = {}
    for node in HIERARCHY_NODES:
        if trained_heads[node.name]["head_type"] == "rbf_kernel_ridge":
            continue
        module = torch.nn.Linear(test_x.shape[1], 2).to(chosen_device)
        module.load_state_dict(trained_heads[node.name]["state_dict"])
        module.eval()
        head_modules[node.name] = module

    decisions: list[ClassificationDecision] = []
    logits_by_node: dict[str, Any] = {}
    with torch.inference_mode():
        for node in HIERARCHY_NODES:
            node_data = trained_heads[node.name]
            if node_data["head_type"] == "rbf_kernel_ridge":
                node_train_x = node_data["train_features"].to(chosen_device)
                distances = (2.0 - 2.0 * (test_x.to(chosen_device) @ node_train_x.T)).clamp_min(0.0)
                scores = torch.exp(-node_data["gamma"] * distances) @ node_data["alpha"].to(chosen_device)
                logits_by_node[node.name] = torch.stack((scores, -scores), dim=1).cpu()
            else:
                logits_by_node[node.name] = head_modules[node.name](test_x.to(chosen_device)).cpu()
        for index, record in enumerate(split_records["test"]):
            node_probs = {
                node.name: torch.softmax(logits_by_node[node.name][index], dim=-1).tolist()
                for node in HIERARCHY_NODES
            }
            person_probability = float(node_probs["person_gate"][0])
            sky_probability = float(node_probs["sky_gate"][0])
            still_probability = float(node_probs["still_vs_landscape"][0])
            routed_category = route_category(person_probability, sky_probability, still_probability)
            leaf_scores = routed_leaf_scores(
                routed_category,
                person_probability,
                sky_probability,
                still_probability,
            )
            prediction = Prediction(
                image_id=record["image_id"],
                source_path=record["path"],
                scores=tuple(CandidateScore(name, float(leaf_scores[name])) for name in CATEGORY_NAMES),
                model_name=f"open_clip_hierarchical:{model_name}",
                model_version=f"{model_name}:{pretrained}:cascade:{checkpoint['variant']}",
            )
            decisions.append(make_hierarchical_decision(
                prediction.image_id,
                prediction.source_path,
                person_probability,
                sky_probability,
                still_probability,
                classifier_version=prediction.model_version,
            ))

    node_metrics = {}
    for node in HIERARCHY_NODES:
        eligible = [
            (index, record)
            for index, record in enumerate(split_records["test"])
            if target_for(node, record["category"]) is not None
        ]
        if eligible:
            indexes = torch.tensor([index for index, _ in eligible], dtype=torch.long)
            rows = [record | {"category": target_for(node, record["category"])} for _, record in eligible]
            node_metrics[node.name] = _metrics_from_logits(
                logits_by_node[node.name].index_select(0, indexes), rows,
                (node.positive_label, node.negative_label),
            )
        else:
            node_metrics[node.name] = {"count": 0, "note": "no test samples reached this branch"}
    decision_path = output / "test_decisions.jsonl"
    write_decisions(decisions, decision_path)
    truth = {record["image_id"]: record["category"] for record in split_records["test"]}
    result = evaluate(decisions, truth)
    metrics_path = write_evaluation(result, output / "test_metrics.json")
    summary = {
        "manifest_sha256": manifest_sha256,
        "split_counts": {split: len(rows) for split, rows in split_records.items()},
        "node_manifests": node_manifests,
        "node_training": {
            name: {
                key: value for key, value in record.items()
                if key not in {"state_dict", "train_features", "alpha"}
            }
            for name, record in trained_heads.items()
        },
        "node_test_metrics": node_metrics,
        "cascade_test_metrics": result.to_dict(),
        "checkpoint": str(checkpoint_path),
        "test_decisions": str(decision_path),
        "test_metrics_path": str(metrics_path),
    }
    (output / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
