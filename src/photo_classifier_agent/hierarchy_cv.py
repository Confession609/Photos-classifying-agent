"""Stratified K-fold evaluation for the three-stage photo-classification cascade."""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .categories import CATEGORY_NAMES
from .evaluation import evaluate
from .hierarchy import HIERARCHY_NODES, make_hierarchical_decision, route_category, routed_leaf_scores, target_for
from .reports import write_decisions
from .schemas import CandidateScore, Prediction
from .training import _metrics_from_logits, read_training_manifest


def _stratified_buckets(records: list[dict[str, Any]], folds: int, seed: int) -> list[list[dict[str, Any]]]:
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_category.setdefault(row["category"], []).append(row)
    rng = random.Random(seed)
    result: list[list[dict[str, Any]]] = [[] for _ in range(folds)]
    for category_index, category in enumerate(CATEGORY_NAMES):
        bucket = by_category[category]
        if len(bucket) < folds:
            raise ValueError(f"class {category!r} has fewer samples than folds={folds}")
        rng.shuffle(bucket)
        offset = category_index % folds
        for index, row in enumerate(bucket):
            result[(index + offset) % folds].append(row)
    return result


def _inner_split(records: list[dict[str, Any]], seed: int, fraction: float = 0.15):
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_category.setdefault(row["category"], []).append(row)
    rng = random.Random(seed)
    fit_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for category, bucket in sorted(by_category.items()):
        if len(bucket) < 2:
            raise ValueError(f"outer training fold has too few {category!r} samples for inner validation")
        rng.shuffle(bucket)
        validation_count = max(1, round(len(bucket) * fraction))
        validation_count = min(validation_count, len(bucket) - 1)
        validation_rows.extend(bucket[:validation_count])
        fit_rows.extend(bucket[validation_count:])
    return fit_rows, validation_rows


def _binary_data(rows, feature_by_id, node, class_names, torch):
    eligible = [row for row in rows if target_for(node, row["category"]) is not None]
    mapped = [row | {"category": target_for(node, row["category"])} for row in eligible]
    if not mapped:
        raise ValueError(f"no eligible samples for hierarchy node {node.name}")
    x = torch.stack([feature_by_id[row["image_id"]] for row in eligible])
    y = torch.tensor([class_names.index(row["category"]) for row in mapped], dtype=torch.long)
    return x, y, mapped


def _class_weights(labels, class_names, rows, torch, device):
    counts = Counter(row["category"] for row in rows)
    return torch.tensor(
        [len(rows) / (len(class_names) * counts[label]) for label in class_names],
        dtype=torch.float32,
        device=device,
    )


def _fit_linear(
    train_rows, validation_rows, feature_by_id, node, *, input_dim, epochs, patience, learning_rate,
    seed, device, torch,
):
    class_names = (node.positive_label, node.negative_label)
    train_x, train_y, mapped_train = _binary_data(train_rows, feature_by_id, node, class_names, torch)
    validation_x, validation_y, mapped_validation = _binary_data(
        validation_rows, feature_by_id, node, class_names, torch
    )
    train_x, train_y = train_x.to(device), train_y.to(device)
    validation_x, validation_y = validation_x.to(device), validation_y.to(device)
    weights = _class_weights(train_y, class_names, mapped_train, torch, device)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    head = torch.nn.Linear(input_dim, 2).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=0.01)
    best_f1, best_epoch, stale = -1.0, 0, 0
    for epoch in range(1, epochs + 1):
        head.train()
        permutation = torch.randperm(len(train_y), generator=generator)
        for start in range(0, len(permutation), 64):
            indices = permutation[start : start + 64].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(head(train_x[indices]), train_y[indices], weight=weights)
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.inference_mode():
            metrics = _metrics_from_logits(head(validation_x).cpu(), mapped_validation, class_names)
        if metrics["macro_f1"] > best_f1:
            best_f1, best_epoch, stale = metrics["macro_f1"], epoch, 0
        else:
            stale += 1
        if stale >= patience:
            break
    return best_epoch, best_f1


def _refit_linear(rows, feature_by_id, node, *, input_dim, epochs, learning_rate, seed, device, torch):
    class_names = (node.positive_label, node.negative_label)
    train_x, train_y, mapped_rows = _binary_data(rows, feature_by_id, node, class_names, torch)
    train_x, train_y = train_x.to(device), train_y.to(device)
    weights = _class_weights(train_y, class_names, mapped_rows, torch, device)
    torch.manual_seed(seed)
    head = torch.nn.Linear(input_dim, 2).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=0.01)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        head.train()
        permutation = torch.randperm(len(train_y), generator=generator)
        for start in range(0, len(permutation), 64):
            indices = permutation[start : start + 64].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(head(train_x[indices]), train_y[indices], weight=weights)
            loss.backward()
            optimizer.step()
    return head.eval()


def _fit_rbf(train_rows, validation_rows, feature_by_id, node, class_names, torch):
    train_x, train_y, mapped_train = _binary_data(train_rows, feature_by_id, node, class_names, torch)
    validation_x, _, mapped_validation = _binary_data(
        validation_rows, feature_by_id, node, class_names, torch
    )
    counts = Counter(row["category"] for row in mapped_train)
    sample_weights = torch.tensor(
        [len(train_y) / (2 * counts[class_names[int(label)]]) for label in train_y.tolist()],
        dtype=torch.float32,
    )
    targets = torch.where(train_y == 0, 1.0, -1.0)
    train_distances = (2.0 - 2.0 * (train_x @ train_x.T)).clamp_min(0.0)
    validation_distances = (2.0 - 2.0 * (validation_x @ train_x.T)).clamp_min(0.0)
    best = None
    for gamma in (1.0, 2.0, 4.0, 8.0):
        train_kernel = torch.exp(-gamma * train_distances)
        validation_kernel = torch.exp(-gamma * validation_distances)
        for regularization in (0.01, 0.1, 1.0):
            system = train_kernel + torch.diag(regularization / sample_weights)
            alpha = torch.linalg.solve(system, targets)
            scores = validation_kernel @ alpha
            logits = torch.stack((scores, -scores), dim=1)
            metrics = _metrics_from_logits(logits, mapped_validation, class_names)
            rank = (metrics["macro_f1"], metrics["accuracy"])
            if best is None or rank > best[0]:
                best = (rank, gamma, regularization)
    assert best is not None
    _, gamma, regularization = best

    full_x, full_y, full_rows = _binary_data(train_rows + validation_rows, feature_by_id, node, class_names, torch)
    full_counts = Counter(row["category"] for row in full_rows)
    full_weights = torch.tensor(
        [len(full_y) / (2 * full_counts[class_names[int(label)]]) for label in full_y.tolist()],
        dtype=torch.float32,
    )
    full_targets = torch.where(full_y == 0, 1.0, -1.0)
    full_distances = (2.0 - 2.0 * (full_x @ full_x.T)).clamp_min(0.0)
    full_kernel = torch.exp(-gamma * full_distances)
    alpha = torch.linalg.solve(full_kernel + torch.diag(regularization / full_weights), full_targets)
    return {
        "head_type": "rbf_kernel_ridge",
        "class_names": class_names,
        "train_features": full_x,
        "alpha": alpha,
        "gamma": gamma,
        "regularization": regularization,
    }, {"gamma": gamma, "regularization": regularization, "inner_validation_macro_f1": best[0][0]}


def run_hierarchical_rbf_cross_validation(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    folds: int = 5,
    seed: int = 1337,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: str | None = None,
    batch_size: int = 16,
    epochs: int = 60,
    patience: int = 10,
    learning_rate: float = 0.01,
    third_head: str = "rbf_kernel_ridge",
) -> dict[str, Any]:
    """Run nested stratified K-fold CV with a selectable linear or RBF third node."""
    import open_clip
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    if folds < 2 or batch_size < 1 or epochs < 1 or patience < 1:
        raise ValueError("folds must be >=2 and batch_size/epochs/patience must be positive")
    if third_head not in {"linear", "rbf_kernel_ridge"}:
        raise ValueError("third_head must be 'linear' or 'rbf_kernel_ridge'")
    records, manifest_sha256 = read_training_manifest(manifest_path)
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty CV output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    fold_test_rows = _stratified_buckets(records, folds, seed)
    for index, rows in enumerate(fold_test_rows):
        print(f"fold {index + 1}/{folds} held-out class counts: {dict(Counter(r['category'] for r in rows))}", flush=True)

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

    class AllImages(Dataset):
        def __len__(self):
            return len(records)

        def __getitem__(self, index):
            with Image.open(records[index]["path"]) as image:
                return preprocess(image.convert("RGB"))

    feature_chunks = []
    loader = DataLoader(AllImages(), batch_size=batch_size, shuffle=False, num_workers=0)
    print(
        f"Extracting frozen OpenCLIP features for all {len(records)} samples once; "
        "outer-fold labels remain isolated from fitting.",
        flush=True,
    )
    with torch.inference_mode():
        for batch_index, images in enumerate(loader, 1):
            vectors = backbone.encode_image(images.to(chosen_device))
            vectors = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            feature_chunks.append(vectors.float().cpu())
            print(f"CV image features: {min(batch_index * batch_size, len(records))}/{len(records)}", flush=True)
    all_features = torch.cat(feature_chunks, dim=0)
    feature_by_id = {row["image_id"]: all_features[i] for i, row in enumerate(records)}
    node_person, node_sky, node_still = HIERARCHY_NODES
    class_names_by_node = {
        node.name: (node.positive_label, node.negative_label)
        for node in HIERARCHY_NODES
    }
    fold_results = []
    oof_decisions: list[ClassificationDecision] = []
    oof_truth: dict[str, str] = {}
    oof_prediction_rows: list[dict[str, Any]] = []
    node_logits: dict[str, list[torch.Tensor]] = {node.name: [] for node in HIERARCHY_NODES}
    node_truth_rows: dict[str, list[dict[str, Any]]] = {node.name: [] for node in HIERARCHY_NODES}
    seen_test_ids: set[str] = set()
    folds_dir = output / "folds"
    folds_dir.mkdir(exist_ok=True)

    for fold_index, fold_test in enumerate(fold_test_rows):
        fold_number = fold_index + 1
        test_ids = {row["image_id"] for row in fold_test}
        train_rows = [row for row in records if row["image_id"] not in test_ids]
        if seen_test_ids.intersection(test_ids):
            raise ValueError("a sample was assigned to multiple held-out folds")
        seen_test_ids.update(test_ids)
        inner_train_rows, inner_validation_rows = _inner_split(train_rows, seed + 100 * fold_number)
        print(
            f"Training fold {fold_number}/{folds}: outer_train={len(train_rows)}, "
            f"inner_train={len(inner_train_rows)}, inner_validation={len(inner_validation_rows)}, "
            f"held_out={len(fold_test)}",
            flush=True,
        )

        fold_heads: dict[str, dict[str, Any]] = {}
        selected_epochs: dict[str, int] = {}
        inner_scores: dict[str, float] = {}
        for node_index, node in enumerate((node_person, node_sky)):
            best_epoch, inner_f1 = _fit_linear(
                inner_train_rows,
                inner_validation_rows,
                feature_by_id,
                node,
                input_dim=all_features.shape[1],
                epochs=epochs,
                patience=patience,
                learning_rate=learning_rate,
                seed=seed + fold_number * 1000 + node_index,
                device=chosen_device,
                torch=torch,
            )
            head = _refit_linear(
                train_rows,
                feature_by_id,
                node,
                input_dim=all_features.shape[1],
                epochs=best_epoch,
                learning_rate=learning_rate,
                seed=seed + fold_number * 1000 + node_index + 50,
                device=chosen_device,
                torch=torch,
            )
            fold_heads[node.name] = {
                "head_type": "linear",
                "class_names": class_names_by_node[node.name],
                "state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
                "best_epoch": best_epoch,
                "inner_validation_macro_f1": inner_f1,
            }
            selected_epochs[node.name] = best_epoch
            inner_scores[node.name] = inner_f1

        third_selection: dict[str, Any] = {}
        if third_head == "rbf_kernel_ridge":
            inner_still_train, inner_still_validation = _inner_split(
                [row for row in train_rows if target_for(node_still, row["category"]) is not None],
                seed + 100 * fold_number + 77,
            )
            final_head, third_selection = _fit_rbf(
                inner_still_train,
                inner_still_validation,
                feature_by_id,
                node_still,
                class_names_by_node[node_still.name],
                torch,
            )
            # Refit selected RBF settings on all eligible outer-training rows.
            outer_still_rows = [row for row in train_rows if target_for(node_still, row["category"]) is not None]
            rbf_x, rbf_y, _ = _binary_data(
                outer_still_rows, feature_by_id, node_still, class_names_by_node[node_still.name], torch
            )
            rbf_counts = Counter(rbf_y.tolist())
            rbf_weights = torch.tensor(
                [len(rbf_y) / (2 * rbf_counts[i]) for i in rbf_y.tolist()], dtype=torch.float32
            )
            rbf_targets = torch.where(rbf_y == 0, 1.0, -1.0)
            rbf_kernel = torch.exp(
                -third_selection["gamma"] * (2.0 - 2.0 * (rbf_x @ rbf_x.T)).clamp_min(0.0)
            )
            rbf_alpha = torch.linalg.solve(
                rbf_kernel + torch.diag(third_selection["regularization"] / rbf_weights), rbf_targets
            )
            final_head.update({
                "train_features": rbf_x,
                "alpha": rbf_alpha,
                "inner_validation_macro_f1": third_selection["inner_validation_macro_f1"],
            })
            inner_scores[node_still.name] = third_selection["inner_validation_macro_f1"]
        else:
            still_rows = [row for row in train_rows if target_for(node_still, row["category"]) is not None]
            inner_still_train, inner_still_validation = _inner_split(
                still_rows, seed + 100 * fold_number + 77
            )
            best_epoch, inner_f1 = _fit_linear(
                inner_still_train,
                inner_still_validation,
                feature_by_id,
                node_still,
                input_dim=all_features.shape[1],
                epochs=epochs,
                patience=patience,
                learning_rate=learning_rate,
                seed=seed + fold_number * 1000 + 2,
                device=chosen_device,
                torch=torch,
            )
            head = _refit_linear(
                train_rows,
                feature_by_id,
                node_still,
                input_dim=all_features.shape[1],
                epochs=best_epoch,
                learning_rate=learning_rate,
                seed=seed + fold_number * 1000 + 52,
                device=chosen_device,
                torch=torch,
            )
            final_head = {
                "head_type": "linear",
                "class_names": class_names_by_node[node_still.name],
                "state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
                "best_epoch": best_epoch,
                "inner_validation_macro_f1": inner_f1,
            }
            selected_epochs[node_still.name] = best_epoch
            inner_scores[node_still.name] = inner_f1
        fold_heads[node_still.name] = final_head

        # Model selection is complete before any held-out fold labels are inspected.
        fold_test_x = torch.stack([feature_by_id[row["image_id"]] for row in fold_test])
        fold_logits: dict[str, torch.Tensor] = {}
        for node in (node_person, node_sky):
            head = torch.nn.Linear(all_features.shape[1], 2)
            head.load_state_dict(fold_heads[node.name]["state_dict"])
            head.eval()
            with torch.inference_mode():
                fold_logits[node.name] = head(fold_test_x).cpu()
        if third_head == "rbf_kernel_ridge":
            distances = (2.0 - 2.0 * (fold_test_x @ final_head["train_features"].T)).clamp_min(0.0)
            scores = torch.exp(-final_head["gamma"] * distances) @ final_head["alpha"]
            fold_logits[node_still.name] = torch.stack((scores, -scores), dim=1)
        else:
            head = torch.nn.Linear(all_features.shape[1], 2)
            head.load_state_dict(final_head["state_dict"])
            head.eval()
            with torch.inference_mode():
                fold_logits[node_still.name] = head(fold_test_x).cpu()

        fold_decisions = []
        for local_index, row in enumerate(fold_test):
            probabilities = {
                name: torch.softmax(logits[local_index], dim=-1).tolist()
                for name, logits in fold_logits.items()
            }
            person_probability = float(probabilities[node_person.name][0])
            sky_probability = float(probabilities[node_sky.name][0])
            still_probability = float(probabilities[node_still.name][0])
            category = route_category(person_probability, sky_probability, still_probability)
            scores_by_leaf = routed_leaf_scores(category, person_probability, sky_probability, still_probability)
            version = f"{model_name}:{pretrained}:stratified-{folds}fold:third-stage-{third_head}"
            prediction = Prediction(
                image_id=row["image_id"],
                source_path=row["path"],
                scores=tuple(CandidateScore(name, float(scores_by_leaf[name])) for name in CATEGORY_NAMES),
                model_name=f"open_clip_hierarchical_cv:{model_name}",
                model_version=version,
            )
            decision = make_hierarchical_decision(
                prediction.image_id,
                prediction.source_path,
                person_probability,
                sky_probability,
                still_probability,
                classifier_version=version,
            )
            fold_decisions.append(decision)
            oof_decisions.append(decision)
            oof_truth[row["image_id"]] = row["category"]
            oof_prediction_rows.append({
                "image_id": row["image_id"],
                "source_path": row["path"],
                "ground_truth": row["category"],
                "prediction": category,
                "fold": fold_number,
                "split": "out_of_fold",
                "node_probabilities": {key: float(value[0]) for key, value in probabilities.items()},
            })
            for node in HIERARCHY_NODES:
                target = target_for(node, row["category"])
                if target is not None:
                    node_logits[node.name].append(fold_logits[node.name][local_index].cpu())
                    node_truth_rows[node.name].append(row | {"category": target})

        fold_result = evaluate(fold_decisions, {row["image_id"]: row["category"] for row in fold_test})
        fold_results.append({
            "fold": fold_number,
            "train_count": len(train_rows),
            "held_out_count": len(fold_test),
            "inner_selected_epochs": selected_epochs,
            "inner_validation_macro_f1": inner_scores,
            "third_head": third_head,
            **({"third_head_gamma": third_selection["gamma"],
                "third_head_regularization": third_selection["regularization"]}
               if third_head == "rbf_kernel_ridge" else {}),
            "metrics": fold_result.to_dict(),
        })
        fold_checkpoint = {
            "heads": fold_heads,
            "node_order": tuple(node.name for node in HIERARCHY_NODES),
            "leaf_categories": tuple(CATEGORY_NAMES),
            "input_dim": int(all_features.shape[1]),
            "model_name": model_name,
            "pretrained": pretrained,
            "manifest_sha256": manifest_sha256,
            "seed": seed,
            "variant": f"fold_{fold_number}_third_stage_{third_head}",
            "fold": fold_number,
            "training_method": f"stratified outer K-fold; inner validation; linear gates + {third_head} third head",
        }
        torch.save(fold_checkpoint, folds_dir / f"fold_{fold_number}.pt")
        print(json.dumps({"fold": fold_number, "metrics": fold_result.to_dict()}, ensure_ascii=False), flush=True)

    if seen_test_ids != {row["image_id"] for row in records}:
        raise ValueError("out-of-fold coverage is incomplete or contains unknown samples")
    aggregate = evaluate(oof_decisions, oof_truth)
    node_summary = {}
    for node in HIERARCHY_NODES:
        node_summary[node.name] = _metrics_from_logits(
            torch.stack(node_logits[node.name]),
            node_truth_rows[node.name],
            (node.positive_label, node.negative_label),
        )

    pred_path = output / "out_of_fold_predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as handle:
        for row in sorted(oof_prediction_rows, key=lambda item: item["image_id"]):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    decision_path = output / "out_of_fold_decisions.jsonl"
    write_decisions(oof_decisions, decision_path)
    summary = {
        "manifest_sha256": manifest_sha256,
        "method": "stratified nested K-fold cross-validation",
        "folds": folds,
        "samples": len(records),
        "each_sample_held_out_once": True,
        "inner_validation_fraction_of_outer_train": 0.15,
        "stratification": "leaf category",
        "backbone": {"name": model_name, "pretrained": pretrained, "frozen": True},
        "cascade_heads": {"person_gate": "linear", "sky_gate": "linear", "still_vs_landscape": third_head},
        **({"rbf_grid": {"gamma": [1.0, 2.0, 4.0, 8.0], "regularization": [0.01, 0.1, 1.0]}}
           if third_head == "rbf_kernel_ridge" else {}),
        "fold_metrics": fold_results,
        "out_of_fold_metrics": aggregate.to_dict(),
        "node_out_of_fold_metrics": node_summary,
        "out_of_fold_predictions": str(pred_path),
        "out_of_fold_decisions": str(decision_path),
        "fold_checkpoints": [str(folds_dir / f"fold_{index}.pt") for index in range(1, folds + 1)],
        "test_set_note": "Earlier experiments used this dataset for model comparisons; CV results are exploratory, not a fresh untouched test estimate.",
    }
    metrics_path = output / "cross_validation_metrics.json"
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
