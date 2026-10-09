"""Warm-started linear heads with explicit, train-only class loss weights."""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

from .hierarchy_v2_training import _metrics


def class_loss_weights(torch: Any, targets: Any, class_names: tuple[str, ...],
                       multipliers: tuple[float, ...]) -> Any:
    """Inverse train frequency times a named-class penalty multiplier.

    CrossEntropyLoss(mean) divides the weighted sum by the sum of target
    weights, not by the batch size. Weights never depend on validation labels.
    """
    if len(multipliers) != len(class_names) or any(
        not math.isfinite(v) or v <= 0 for v in multipliers
    ):
        raise ValueError("Provide one finite positive multiplier per class")
    counts = Counter(targets.tolist())
    if set(counts) != set(range(len(class_names))):
        raise ValueError("Training targets must contain every class and no unknown class")
    return torch.tensor([
        len(targets) / (len(class_names) * counts[i]) * multipliers[i]
        for i in range(len(class_names))
    ], dtype=torch.float32)


def fit_weighted_head(torch: Any, train_x: Any, train_y: Any,
                      validation_x: Any, validation_y: Any,
                      class_names: tuple[str, ...], initial_state: dict, *,
                      multipliers: tuple[float, ...], epochs: int, patience: int,
                      learning_rate: float, seed: int, on_epoch=None):
    """Keep epoch 0 as a validation safeguard; return best AND last weights.

    Like the original auto-crop trainer, each epoch is one full-batch AdamW
    step. No additional sampling or loss weighting is silently introduced.
    """
    if epochs < 1 or patience < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("epochs, patience and learning_rate must be positive")
    torch.manual_seed(seed)
    head = torch.nn.Linear(train_x.shape[1], len(class_names))
    head.load_state_dict(initial_state)
    weights = class_loss_weights(torch, train_y, class_names, multipliers)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=0.01)

    def snapshot():
        return {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}

    head.eval()
    with torch.inference_mode():
        baseline = _metrics(head(validation_x), validation_y, class_names)
    best_f1, best_epoch, best_state = baseline["macro_f1"], 0, snapshot()
    stale, history = 0, []
    for epoch in range(1, epochs + 1):
        head.train()
        order = torch.randperm(len(train_x), generator=torch.Generator().manual_seed(seed + epoch))
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(head(train_x[order]), train_y[order])
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in head.parameters()):
            raise RuntimeError(f"Non-finite gradients at epoch {epoch}")
        optimizer.step()
        head.eval()
        with torch.inference_mode():
            metric = _metrics(head(validation_x), validation_y, class_names)
        row = {"epoch": epoch, "weighted_training_loss": float(loss.item()), **metric}
        history.append(row)
        if metric["macro_f1"] > best_f1:
            best_f1, best_epoch, best_state = metric["macro_f1"], epoch, snapshot()
            stale = 0
        else:
            stale += 1
        if on_epoch:
            on_epoch(row, best_epoch)
        if stale >= patience:
            break
    metadata = {
        "class_names": class_names, "best_epoch": best_epoch,
        "validation_macro_f1": best_f1,
        "train_counts": dict(Counter(train_y.tolist())),
        "class_multipliers": list(multipliers),
        "loss_weights": weights.tolist(), "loss_reduction": "weighted_mean",
        "completed_epochs": len(history), "optimizer_steps": len(history),
        "selection_metric": "validation macro_f1 (strict improvement, epoch 0 eligible)",
        "warm_started": True,
    }
    return (metadata | {"state_dict": best_state},
            {"history": history, "baseline_validation": baseline,
             "stop_reason": "patience" if stale >= patience else "max_epochs"},
            metadata | {"state_dict": snapshot(), "saved_epoch": len(history)})


def add_per_class_metrics(metrics: dict) -> dict:
    """Attach precision/recall/F1/support to every node and the cascade."""
    for metric in metrics.values():
        matrix = metric["confusion_matrix"]
        details = {}
        for i, name in enumerate(metric["class_names"]):
            tp, support = matrix[i][i], sum(matrix[i])
            predicted = sum(row[i] for row in matrix)
            precision = tp / predicted if predicted else 0.0
            recall = tp / support if support else 0.0
            f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
            details[name] = {"support": support, "precision": precision, "recall": recall, "f1": f1}
        metric["per_class"] = details
    return metrics
