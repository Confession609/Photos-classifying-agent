"""Leaf-class penalties for the existing binary + three-way cascade."""
from __future__ import annotations

import math
from itertools import product


def canonical_multipliers(values):
    """Remove common positive scale, which cancels in weighted-mean CE."""
    if len(values) != 4 or any(not math.isfinite(v) or not 1 <= v <= 3 for v in values):
        raise ValueError("Four finite leaf multipliers in [1, 3] are required")
    smallest = min(values)
    return tuple(round(v / smallest, 10) for v in values)


def coarse_candidates():
    levels = (1., 1.5, 2., 2.5, 3.)
    return sorted({canonical_multipliers(p) for p in product(levels, repeat=4)})


def local_candidates(center, step=.1, radius=2):
    axes = [sorted({round(min(3., max(1., v + k * step)), 10)
                    for k in range(-radius, radius + 1)}) for v in center]
    return sorted({canonical_multipliers(p) for p in product(*axes)})


def coordinate_candidates(center, step=.05, radius=4):
    pairs = set()
    for index in range(4):
        for offset in range(-radius, radius + 1):
            values = list(center)
            values[index] = round(min(3., max(1., values[index] + offset * step)), 10)
            pairs.add(canonical_multipliers(values))
    return sorted(pairs)


def weighted_gate_loss(torch, logits, binary_targets, leaf_targets, base_weights, multipliers):
    """Binary gate task, with each non-person leaf getting its own penalty.

    The gate is NOT made four-way: its targets remain person/non-person.
    Original binary inverse-frequency weights are retained. Ground-truth leaf
    labels only choose training-sample penalties; none are inference inputs.
    """
    factors = torch.tensor(multipliers, dtype=logits.dtype, device=logits.device)
    weights = base_weights[binary_targets] * factors[leaf_targets]
    losses = torch.nn.functional.cross_entropy(logits, binary_targets, reduction="none")
    return (losses * weights).sum() / weights.sum()


def examination_rank(row):
    m = row["examination"]
    return (m["macro_f1"], m["accuracy"], -round(sum(row["multipliers"]), 8),
            tuple(-v for v in row["multipliers"]))
