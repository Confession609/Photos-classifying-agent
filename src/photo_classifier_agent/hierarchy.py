"""Definitions and manifest projection for the staged photo taxonomy."""

from __future__ import annotations

import json
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .categories import CATEGORY_NAMES
from .dataset import SUPPORTED_EXTENSIONS, _content_hash
from .schemas import ClassificationDecision


@dataclass(frozen=True)
class BinaryNode:
    name: str
    positive_label: str
    negative_label: str
    eligible_leaf_categories: tuple[str, ...]


HIERARCHY_NODES: tuple[BinaryNode, ...] = (
    BinaryNode("person_gate", "人像", "非人像", CATEGORY_NAMES),
    BinaryNode("sky_gate", "星空", "非星空", ("风光摄影", "星空", "静物摄影")),
    BinaryNode("still_vs_landscape", "静物摄影", "风光摄影", ("风光摄影", "静物摄影")),
)
NODE_BY_NAME = {node.name: node for node in HIERARCHY_NODES}

HIERARCHICAL_LEAF_PATHS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("人像",), "人像"),
    (("非人像", "星空"), "星空"),
    (("非人像", "非星空", "风光摄影"), "风光摄影"),
    (("非人像", "非星空", "静物摄影"), "静物摄影"),
)


def build_hierarchical_manifest(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    previous_manifest: str | Path | None = None,
    new_sample_split: str = "train",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Scan the user's nested hierarchy, preserving prior splits by content hash."""
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"hierarchical dataset directory does not exist: {root}")
    if new_sample_split not in {"train", "validation", "test"}:
        raise ValueError("new_sample_split must be train, validation, or test")

    prior_splits: dict[str, str] = {}
    if previous_manifest is not None:
        previous_path = Path(previous_manifest).expanduser().resolve()
        with previous_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                digest = record.get("content_hash")
                split = record.get("split")
                if not digest or split not in {"train", "validation", "test"}:
                    raise ValueError(f"invalid old split-manifest entry on line {line_number}")
                prior = prior_splits.setdefault(digest, split)
                if prior != split:
                    raise ValueError(f"old manifest assigns one content hash to multiple splits: {digest}")

    records: list[dict[str, Any]] = []
    seen_content: dict[str, tuple[str, str]] = {}
    duplicate_files = 0
    preserved = 0
    newly_assigned = 0
    category_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    for relative_parts, category in HIERARCHICAL_LEAF_PATHS:
        category_root = root.joinpath(*relative_parts)
        if not category_root.is_dir():
            raise FileNotFoundError(f"expected hierarchy folder is missing: {category_root}")
        for path in sorted(category_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            content_hash = _content_hash(path)
            if content_hash in seen_content:
                prior_category, prior_path = seen_content[content_hash]
                if prior_category != category:
                    raise ValueError(
                        "identical image content appears under conflicting leaf labels: "
                        f"{prior_path} ({prior_category}) and {path} ({category})"
                    )
                duplicate_files += 1
                continue
            seen_content[content_hash] = (category, str(path))
            split = prior_splits.get(content_hash)
            if split is None:
                split = new_sample_split
                newly_assigned += 1
            else:
                preserved += 1
            resolved = path.resolve()
            image_id = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:16]
            records.append({
                "image_id": image_id,
                "path": str(resolved),
                "category": category,
                "content_hash": content_hash,
                "split": split,
            })
            category_counts[category] += 1
            split_counts[split] += 1
    if not records:
        raise ValueError("no supported images found in the hierarchical dataset")
    if set(category_counts) != set(CATEGORY_NAMES):
        raise ValueError(f"hierarchy must contain all leaf classes; found {sorted(category_counts)}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "dataset_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "manifest": str(manifest_path),
        "samples": len(records),
        "category_counts": dict(category_counts),
        "split_counts": dict(split_counts),
        "preserved_prior_splits": preserved,
        "new_samples_assigned_to": new_sample_split,
        "new_samples": newly_assigned,
        "duplicate_files_skipped": duplicate_files,
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    return records, summary


def target_for(node: BinaryNode, leaf_category: str) -> str | None:
    if leaf_category not in node.eligible_leaf_categories:
        return None
    if node.name == "person_gate":
        return node.positive_label if leaf_category == "人像" else node.negative_label
    if node.name == "sky_gate":
        return node.positive_label if leaf_category == "星空" else node.negative_label
    if node.name == "still_vs_landscape":
        return leaf_category
    raise ValueError(f"unknown hierarchy node: {node.name}")


def route_category(person_probability: float, sky_probability: float, still_probability: float) -> str:
    """Apply hard 0.5 gates in order; later nodes are conditional experts."""
    if person_probability >= 0.5:
        return "人像"
    if sky_probability >= 0.5:
        return "星空"
    if still_probability >= 0.5:
        return "静物摄影"
    return "风光摄影"


def leaf_probabilities(person_probability: float, sky_probability: float, still_probability: float) -> dict[str, float]:
    """Compose normalized leaf probabilities from the three conditional gates."""
    non_person = 1.0 - person_probability
    non_sky = 1.0 - sky_probability
    return {
        "人像": person_probability,
        "风光摄影": non_person * non_sky * (1.0 - still_probability),
        "星空": non_person * sky_probability,
        "静物摄影": non_person * non_sky * still_probability,
    }


def routed_leaf_scores(
    category: str,
    person_probability: float,
    sky_probability: float,
    still_probability: float,
) -> dict[str, float]:
    """Return a normalized score vector whose winner follows the hard cascade.

    Scores are conditional on the terminal node (excluded branches receive zero),
    while the full path confidence is recorded on the final decision.
    """
    if category == "人像":
        remainder = (1.0 - person_probability) / 3.0
        return {"人像": person_probability, "风光摄影": remainder, "星空": remainder, "静物摄影": remainder}
    if category == "星空":
        remainder = (1.0 - sky_probability) / 2.0
        return {"人像": 0.0, "风光摄影": remainder, "星空": sky_probability, "静物摄影": remainder}
    if category == "静物摄影":
        return {"人像": 0.0, "风光摄影": 1.0 - still_probability, "星空": 0.0, "静物摄影": still_probability}
    if category == "风光摄影":
        return {"人像": 0.0, "风光摄影": 1.0 - still_probability, "星空": 0.0, "静物摄影": still_probability}
    raise ValueError(f"unknown routed category: {category}")


def make_hierarchical_decision(
    image_id: str,
    source_path: str,
    person_probability: float,
    sky_probability: float,
    still_probability: float,
    *,
    classifier_version: str,
    confidence_threshold: float = 0.75,
    margin_threshold: float = 0.15,
) -> ClassificationDecision:
    """Create the final decision from sequential node outputs and route margins."""
    category = route_category(person_probability, sky_probability, still_probability)
    visited: list[tuple[str, float]] = [("人像门控", person_probability if category == "人像" else 1.0 - person_probability)]
    if category != "人像":
        visited.append(("星空门控", sky_probability if category == "星空" else 1.0 - sky_probability))
    if category not in {"人像", "星空"}:
        visited.append(("静物/风光门控", still_probability if category == "静物摄影" else 1.0 - still_probability))
    path_probability = 1.0
    reasons: list[str] = []
    evidence: list[str] = []
    for node_name, selected_probability in visited:
        path_probability *= selected_probability
        evidence.append(f"{node_name}选择当前分支（{selected_probability:.3f}）")
        if selected_probability < confidence_threshold:
            reasons.append(f"{node_name}置信度低于{confidence_threshold:.2f}")
        if abs(2.0 * selected_probability - 1.0) < margin_threshold:
            reasons.append(f"{node_name}两类差距小于{margin_threshold:.2f}")
    return ClassificationDecision(
        image_id=image_id,
        source_path=source_path,
        final_category=category,
        confidence=path_probability,
        evidence=tuple(evidence),
        review_required=bool(reasons),
        review_reason="；".join(reasons) if reasons else None,
        classifier_version=classifier_version,
    )


def project_manifest_to_nodes(
    records: list[dict[str, Any]], output_dir: str | Path
) -> dict[str, dict[str, Any]]:
    """Write binary node manifests while preserving source IDs and split membership."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, Any]] = {}
    for node in HIERARCHY_NODES:
        projected: list[dict[str, Any]] = []
        by_split: dict[str, Counter[str]] = {
            "train": Counter(), "validation": Counter(), "test": Counter()
        }
        for record in records:
            label = target_for(node, record["category"])
            if label is None:
                continue
            projected.append(record | {"leaf_category": record["category"], "target_label": label, "node": node.name})
            by_split[record["split"]][label] += 1
        if not projected:
            raise ValueError(f"no samples eligible for hierarchy node {node.name}")
        manifest_path = output / f"{node.name}.jsonl"
        with manifest_path.open("w", encoding="utf-8") as handle:
            for record in projected:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        summaries[node.name] = {
            "manifest": str(manifest_path),
            "samples": len(projected),
            "splits": {split: dict(counts) for split, counts in by_split.items()},
        }
    return summaries
