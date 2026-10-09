"""Version 2 cascade taxonomy and split-preserving manifests.

The v2 taxonomy intentionally keeps the old experiment untouched.  It has one
person gate and one conditional three-way non-person classifier.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .dataset import SUPPORTED_EXTENSIONS, _content_hash

V2_PERSON_LABELS = ("人像", "非人像")
V2_NON_PERSON_LABELS = ("风光摄影", "静物摄影", "活动事件摄影")
V2_LEAF_LABELS = ("人像", "风光摄影", "静物摄影", "活动事件摄影")


def _assign_new_splits(items: list[dict[str, Any]], seed: int) -> None:
    """Assign new records by label using a deterministic 70/15/15 split."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[item["category"]].append(item)
    rng = random.Random(seed)
    for category, rows in grouped.items():
        rng.shuffle(rows)
        count = len(rows)
        train_count = max(1, int(count * 0.70)) if count else 0
        validation_count = max(1, int(count * 0.15)) if count >= 3 else 0
        if train_count + validation_count >= count and count >= 3:
            validation_count = max(0, count - train_count - 1)
        for row in rows[:train_count]:
            row["split"] = "train"
        for row in rows[train_count:train_count + validation_count]:
            row["split"] = "validation"
        for row in rows[train_count + validation_count:]:
            row["split"] = "test"


def build_v2_manifest(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    previous_manifest: str | Path | None = None,
    seed: int = 1337,
) -> dict[str, Any]:
    """Scan ``人像`` and ``非人像/<leaf>`` without copying source images."""
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"v2 dataset directory does not exist: {root}")

    prior_splits: dict[str, str] = {}
    if previous_manifest:
        prior_path = Path(previous_manifest).expanduser().resolve()
        with prior_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if row.get("split") in {"train", "validation", "test"} and row.get("content_hash"):
                        prior_splits[row["content_hash"]] = row["split"]

    category_dirs = {
        "人像": root / "人像",
        **{label: root / "非人像" / label for label in V2_NON_PERSON_LABELS},
    }
    rows: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    duplicate_count = 0
    for category, category_dir in category_dirs.items():
        if not category_dir.is_dir():
            raise FileNotFoundError(f"required v2 category directory is missing: {category_dir}")
        for path in sorted(category_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            digest = _content_hash(path)
            if digest in seen:
                if seen[digest] != category:
                    raise ValueError(f"duplicate image content has conflicting labels: {path}")
                duplicate_count += 1
                continue
            seen[digest] = category
            resolved = path.resolve()
            rows.append({
                "image_id": hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:16],
                "path": str(resolved),
                "category": category,
                "content_hash": digest,
                "split": prior_splits.get(digest),
            })

    if not rows:
        raise ValueError("no supported images found for v2 taxonomy")
    new_rows = [row for row in rows if row["split"] is None]
    _assign_new_splits(new_rows, seed)
    for row in rows:
        if row["split"] is None:
            raise RuntimeError(f"split assignment failed for {row['path']}")

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "dataset_manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda value: value["image_id"]):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    counts = Counter(row["category"] for row in rows)
    splits = Counter(row["split"] for row in rows)
    return {
        "manifest": str(manifest),
        "manifest_sha256": digest,
        "samples": len(rows),
        "category_counts": dict(counts),
        "split_counts": dict(splits),
        "duplicate_files_skipped": duplicate_count,
        "previous_splits_preserved": sum(1 for row in rows if row["content_hash"] in prior_splits),
        "warning": "活动事件摄影样本少于 500 张，第二层指标会有较大不确定性。" if counts["活动事件摄影"] < 500 else None,
    }


def read_v2_manifest(path: str | Path) -> tuple[list[dict[str, Any]], str]:
    manifest = Path(path).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    ids: set[str] = set()
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        required = {"image_id", "path", "category", "content_hash", "split"}
        if not required.issubset(row):
            raise ValueError(f"v2 manifest line {line_number} is missing fields")
        if row["category"] not in V2_LEAF_LABELS or row["split"] not in {"train", "validation", "test"}:
            raise ValueError(f"invalid v2 manifest label/split on line {line_number}")
        if row["image_id"] in ids:
            raise ValueError(f"duplicate image_id in v2 manifest: {row['image_id']}")
        if row["content_hash"] in hashes and hashes[row["content_hash"]] != row["split"]:
            raise ValueError(f"duplicate content_hash crosses splits: {row['content_hash']}")
        ids.add(row["image_id"])
        hashes[row["content_hash"]] = row["split"]
        rows.append(row)
    if not rows:
        raise ValueError("v2 manifest is empty")
    return rows, hashlib.sha256(manifest.read_bytes()).hexdigest()


def project_v2_node_manifests(records: list[dict[str, Any]], output_dir: str | Path) -> dict[str, Any]:
    """Write node manifests while retaining the authoritative leaf split."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    nodes = {
        "person_gate": lambda category: "人像" if category == "人像" else "非人像",
        "non_person_classifier": lambda category: category if category != "人像" else None,
    }
    summaries: dict[str, Any] = {}
    for node, labeler in nodes.items():
        node_rows = []
        for row in records:
            target = labeler(row["category"])
            if target is not None:
                node_rows.append(row | {"target_label": target, "node": node})
        path = output / f"{node}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in node_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summaries[node] = {
            "manifest": str(path),
            "samples": len(node_rows),
            "split_counts": dict(Counter(row["split"] for row in node_rows)),
            "label_counts": dict(Counter(row["target_label"] for row in node_rows)),
        }
    return summaries
