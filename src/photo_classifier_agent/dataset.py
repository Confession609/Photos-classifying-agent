"""Dataset discovery, validation, deterministic splitting, and manifests."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .categories import CATEGORY_NAMES

SUPPORTED_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".heic", ".heif"
})


@dataclass(frozen=True)
class ImageSample:
    image_id: str
    path: str
    category: str
    content_hash: str


@dataclass(frozen=True)
class DatasetSplit:
    train: tuple[ImageSample, ...]
    validation: tuple[ImageSample, ...]
    test: tuple[ImageSample, ...]


def _content_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def scan_dataset(root: str | Path, categories: Iterable[str] = CATEGORY_NAMES) -> tuple[ImageSample, ...]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {root_path}")

    allowed = set(categories)
    samples: list[ImageSample] = []
    for category in CATEGORY_NAMES:
        if category not in allowed:
            continue
        category_dir = root_path / category
        if not category_dir.is_dir():
            continue
        for path in sorted(category_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            # Keep the identity rule identical to inference backends, which do
            # not know the dataset root and therefore hash the resolved path.
            normalized_path = str(path.resolve())
            samples.append(
                ImageSample(
                    image_id=hashlib.sha1(normalized_path.encode("utf-8")).hexdigest()[:16],
                    path=str(path),
                    category=category,
                    content_hash=_content_hash(path),
                )
            )
    return tuple(samples)


def stratified_split(
    samples: Iterable[ImageSample],
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    seed: int = 1337,
) -> DatasetSplit:
    if not 0 < train_ratio < 1 or not 0 <= validation_ratio < 1:
        raise ValueError("split ratios must be between 0 and 1")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be less than 1")

    grouped: dict[str, list[ImageSample]] = {}
    seen_hashes: set[str] = set()
    for sample in samples:
        if sample.content_hash in seen_hashes:
            continue
        seen_hashes.add(sample.content_hash)
        grouped.setdefault(sample.category, []).append(sample)

    rng = random.Random(seed)
    train: list[ImageSample] = []
    validation: list[ImageSample] = []
    test: list[ImageSample] = []
    for category in CATEGORY_NAMES:
        items = grouped.get(category, [])[:]
        rng.shuffle(items)
        count = len(items)
        train_count = int(count * train_ratio)
        validation_count = int(count * validation_ratio)
        if count >= 3:
            train_count = max(1, train_count)
            validation_count = max(1, validation_count)
            if train_count + validation_count >= count:
                validation_count = max(0, count - train_count - 1)
        train.extend(items[:train_count])
        validation.extend(items[train_count:train_count + validation_count])
        test.extend(items[train_count + validation_count:])

    return DatasetSplit(tuple(train), tuple(validation), tuple(test))


def write_split_manifest(split: DatasetSplit, output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path / "dataset_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for split_name, items in (
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        ):
            for item in items:
                record = asdict(item) | {"split": split_name}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return manifest_path
