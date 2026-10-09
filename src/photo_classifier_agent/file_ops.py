"""Safe, copy-only application of approved classification decisions."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from .categories import CATEGORY_NAMES
from .schemas import ClassificationDecision


@dataclass(frozen=True)
class CopyResult:
    image_id: str
    source_path: str
    destination_path: str | None
    status: str
    reason: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _collision_safe_path(directory: Path, filename: str, source_hash: str) -> tuple[Path, str]:
    candidate = directory / filename
    if not candidate.exists():
        return candidate, "copied"
    if candidate.is_file() and _sha256(candidate) == source_hash:
        return candidate, "skipped_identical"
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    index = 1
    while True:
        candidate = directory / f"{stem} ({index}){suffix}"
        if not candidate.exists():
            return candidate, "copied_collision_renamed"
        index += 1


def apply_decisions(
    decisions: list[ClassificationDecision],
    output_root: str | Path,
    include_review: bool = False,
) -> list[CopyResult]:
    destination_root = Path(output_root).expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    results: list[CopyResult] = []
    for decision in decisions:
        source = Path(decision.source_path).expanduser().resolve()
        if not source.is_file():
            results.append(CopyResult(decision.image_id, str(source), None, "missing_source", "源文件不存在"))
            continue
        if decision.final_category not in CATEGORY_NAMES:
            results.append(CopyResult(decision.image_id, str(source), None, "invalid_category", "类别不在固定列表中"))
            continue
        if decision.review_required and not include_review:
            results.append(CopyResult(decision.image_id, str(source), None, "review_skipped", decision.review_reason))
            continue

        category_dir = destination_root / decision.final_category
        category_dir.mkdir(parents=True, exist_ok=True)
        source_hash = _sha256(source)
        destination, status = _collision_safe_path(category_dir, source.name, source_hash)
        if destination.resolve() == source:
            results.append(CopyResult(decision.image_id, str(source), str(destination), "same_file_skipped"))
            continue
        if status != "skipped_identical":
            shutil.copy2(source, destination)
        results.append(CopyResult(decision.image_id, str(source), str(destination), status))

    manifest = destination_root / "copy_manifest.jsonl"
    with manifest.open("a", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
    return results

