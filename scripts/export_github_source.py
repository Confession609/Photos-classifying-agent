"""Export an allowlisted, source-only repository; never include photos/weights.

Existing destinations are refused. This command does not initialize Git,
commit, push, train, or load models. Local historical scripts remain untouched.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = (
    ".gitignore", ".gitattributes", "README.md", "pyproject.toml", "PUBLICATION.md",
    "TESTED_ENVIRONMENT.md", "MODEL_NOTICES.md", "GITHUB_UPLOAD_GUIDE.md",
    "daily_workflow_config.example.json", "ACTIVE_WORKFLOW.example.json",
    "开始照片分类.cmd", "运行照片分类.ps1",
    "algorithms/requirements-core.txt", "algorithms/requirements-vision.txt",
    "scripts/classify_daily_photos.py", "scripts/export_github_source.py",
    "scripts/verify_model_bundle.py",
    "scripts/train_grounding_dino_subject_boxes.py", "scripts/train_florence_subject_captions.py",
    "scripts/train_v2_ablation.py", "scripts/train_v2_auto_crop_heads.py",
    "scripts/generate_gd_auto_crops_for_heads.py", "scripts/evaluate_grounding_fixed_prompt.py",
    "tests/test_core.py", "tests/test_daily_workflow.py", "tests/test_daily_launcher.py",
    "tests/test_v2_ablation.py", "tests/test_github_export.py",
)
SENSITIVE_PATTERNS = (
    re.compile(r"\b[A-Za-z]:[\\/]"),
    re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16})\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def linked(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def check_source(path: Path, root: Path) -> dict:
    root = root.resolve()
    if not path.resolve().is_relative_to(root):
        raise ValueError("Export source resolves outside project")
    for parent in (path, *path.parents):
        if linked(parent):
            raise ValueError("Linked export source is forbidden")
        if parent == root:
            break
    if not path.is_file() or path.stat().st_size > 2_000_000:
        raise ValueError("Missing or unexpectedly large source file")
    content = path.read_bytes()
    text = content.decode("utf-8-sig")
    if any(pattern.search(text) for pattern in SENSITIVE_PATTERNS):
        # Do not print potentially secret matched values.
        raise ValueError(f"Privacy scan requires manual review: {path.relative_to(root)}")
    if path.suffix == ".py":
        ast.parse(text, filename=path.name)
    elif path.suffix == ".json":
        json.loads(text)
    elif path.suffix == ".cmd":
        content.decode("ascii")
        if b"\n" in content.replace(b"\r\n", b""):
            raise ValueError("Windows launcher must use CRLF")
    return {"path": path.relative_to(root).as_posix(), "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest()}


def collect_files(root: Path) -> list[Path]:
    paths = [root / relative for relative in PUBLIC_FILES]
    # walk() with explicit link rejection; do not traverse node_modules/junctions.
    pending = [root / "src"]
    while pending:
        directory = pending.pop()
        if linked(directory):
            raise ValueError("Linked source directory is forbidden")
        for entry in sorted(directory.iterdir()):
            if linked(entry):
                raise ValueError("Linked source entry is forbidden")
            if entry.is_dir() and entry.name != "__pycache__":
                pending.append(entry)
            elif entry.is_file() and entry.suffix == ".py":
                paths.append(entry)
    return sorted(paths, key=lambda p: p.relative_to(root).as_posix())


def export(root: Path, destination: Path) -> dict:
    root = root.resolve()
    destination = destination.absolute()
    if destination.parent.resolve() != root or not re.fullmatch(r"github_upload(?:_[A-Za-z0-9_-]+)?", destination.name):
        raise ValueError("Export must be a new github_upload[_suffix] directory directly in the project")
    if destination.exists() or linked(destination):
        raise FileExistsError("Destination already exists; use a new suffix, never overwrite")
    files = collect_files(root)
    inventory = [check_source(path, root) for path in files]
    destination.mkdir()
    for source, record in zip(files, inventory):
        target = destination / record["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Export copy checksum mismatch")
    return {"files": len(inventory), "bytes": sum(r["bytes"] for r in inventory),
            "photos_included": False, "weights_included": False, "private_configs_included": False,
            "heuristic_privacy_scan_passed": True, "git_initialized": False, "pushed": False,
            "destination": destination.name, "inventory": inventory}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "github_upload")
    args = parser.parse_args()
    print(json.dumps(export(ROOT, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
