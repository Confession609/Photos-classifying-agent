"""Verify the locally extracted Release files without loading any model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(root: Path) -> dict:
    root = root.resolve()
    manifest = json.loads((root / "MODEL_MANIFEST.json").read_text(encoding="utf-8"))
    seen = set()
    for row in manifest["files"]:
        name = row["path"]
        if name in seen or PurePosixPath(name).is_absolute() or PureWindowsPath(name).drive or ".." in PurePosixPath(name).parts:
            raise ValueError("Unsafe or duplicate model manifest path")
        seen.add(name)
        path = root / name
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise FileNotFoundError(f"Missing or redirected model file: {name}")
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError(f"Model package verification failed: {name}")
    required = {"ACTIVE_WORKFLOW.json", "daily_workflow_config.json", "models/heads/heads.pt",
                "models/grounding_dino/best_model/model.safetensors", "models/open_clip/open_clip_model.safetensors"}
    if not required.issubset(seen):
        raise ValueError("Model manifest lacks required deployment files")
    return {"verified": True, "files": len(seen), "photos_included": False,
            "models_loaded": False, "version": manifest["version"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
