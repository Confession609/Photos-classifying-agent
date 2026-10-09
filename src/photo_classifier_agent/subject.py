"""Pretrained subject analysis for the v2 cascade.

Florence-2 supplies a human-readable caption and Grounding DINO supplies
candidate regions.  Both models are optional at import time and are loaded
only when the subject-report command is run.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .schemas import ContextSignals, SubjectRegion, VisionReport


def _image_id(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]


def _model_root() -> Path:
    return Path(__file__).resolve().parents[2] / "algorithms" / "models"


def _hf_cache_env() -> None:
    cache = _model_root() / "huggingface"
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache / "transformers"))
    os.environ.setdefault("HF_MODULES_CACHE", str(cache / "modules"))


class SubjectAnalyzer:
    """Run local Florence-2 and Grounding DINO analysis for one image."""

    DEFAULT_QUERIES = (
        "person. human silhouette. mountain. mountain range. forest. tree. lake. "
        "food. meat. fruit. sculpture. statue. fireworks. rocket. airplane. stage. athlete."
    )
    CATEGORY_QUERIES = {
        "人像": "person. human silhouette. man. woman. child.",
        "风光摄影": "mountain. mountain range. landscape. forest. tree. lake. waterfall. river. coast. city skyline. building.",
        "静物摄影": "food. dish. meat. vegetable. fruit. product. object. sculpture. statue. artwork. vase.",
        "活动事件摄影": "fireworks. rocket. space shuttle. airplane. launch. stage. athlete. performer. crowd. event.",
    }
    CAPTION_TERMS = (
        "silhouette", "person", "man", "woman", "child", "mountain", "mountain range", "forest", "tree",
        "lake", "waterfall", "river", "coast", "building", "city", "food", "meat", "vegetable", "fruit",
        "dish", "plate", "sculpture", "statue", "artwork", "vase", "fireworks", "rocket", "space shuttle",
        "airplane", "launch", "athlete", "performer", "stage",
    )
    CAPTION_SUPPORT = {
        "人像": {
            "person": ("person", "people", "man", "men", "woman", "women", "child", "boy", "girl", "silhouette"),
            "man": ("man", "men", "male", "person", "silhouette"),
            "woman": ("woman", "women", "female", "person", "silhouette"),
            "child": ("child", "children", "boy", "girl", "kid"),
            "human silhouette": ("silhouette", "person", "man", "woman"),
        },
        "风光摄影": {
            "mountain": ("mountain", "mountains", "mountain range", "peak", "hill"),
            "mountain range": ("mountain", "mountains", "mountain range", "peak", "hill"),
            "landscape": ("landscape", "scenery", "view", "mountain", "forest", "lake", "river", "coast"),
            "forest": ("forest", "trees", "woods", "pine"),
            "tree": ("tree", "trees", "forest", "pine", "oak"),
            "lake": ("lake", "water", "river", "shore"),
            "waterfall": ("waterfall", "falls", "cascade"),
            "river": ("river", "stream", "water"),
            "coast": ("coast", "shore", "ocean", "sea", "beach"),
            "building": ("building", "architecture", "city", "house", "church", "tower"),
            "city skyline": ("city", "skyline", "buildings", "urban"),
        },
        "静物摄影": {
            "food": ("food", "dish", "meal", "salad", "meat", "vegetable", "fruit", "pepper", "plate"),
            "dish": ("dish", "food", "meal", "plate", "salad"),
            "meat": ("meat", "beef", "steak", "chicken", "pork", "food"),
            "vegetable": ("vegetable", "vegetables", "pepper", "lettuce", "salad", "food"),
            "fruit": ("fruit", "apple", "orange", "grape", "food"),
            "plate": ("plate", "dish", "food", "meal"),
            "product": ("product", "bottle", "device", "object", "item"),
            "object": ("object", "item", "product", "artifact", "sculpture", "statue"),
            "sculpture": ("sculpture", "statue", "artwork", "art", "figure"),
            "statue": ("statue", "sculpture", "artwork", "art", "figure"),
            "artwork": ("artwork", "art", "sculpture", "painting", "statue"),
            "vase": ("vase", "flower pot", "pottery", "container"),
        },
        "活动事件摄影": {
            "fireworks": ("firework", "fireworks", "pyrotechnic"),
            "rocket": ("rocket", "space shuttle", "launch", "missile"),
            "space shuttle": ("space shuttle", "shuttle", "rocket", "launch"),
            "airplane": ("airplane", "aircraft", "plane", "jet"),
            "launch": ("launch", "rocket", "space shuttle", "liftoff"),
            "stage": ("stage", "concert", "performance", "performer", "show"),
            "athlete": ("athlete", "sport", "player", "runner", "cyclist"),
            "performer": ("performer", "singer", "dancer", "performance", "stage"),
            "crowd": ("crowd", "audience", "spectators", "people"),
        },
    }

    def __init__(
        self,
        *,
        device: str | None = None,
        grounding_threshold: float = 0.30,
        text_threshold: float = 0.25,
        use_florence: bool = True,
        use_grounding: bool = True,
        florence_path: str | Path | None = None,
        grounding_path: str | Path | None = None,
    ) -> None:
        _hf_cache_env()
        try:
            import torch
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("subject analysis requires torch and Pillow") from exc
        self.torch = torch
        self.Image = Image
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.grounding_threshold = grounding_threshold
        self.text_threshold = text_threshold
        self.use_florence = use_florence
        self.use_grounding = use_grounding
        self.florence = None
        self.florence_processor = None
        self.grounding = None
        self.grounding_processor = None
        self.florence_path = Path(florence_path) if florence_path else _model_root() / "microsoft__Florence-2-base-ft"
        self.grounding_path = Path(grounding_path) if grounding_path else _model_root() / "IDEA-Research__grounding-dino-tiny"

    def _load_florence(self) -> None:
        if self.florence is not None or not self.use_florence:
            return
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.florence_processor = AutoProcessor.from_pretrained(
            str(self.florence_path), trust_remote_code=True, local_files_only=True
        )
        self.florence = AutoModelForCausalLM.from_pretrained(
            str(self.florence_path), trust_remote_code=True, local_files_only=True
        ).to(self.device)
        self.florence.eval()

    def _load_grounding(self) -> None:
        if self.grounding is not None or not self.use_grounding:
            return
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.grounding_processor = AutoProcessor.from_pretrained(
            str(self.grounding_path), local_files_only=True
        )
        self.grounding = AutoModelForZeroShotObjectDetection.from_pretrained(
            str(self.grounding_path), local_files_only=True
        ).to(self.device)
        self.grounding.eval()

    def _caption(self, image: Any) -> str | None:
        if not self.use_florence:
            return None
        self._load_florence()
        prompt = "<MORE_DETAILED_CAPTION>"
        inputs = self.florence_processor(text=prompt, images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with self.torch.inference_mode():
            generated = self.florence.generate(
                input_ids=inputs.get("input_ids"),
                pixel_values=inputs.get("pixel_values"),
                max_new_tokens=96,
                num_beams=3,
                do_sample=False,
            )
        text = self.florence_processor.batch_decode(generated, skip_special_tokens=False)[0]
        try:
            parsed = self.florence_processor.post_process_generation(text, task=prompt, image_size=image.size)
            if isinstance(parsed, dict):
                value = parsed.get(prompt)
                if isinstance(value, str):
                    return value.strip()
        except Exception:
            pass
        return text.replace(prompt, "").replace("<s>", "").replace("</s>", "").strip()

    def _regions(self, image: Any, caption: str | None = None, category: str | None = None) -> tuple[SubjectRegion, ...]:
        if not self.use_grounding:
            return ()
        self._load_grounding()
        query = self.CATEGORY_QUERIES.get(category, self.DEFAULT_QUERIES)
        caption_lower = (caption or "").lower()
        caption_terms = [term for term in self.CAPTION_TERMS if term in caption_lower]
        if caption_terms:
            existing = {part.strip().rstrip(".").lower() for part in query.split(".") if part.strip()}
            extra = [term for term in caption_terms if term not in existing]
            if extra:
                query = query.rstrip() + " " + ". ".join(extra) + "."
        inputs = self.grounding_processor(images=image, text=query, return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with self.torch.inference_mode():
            outputs = self.grounding(**inputs)
        target_sizes = [image.size[::-1]]
        processed = self.grounding_processor.post_process_grounded_object_detection(
            outputs,
            inputs.get("input_ids"),
            threshold=self.grounding_threshold,
            text_threshold=self.text_threshold,
            target_sizes=target_sizes,
        )[0]
        regions: list[SubjectRegion] = []
        width, height = image.size
        boxes = processed.get("boxes", [])
        scores = processed.get("scores", [])
        labels = processed.get("text_labels", processed.get("labels", []))
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = [float(value) for value in box.tolist()]
            x1, x2 = sorted((max(0.0, min(width, x1)), max(0.0, min(width, x2))))
            y1, y2 = sorted((max(0.0, min(height, y1)), max(0.0, min(height, y2))))
            if x2 <= x1 or y2 <= y1:
                continue
            regions.append(SubjectRegion(
                x=x1 / width,
                y=y1 / height,
                width=(x2 - x1) / width,
                height=(y2 - y1) / height,
                label=str(label),
                confidence=float(score),
            ))
        return tuple(regions)

    @staticmethod
    def _choose_region(
        regions: tuple[SubjectRegion, ...],
        category: str | None = None,
        caption: str | None = None,
    ) -> SubjectRegion | None:
        if not regions:
            return None
        # Near-full-frame detections are frequently false positives. Keep the
        # candidates in the report, but do not call one a usable subject crop.
        useful = [
            region for region in regions
            if region.width * region.height <= 0.85 and region.width <= 0.97 and region.height <= 0.97
        ]
        if not useful:
            return None
        excluded = {"风光摄影": {"sky", "cloud"}}.get(category, set())
        useful = [region for region in useful if (region.label or "").lower() not in excluded]
        if not useful:
            return None
        category_hints = {
            "人像": {"person", "man", "woman", "child", "human silhouette", "silhouette"},
            "风光摄影": {"mountain", "mountain range", "landscape", "forest", "tree", "lake", "waterfall", "river", "coast", "building", "city skyline"},
            "静物摄影": {"food", "dish", "meat", "vegetable", "fruit", "product", "object", "sculpture", "statue", "artwork", "vase"},
            "活动事件摄影": {"fireworks", "rocket", "space shuttle", "airplane", "launch", "stage", "athlete", "performer", "crowd", "event"},
        }.get(category, set())
        caption_lower = (caption or "").lower()

        if category and caption_lower:
            support_map = SubjectAnalyzer.CAPTION_SUPPORT.get(category, {})

            def supported(region: SubjectRegion) -> bool:
                label = (region.label or "").lower().strip().rstrip(".")
                matching_keys = [key for key in support_map if key in label]
                aliases = tuple(alias for key in matching_keys for alias in support_map[key]) or (label,)
                return any(alias in caption_lower for alias in aliases)

            caption_supported = [region for region in useful if supported(region)]
            if not caption_supported:
                return None
            useful = caption_supported

        def rank(region: SubjectRegion) -> tuple[float, float, float]:
            label = (region.label or "").lower().strip().rstrip(".")
            class_match = 1.0 if any(hint in label for hint in category_hints) else 0.0
            caption_match = 1.0 if label and label in caption_lower else 0.0
            # Among semantically supported proposals, prefer a box that covers
            # the complete described subject; confidence breaks similar-area ties.
            return (caption_match, class_match, region.width * region.height, float(region.confidence or 0.0))
        return max(
            useful,
            key=rank,
        )

    def _write_crop(self, image: Any, region: SubjectRegion, crop_dir: Path, image_id: str) -> str:
        crop_dir.mkdir(parents=True, exist_ok=True)
        width, height = image.size
        pad_x = region.width * 0.08
        pad_y = region.height * 0.08
        left = max(0, int((region.x - pad_x) * width))
        top = max(0, int((region.y - pad_y) * height))
        right = min(width, int((region.x + region.width + pad_x) * width))
        bottom = min(height, int((region.y + region.height + pad_y) * height))
        crop = image.crop((left, top, right, bottom))
        output = crop_dir / f"{image_id}.jpg"
        crop.convert("RGB").save(output, quality=92, optimize=True)
        return str(output.resolve())

    def analyze(self, image_path: str | Path, crop_dir: str | Path, *, category: str | None = None) -> VisionReport:
        path = Path(image_path).expanduser().resolve()
        image_id = _image_id(path)
        with self.Image.open(path) as source:
            image = source.convert("RGB")
            caption = self._caption(image)
            regions = self._regions(image, caption, category)
            selected = self._choose_region(regions, category, caption)
            crop_path = self._write_crop(image, selected, Path(crop_dir), image_id) if selected else None
        people = [region for region in regions if "person" in (region.label or "").lower()]
        return VisionReport(
            image_id=image_id,
            source_path=str(path),
            main_subject_description=caption,
            subject_regions=regions,
            context=ContextSignals(
                people_present=bool(people),
                people_is_primary=bool(selected and "person" in (selected.label or "").lower()),
                night_sky_present=any("sky" in (region.label or "").lower() for region in regions),
            ),
            subject_crop_path=crop_path,
            report_complete=True,
            model_name="Florence-2+GroundingDINO",
            model_version="Florence-2-base-ft+grounding-dino-tiny:category-guided-v4",
        )


def write_subject_reports(
    records: Iterable[dict[str, Any]],
    output_dir: str | Path,
    *,
    device: str | None = None,
    max_images: int | None = None,
    use_florence: bool = True,
    use_grounding: bool = True,
) -> dict[str, Any]:
    """Generate JSONL reports strictly for manifest records."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    crop_dir = output / "crops"
    report_path = output / "subject_reports.jsonl"
    analyzer = SubjectAnalyzer(device=device, use_florence=use_florence, use_grounding=use_grounding)
    selected = list(records)
    if max_images is not None:
        selected = selected[:max_images]
    failures = []
    count = 0
    with report_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(selected, 1):
            try:
                report = analyzer.analyze(record["path"], crop_dir)
                handle.write(json.dumps(asdict(report), ensure_ascii=False) + "\n")
                count += 1
                print(f"subject reports {index}/{len(selected)}", flush=True)
            except Exception as exc:  # keep the manifest pipeline resumable
                failures.append({"image_id": record.get("image_id"), "path": record.get("path"), "error": repr(exc)})
                print(f"subject report failed: {record.get('path')}: {exc}", flush=True)
    (output / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"reports": str(report_path), "crops": str(crop_dir), "processed": count, "failures": len(failures)}


def load_subject_reports(path: str | Path) -> dict[str, dict[str, Any]]:
    report_path = Path(path).expanduser().resolve()
    result: dict[str, dict[str, Any]] = {}
    if not report_path.is_file():
        return result
    for line in report_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[row["image_id"]] = row
    return result
