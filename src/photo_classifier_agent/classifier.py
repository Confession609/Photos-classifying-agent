"""Pluggable classification backends.

The OpenCLIP backend is intentionally lazy-loaded so dataset and report tools
remain usable without downloading model weights.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Protocol

from .categories import CATEGORIES
from .hierarchy import HIERARCHY_NODES, make_hierarchical_decision, routed_leaf_scores
from .schemas import CandidateScore, Prediction


def image_id_for(path: str | Path) -> str:
    return hashlib.sha1(str(Path(path).resolve()).encode("utf-8")).hexdigest()[:16]


class ImageClassifier(Protocol):
    model_name: str
    model_version: str

    def classify(self, image_path: str | Path) -> Prediction:
        ...


class StaticClassifier:
    """Deterministic backend useful for tests and pipeline dry runs."""

    model_name = "static"
    model_version = "static-1"

    def __init__(self, scores: dict[str, float]):
        expected = {category.name for category in CATEGORIES}
        if set(scores) != expected:
            raise ValueError(f"static scores must contain exactly {sorted(expected)}")
        self.scores = scores

    def classify(self, image_path: str | Path) -> Prediction:
        path = Path(image_path).resolve()
        return Prediction(
            image_id=image_id_for(path),
            source_path=str(path),
            scores=tuple(CandidateScore(name, float(self.scores[name])) for name in self.scores),
            model_name=self.model_name,
            model_version=self.model_version,
        )


class OpenCLIPZeroShotClassifier:
    """OpenCLIP zero-shot classifier using the project's category prompts."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str | None = None,
    ) -> None:
        project_root = Path(__file__).resolve().parents[2]
        project_cache = project_root / "algorithms" / "models" / "huggingface"
        os.environ.setdefault("HF_HOME", str(project_cache))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(project_cache / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(project_cache / "transformers"))
        os.environ.setdefault("HF_MODULES_CACHE", str(project_cache / "modules"))
        try:
            import open_clip
            import torch
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "OpenCLIP backend requires optional dependencies; "
                "install with `python -m pip install -e .[vision]`"
            ) from exc

        self._torch = torch
        self._image = Image
        self.model_name = f"open_clip:{model_name}"
        self.model_version = f"{model_name}:{pretrained}"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

        prompts = [prompt for category in CATEGORIES for prompt in category.prompts]
        with torch.no_grad():
            tokens = self.tokenizer(prompts).to(self.device)
            features = self.model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)
            grouped = []
            offset = 0
            for category in CATEGORIES:
                count = len(category.prompts)
                vector = features[offset:offset + count].mean(dim=0)
                grouped.append(vector / vector.norm())
                offset += count
            self.text_features = torch.stack(grouped)

    def classify(self, image_path: str | Path) -> Prediction:
        path = Path(image_path).resolve()
        with self._image.open(path) as image:
            image_tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = 100 * image_features @ self.text_features.T
            probabilities = self._torch.softmax(logits, dim=-1)[0].tolist()
        scores = tuple(
            CandidateScore(category.name, float(score))
            for category, score in zip(CATEGORIES, probabilities, strict=True)
        )
        return Prediction(
            image_id=image_id_for(path),
            source_path=str(path),
            scores=scores,
            model_name=self.model_name,
            model_version=self.model_version,
        )


class OpenCLIPLinearProbeClassifier:
    """Load the trained project head on top of its frozen OpenCLIP backbone."""

    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
        try:
            import open_clip
            import torch
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "The trained OpenCLIP classifier requires the optional vision dependencies."
            ) from exc

        project_root = Path(__file__).resolve().parents[2]
        project_cache = project_root / "algorithms" / "models" / "huggingface"
        os.environ.setdefault("HF_HOME", str(project_cache))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(project_cache / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(project_cache / "transformers"))
        os.environ.setdefault("HF_MODULES_CACHE", str(project_cache / "modules"))

        self._torch = torch
        self._image = Image
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if tuple(checkpoint.get("class_names", ())) != tuple(category.name for category in CATEGORIES):
            raise ValueError("checkpoint categories do not match the project's configured categories")
        self.model_name = str(checkpoint["model_name"])
        self.pretrained = str(checkpoint["pretrained"])
        self.model_version = f"{self.model_name}:{self.pretrained}:linear-probe"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=self.device,
        )
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()
        self.head = torch.nn.Linear(int(checkpoint["input_dim"]), len(CATEGORIES))
        self.head.load_state_dict(checkpoint["head_state_dict"])
        self.head.to(self.device).eval()

    def classify(self, image_path: str | Path) -> Prediction:
        path = Path(image_path).expanduser().resolve()
        with self._image.open(path) as image:
            image_tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with self._torch.inference_mode():
            features = self.model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            logits = self.head(features)
            probabilities = self._torch.softmax(logits, dim=-1)[0].cpu().tolist()
        scores = tuple(
            CandidateScore(category.name, float(score))
            for category, score in zip(CATEGORIES, probabilities, strict=True)
        )
        return Prediction(
            image_id=image_id_for(path),
            source_path=str(path),
            scores=scores,
            model_name=f"open_clip_linear_probe:{self.model_name}",
            model_version=self.model_version,
        )


class OpenCLIPHierarchicalClassifier:
    """Run the three independently trained binary nodes as a cascade."""

    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
        try:
            import open_clip
            import torch
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("The hierarchical classifier requires the optional vision dependencies.") from exc

        project_root = Path(__file__).resolve().parents[2]
        project_cache = project_root / "algorithms" / "models" / "huggingface"
        os.environ.setdefault("HF_HOME", str(project_cache))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(project_cache / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(project_cache / "transformers"))
        os.environ.setdefault("HF_MODULES_CACHE", str(project_cache / "modules"))

        self._torch = torch
        self._image = Image
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        expected_nodes = tuple(node.name for node in HIERARCHY_NODES)
        if tuple(checkpoint.get("node_order", ())) != expected_nodes:
            raise ValueError("checkpoint hierarchy does not match the configured cascade")
        self.model_name = str(checkpoint["model_name"])
        self.pretrained = str(checkpoint["pretrained"])
        variant = checkpoint.get("variant")
        suffix = f":{variant}" if variant else ""
        self.model_version = f"{self.model_name}:{self.pretrained}:hierarchical{suffix}"
        self.backbone, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self.device
        )
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.heads = {}
        self.rbf_heads = {}
        for node in HIERARCHY_NODES:
            node_data = checkpoint["heads"][node.name]
            expected_labels = (node.positive_label, node.negative_label)
            if tuple(node_data.get("class_names", ())) != expected_labels:
                raise ValueError(f"checkpoint labels do not match node {node.name}")
            head_type = node_data.get("head_type", "linear")
            if head_type == "rbf_kernel_ridge":
                if node.name != "still_vs_landscape":
                    raise ValueError("RBF kernel heads are only supported at the third hierarchy node")
                train_features = node_data["train_features"].float()
                alpha = node_data["alpha"].float().reshape(-1)
                if train_features.ndim != 2 or train_features.shape[1] != int(checkpoint["input_dim"]):
                    raise ValueError("RBF training features do not match checkpoint input dimension")
                if train_features.shape[0] != alpha.numel():
                    raise ValueError("RBF coefficients do not match the number of training features")
                self.rbf_heads[node.name] = {
                    "train_features": train_features.to(self.device),
                    "alpha": alpha.to(self.device),
                    "gamma": float(node_data["gamma"]),
                }
                continue
            if head_type != "linear":
                raise ValueError(f"unsupported head type for {node.name}: {head_type}")
            head = torch.nn.Linear(int(checkpoint["input_dim"]), 2)
            head.load_state_dict(node_data["state_dict"])
            self.heads[node.name] = head.to(self.device).eval()

    def _positive_probability(self, node_name: str, features):
        torch = self._torch
        if node_name in self.rbf_heads:
            rbf = self.rbf_heads[node_name]
            distances = (2.0 - 2.0 * (features @ rbf["train_features"].T)).clamp_min(0.0)
            scores = torch.exp(-rbf["gamma"] * distances) @ rbf["alpha"]
            logits = torch.stack((scores, -scores), dim=-1)
        else:
            logits = self.heads[node_name](features)
        return torch.softmax(logits, dim=-1)[0, 0]

    def classify_with_decision(self, image_path: str | Path):
        path = Path(image_path).expanduser().resolve()
        with self._image.open(path) as image:
            image_tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with self._torch.inference_mode():
            features = self.backbone.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            person_probability = float(self._positive_probability("person_gate", features))
            # Later experts are invoked only when the previous gate routes onward.
            sky_probability = 0.5
            still_probability = 0.5
            if person_probability < 0.5:
                sky_probability = float(self._positive_probability("sky_gate", features))
                if sky_probability < 0.5:
                    still_probability = float(self._positive_probability("still_vs_landscape", features))
        decision = make_hierarchical_decision(
            image_id_for(path),
            str(path),
            person_probability,
            sky_probability,
            still_probability,
            classifier_version=self.model_version,
        )
        leaf_scores = routed_leaf_scores(
            decision.final_category,
            person_probability,
            sky_probability,
            still_probability,
        )
        prediction = Prediction(
            image_id=image_id_for(path),
            source_path=str(path),
            scores=tuple(CandidateScore(category.name, leaf_scores[category.name]) for category in CATEGORIES),
            model_name=f"open_clip_hierarchical:{self.model_name}",
            model_version=self.model_version,
        )
        return prediction, decision

    def classify(self, image_path: str | Path) -> Prediction:
        prediction, _ = self.classify_with_decision(image_path)
        return prediction
