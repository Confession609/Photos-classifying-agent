"""Serializable domain objects shared by scanners, models, and reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class CandidateScore:
    category: str
    score: float


@dataclass(frozen=True)
class SubjectRegion:
    x: float
    y: float
    width: float
    height: float
    label: str | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class ContextSignals:
    people_present: bool = False
    people_is_primary: bool = False
    night_sky_present: bool = False
    natural_subject_present: bool = False
    artificial_object_present: bool = False


@dataclass(frozen=True)
class PhotographicCues:
    background_blur: str | None = None
    exposure_emphasis: str | None = None
    color_saturation_emphasis: str | None = None
    composition_emphasis: str | None = None


@dataclass(frozen=True)
class Prediction:
    image_id: str
    source_path: str
    scores: tuple[CandidateScore, ...]
    model_name: str
    model_version: str

    @property
    def top_category(self) -> str:
        if not self.scores:
            raise ValueError("prediction has no category scores")
        return max(self.scores, key=lambda item: item.score).category

    @property
    def top_score(self) -> float:
        if not self.scores:
            return 0.0
        return max(self.scores, key=lambda item: item.score).score

    @property
    def margin(self) -> float:
        ordered = sorted((item.score for item in self.scores), reverse=True)
        return ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]


@dataclass(frozen=True)
class VisionReport:
    image_id: str
    source_path: str
    main_subject_description: str | None = None
    subject_regions: tuple[SubjectRegion, ...] = ()
    context: ContextSignals = field(default_factory=ContextSignals)
    photographic_cues: PhotographicCues = field(default_factory=PhotographicCues)
    subject_crop_path: str | None = None
    report_complete: bool = False
    model_name: str = "unavailable"
    model_version: str = "unavailable"


@dataclass(frozen=True)
class ClassificationDecision:
    image_id: str
    source_path: str
    final_category: str
    confidence: float
    evidence: tuple[str, ...] = ()
    review_required: bool = False
    review_reason: str | None = None
    classifier_version: str = "unknown"


def to_dict(value: Any) -> dict[str, Any]:
    """Convert one of the public dataclasses to JSON-compatible data."""

    return asdict(value)

