"""Classification policy and review gating."""

from __future__ import annotations

from .categories import CATEGORY_NAMES
from .schemas import ClassificationDecision, Prediction, VisionReport


def decide(
    prediction: Prediction,
    report: VisionReport | None = None,
    confidence_threshold: float = 0.75,
    margin_threshold: float = 0.15,
) -> ClassificationDecision:
    """Apply the project's single-label policy to a model prediction.

    The person-first rule is applied only when the vision report explicitly says
    that people are the primary subject. A generic ``people_present`` signal is
    not enough to override the model because a tiny background figure should not
    defeat a landscape or night-sky classification.
    """

    if prediction.top_category not in CATEGORY_NAMES:
        raise ValueError(f"unknown predicted category: {prediction.top_category}")

    category = prediction.top_category
    evidence: list[str] = [f"模型最高候选为{category}（{prediction.top_score:.3f}）"]

    if report is not None and report.context.people_is_primary:
        category = "人像"
        evidence.append("视觉报告确认人物是主要主体，因此人物优先")
    elif (
        report is not None
        and report.context.night_sky_present
        and not report.context.people_is_primary
        and category in {"风光摄影", "星空"}
    ):
        category = "星空"
        evidence.append("视觉报告确认夜空是主要内容，因此归入星空")

    reasons: list[str] = []
    if prediction.top_score < confidence_threshold:
        reasons.append(f"置信度低于{confidence_threshold:.2f}")
    if prediction.margin < margin_threshold:
        reasons.append(f"第一、第二候选差距小于{margin_threshold:.2f}")
    if report is not None and not report.report_complete:
        reasons.append("视觉报告不完整")

    return ClassificationDecision(
        image_id=prediction.image_id,
        source_path=prediction.source_path,
        final_category=category,
        confidence=prediction.top_score,
        evidence=tuple(evidence),
        review_required=bool(reasons),
        review_reason="；".join(reasons) if reasons else None,
        classifier_version=prediction.model_version,
    )
