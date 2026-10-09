"""Command line entry points for the first project milestone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .classifier import OpenCLIPHierarchicalClassifier, OpenCLIPLinearProbeClassifier, OpenCLIPZeroShotClassifier
from .dataset import SUPPORTED_EXTENSIONS, scan_dataset, stratified_split, write_split_manifest
from .decision import decide
from .evaluation import evaluate, read_truth_manifest, write_evaluation
from .file_ops import apply_decisions
from .hierarchy import build_hierarchical_manifest, project_manifest_to_nodes
from .reports import read_decisions, write_decisions, write_predictions, write_review_csv, write_review_html
from .training import train_linear_probe
from .training import read_training_manifest
from .hierarchy_training import train_hierarchical_probe
from .head_experiments import (
    build_rbf_cascade_checkpoint,
    compare_third_layer_heads,
    evaluate_third_layer_heads,
)
from .hierarchy_cv import run_hierarchical_rbf_cross_validation
from .hierarchy_v2 import build_v2_manifest, project_v2_node_manifests, read_v2_manifest
from .hierarchy_v2_training import V2CascadeClassifier, train_v2
from .subject import write_subject_reports
from .subject_dataset import generate_category_subject_data


def _prepare_dataset(args: argparse.Namespace) -> int:
    samples = scan_dataset(args.input)
    split = stratified_split(samples, seed=args.seed)
    manifest = write_split_manifest(split, args.output)
    summary = {
        "total": len(samples),
        "train": len(split.train),
        "validation": len(split.validation),
        "test": len(split.test),
        "manifest": str(manifest),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _classify(args: argparse.Namespace) -> int:
    classifier = OpenCLIPZeroShotClassifier(args.model, args.pretrained, args.device)
    root = Path(args.input).expanduser().resolve()
    paths = [path for path in sorted(root.rglob("*")) if path.is_file() and path.suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".heic", ".heif"
    }]
    predictions = [classifier.classify(path) for path in paths]
    prediction_path = write_predictions(predictions, args.predictions)
    decisions = [decide(prediction) for prediction in predictions]
    decision_path = write_decisions(decisions, args.decisions)
    write_review_csv(decisions, args.csv)
    write_review_html(decisions, args.html)
    print(json.dumps({"images": len(paths), "predictions": str(prediction_path), "decisions": str(decision_path)}, ensure_ascii=False, indent=2))
    return 0


def _classify_trained(args: argparse.Namespace) -> int:
    root = Path(args.input).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {root}")
    paths = [
        path for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not paths:
        raise ValueError(f"no supported images found under: {root}")
    classifier = OpenCLIPLinearProbeClassifier(args.checkpoint, args.device)
    predictions = []
    for index, path in enumerate(paths, 1):
        predictions.append(classifier.classify(path))
        print(f"classified {index}/{len(paths)}", flush=True)
    prediction_path = write_predictions(predictions, args.predictions)
    decisions = [decide(prediction) for prediction in predictions]
    decision_path = write_decisions(decisions, args.decisions)
    csv_path = write_review_csv(decisions, args.csv)
    html_path = write_review_html(decisions, args.html)
    summary = {
        "images": len(paths),
        "predictions": str(prediction_path),
        "decisions": str(decision_path),
        "csv": str(csv_path),
        "html": str(html_path),
        "review_required": sum(decision.review_required for decision in decisions),
        "source_images_modified": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    decisions = read_decisions(args.decisions)
    truth = read_truth_manifest(args.truth_manifest)
    result = evaluate(decisions, truth)
    output = write_evaluation(result, args.output)
    print(json.dumps(result.to_dict() | {"output": str(output)}, ensure_ascii=False, indent=2))
    return 0


def _train_linear(args: argparse.Namespace) -> int:
    summary = train_linear_probe(
        args.manifest,
        args.output_dir,
        model_name=args.model,
        pretrained=args.pretrained,
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _prepare_hierarchy(args: argparse.Namespace) -> int:
    if args.input:
        records, source_summary = build_hierarchical_manifest(
            args.input,
            args.output_dir,
            previous_manifest=args.previous_manifest,
            new_sample_split=args.new_sample_split,
        )
        manifest_sha256 = source_summary["sha256"]
    else:
        records, manifest_sha256 = read_training_manifest(args.manifest)
        source_summary = {"samples": len(records), "manifest_sha256": manifest_sha256}
    summaries = project_manifest_to_nodes(records, Path(args.output_dir) / "manifests")
    print(json.dumps({"source": source_summary, "nodes": summaries}, ensure_ascii=False, indent=2))
    return 0


def _train_hierarchy(args: argparse.Namespace) -> int:
    summary = train_hierarchical_probe(
        args.manifest,
        args.output_dir,
        model_name=args.model,
        pretrained=args.pretrained,
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        seed=args.seed,
        third_head=args.third_head,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _train_hierarchy_cv(args: argparse.Namespace) -> int:
    summary = run_hierarchical_rbf_cross_validation(
        args.manifest,
        args.output_dir,
        folds=args.folds,
        seed=args.seed,
        model_name=args.model,
        pretrained=args.pretrained,
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        third_head=args.third_head,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _compare_third_heads(args: argparse.Namespace) -> int:
    summary = compare_third_layer_heads(
        args.manifest,
        args.output_dir,
        model_name=args.model,
        pretrained=args.pretrained,
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _evaluate_third_heads(args: argparse.Namespace) -> int:
    summary = evaluate_third_layer_heads(
        args.manifest,
        args.experiment_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _build_rbf_cascade(args: argparse.Namespace) -> int:
    summary = build_rbf_cascade_checkpoint(
        args.cascade_checkpoint,
        args.rbf_head_checkpoint,
        args.output_checkpoint,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _prepare_v2(args: argparse.Namespace) -> int:
    summary = build_v2_manifest(
        args.input,
        args.output_dir,
        previous_manifest=args.previous_manifest,
        seed=args.seed,
    )
    records, _ = read_v2_manifest(summary["manifest"])
    summary["nodes"] = project_v2_node_manifests(records, Path(args.output_dir) / "manifests")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _generate_subject_reports(args: argparse.Namespace) -> int:
    records, manifest_sha = read_v2_manifest(args.manifest)
    if args.split:
        records = [row for row in records if row["split"] == args.split]
    summary = write_subject_reports(
        records,
        args.output_dir,
        device=args.device,
        max_images=args.max_images,
        use_florence=not args.no_florence,
        use_grounding=not args.no_grounding,
    )
    summary["manifest_sha256"] = manifest_sha
    summary["split"] = args.split
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _train_v2(args: argparse.Namespace) -> int:
    summary = train_v2(
        args.manifest,
        args.output_dir,
        reports_path=args.reports,
        model_name=args.model,
        pretrained=args.pretrained,
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _classify_v2(args: argparse.Namespace) -> int:
    root = Path(args.input).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {root}")
    paths = [path for path in sorted(root.rglob("*")) if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not paths:
        raise ValueError(f"no supported images found under: {root}")
    classifier = V2CascadeClassifier(args.checkpoint, reports_path=args.reports, device=args.device)
    predictions = []
    decisions = []
    for index, path in enumerate(paths, 1):
        prediction, decision = classifier.classify_with_decision(path)
        predictions.append(prediction)
        decisions.append(decision)
        print(f"classified {index}/{len(paths)}", flush=True)
    prediction_path = write_predictions(predictions, args.predictions)
    decision_path = write_decisions(decisions, args.decisions)
    csv_path = write_review_csv(decisions, args.csv)
    html_path = write_review_html(decisions, args.html)
    print(json.dumps({
        "images": len(paths),
        "predictions": str(prediction_path),
        "decisions": str(decision_path),
        "csv": str(csv_path),
        "html": str(html_path),
        "review_required": sum(decision.review_required for decision in decisions),
        "subject_reports_used": bool(args.reports),
        "source_images_modified": False,
    }, ensure_ascii=False, indent=2))
    return 0


def _generate_category_subject_data(args: argparse.Namespace) -> int:
    summary = generate_category_subject_data(
        args.input_root,
        device=args.device,
        limit_per_category=args.limit_per_category,
        checkpoint_every=args.checkpoint_every,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _classify_hierarchical(args: argparse.Namespace) -> int:
    root = Path(args.input).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {root}")
    paths = [
        path for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not paths:
        raise ValueError(f"no supported images found under: {root}")
    classifier = OpenCLIPHierarchicalClassifier(args.checkpoint, args.device)
    predictions = []
    decisions = []
    for index, path in enumerate(paths, 1):
        prediction, decision = classifier.classify_with_decision(path)
        predictions.append(prediction)
        decisions.append(decision)
        print(f"classified {index}/{len(paths)}", flush=True)
    prediction_path = write_predictions(predictions, args.predictions)
    decision_path = write_decisions(decisions, args.decisions)
    csv_path = write_review_csv(decisions, args.csv)
    html_path = write_review_html(decisions, args.html)
    print(json.dumps({
        "images": len(paths),
        "predictions": str(prediction_path),
        "decisions": str(decision_path),
        "csv": str(csv_path),
        "html": str(html_path),
        "review_required": sum(decision.review_required for decision in decisions),
        "source_images_modified": False,
    }, ensure_ascii=False, indent=2))
    return 0


def _apply(args: argparse.Namespace) -> int:
    decisions = read_decisions(args.decisions)
    results = apply_decisions(decisions, args.output, include_review=args.include_review)
    summary: dict[str, int] = {}
    for result in results:
        summary[result.status] = summary.get(result.status, 0) + 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="photo-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-dataset", help="scan labeled folders and create a split manifest")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--seed", type=int, default=1337)
    prepare.set_defaults(handler=_prepare_dataset)

    classify = subparsers.add_parser("classify", help="run the optional OpenCLIP zero-shot baseline")
    classify.add_argument("--input", required=True)
    classify.add_argument("--predictions", default="artifacts/predictions.jsonl")
    classify.add_argument("--decisions", default="artifacts/decisions.jsonl")
    classify.add_argument("--csv", default="artifacts/review.csv")
    classify.add_argument("--html", default="artifacts/review.html")
    classify.add_argument("--model", default="ViT-B-32")
    classify.add_argument("--pretrained", default="openai")
    classify.add_argument("--device", default=None)
    classify.set_defaults(handler=_classify)

    classify_trained = subparsers.add_parser(
        "classify-trained",
        help="classify a folder with the locally trained OpenCLIP linear head; originals remain untouched",
    )
    classify_trained.add_argument("--input", required=True)
    classify_trained.add_argument("--checkpoint", default="artifacts/linear_probe/openclip_linear_probe.pt")
    classify_trained.add_argument("--predictions", default="artifacts/inference/predictions.jsonl")
    classify_trained.add_argument("--decisions", default="artifacts/inference/decisions.jsonl")
    classify_trained.add_argument("--csv", default="artifacts/inference/review.csv")
    classify_trained.add_argument("--html", default="artifacts/inference/review.html")
    classify_trained.add_argument("--device", default=None)
    classify_trained.set_defaults(handler=_classify_trained)

    evaluate_parser = subparsers.add_parser("evaluate", help="calculate metrics against a truth manifest")
    evaluate_parser.add_argument("--truth-manifest", required=True)
    evaluate_parser.add_argument("--decisions", required=True)
    evaluate_parser.add_argument("--output", required=True)
    evaluate_parser.set_defaults(handler=_evaluate)

    train = subparsers.add_parser(
        "train-linear",
        help="train a classifier head on frozen OpenCLIP features using only manifest splits",
    )
    train.add_argument("--manifest", required=True)
    train.add_argument("--output-dir", default="artifacts/linear_probe")
    train.add_argument("--model", default="ViT-B-32")
    train.add_argument("--pretrained", default="openai")
    train.add_argument("--device", default=None)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--epochs", type=int, default=60)
    train.add_argument("--patience", type=int, default=10)
    train.add_argument("--learning-rate", type=float, default=0.01)
    train.add_argument("--seed", type=int, default=1337)
    train.set_defaults(handler=_train_linear)

    prepare_hierarchy = subparsers.add_parser(
        "prepare-hierarchy",
        help="build a split-preserving manifest from nested hierarchy folders and derive node manifests",
    )
    hierarchy_source = prepare_hierarchy.add_mutually_exclusive_group(required=True)
    hierarchy_source.add_argument("--input", help="root of the nested hierarchy dataset")
    hierarchy_source.add_argument("--manifest", help="existing four-leaf manifest to project")
    prepare_hierarchy.add_argument("--previous-manifest", help="preserve prior split membership by content hash")
    prepare_hierarchy.add_argument("--new-sample-split", choices=("train", "validation", "test"), default="train")
    prepare_hierarchy.add_argument("--output-dir", default="artifacts/hierarchical")
    prepare_hierarchy.set_defaults(handler=_prepare_hierarchy)

    train_hierarchy = subparsers.add_parser(
        "train-hierarchy",
        help="train independent binary heads for the person, sky, and still/landscape nodes",
    )
    train_hierarchy.add_argument("--manifest", required=True)
    train_hierarchy.add_argument("--output-dir", default="artifacts/hierarchical")
    train_hierarchy.add_argument("--model", default="ViT-B-32")
    train_hierarchy.add_argument("--pretrained", default="openai")
    train_hierarchy.add_argument("--device", default=None)
    train_hierarchy.add_argument("--batch-size", type=int, default=16)
    train_hierarchy.add_argument("--epochs", type=int, default=60)
    train_hierarchy.add_argument("--patience", type=int, default=10)
    train_hierarchy.add_argument("--learning-rate", type=float, default=0.01)
    train_hierarchy.add_argument("--seed", type=int, default=1337)
    train_hierarchy.add_argument(
        "--third-head",
        choices=("linear", "rbf_kernel_ridge"),
        default="linear",
        help="classifier type for the final still-life/landscape node",
    )
    train_hierarchy.set_defaults(handler=_train_hierarchy)

    train_hierarchy_cv = subparsers.add_parser(
        "train-hierarchy-cv",
        help="run stratified K-fold CV for the cascade with a selectable third-stage head",
    )
    train_hierarchy_cv.add_argument("--manifest", required=True)
    train_hierarchy_cv.add_argument("--output-dir", default="artifacts/hierarchical_rbf_cv")
    train_hierarchy_cv.add_argument("--folds", type=int, default=5)
    train_hierarchy_cv.add_argument("--seed", type=int, default=1337)
    train_hierarchy_cv.add_argument("--model", default="ViT-B-32")
    train_hierarchy_cv.add_argument("--pretrained", default="openai")
    train_hierarchy_cv.add_argument("--device", default=None)
    train_hierarchy_cv.add_argument("--batch-size", type=int, default=16)
    train_hierarchy_cv.add_argument("--epochs", type=int, default=60)
    train_hierarchy_cv.add_argument("--patience", type=int, default=10)
    train_hierarchy_cv.add_argument("--learning-rate", type=float, default=0.01)
    train_hierarchy_cv.add_argument(
        "--third-head",
        choices=("linear", "rbf_kernel_ridge"),
        default="rbf_kernel_ridge",
        help="classifier for the final still-life/landscape node",
    )
    train_hierarchy_cv.set_defaults(handler=_train_hierarchy_cv)

    compare_heads = subparsers.add_parser(
        "compare-third-heads",
        help="compare five classifier heads on third-stage train/validation data only",
    )
    compare_heads.add_argument("--manifest", required=True)
    compare_heads.add_argument("--output-dir", default="artifacts/third_head_comparison")
    compare_heads.add_argument("--model", default="ViT-B-32")
    compare_heads.add_argument("--pretrained", default="openai")
    compare_heads.add_argument("--device", default=None)
    compare_heads.add_argument("--batch-size", type=int, default=16)
    compare_heads.add_argument("--epochs", type=int, default=80)
    compare_heads.add_argument("--patience", type=int, default=12)
    compare_heads.add_argument("--learning-rate", type=float, default=0.01)
    compare_heads.add_argument("--seed", type=int, default=1337)
    compare_heads.set_defaults(handler=_compare_third_heads)

    evaluate_heads = subparsers.add_parser(
        "evaluate-third-heads",
        help="evaluate saved third-stage heads on the held-out test subset once",
    )
    evaluate_heads.add_argument("--manifest", required=True)
    evaluate_heads.add_argument("--experiment-dir", required=True)
    evaluate_heads.add_argument("--output-dir", default=None)
    evaluate_heads.add_argument("--batch-size", type=int, default=16)
    evaluate_heads.add_argument("--device", default=None)
    evaluate_heads.set_defaults(handler=_evaluate_third_heads)

    build_rbf = subparsers.add_parser(
        "build-rbf-cascade",
        help="create a new cascade checkpoint with only the third-stage head replaced by RBF kernel ridge",
    )
    build_rbf.add_argument("--cascade-checkpoint", required=True)
    build_rbf.add_argument("--rbf-head-checkpoint", required=True)
    build_rbf.add_argument("--output-checkpoint", required=True)
    build_rbf.set_defaults(handler=_build_rbf_cascade)

    prepare_v2 = subparsers.add_parser(
        "prepare-hierarchy-v2",
        help="prepare the v2 person/non-person and three-way non-person manifests",
    )
    prepare_v2.add_argument("--input", required=True)
    prepare_v2.add_argument("--output-dir", default="artifacts/hierarchical_v2")
    prepare_v2.add_argument("--previous-manifest", default=None)
    prepare_v2.add_argument("--seed", type=int, default=1337)
    prepare_v2.set_defaults(handler=_prepare_v2)

    subject_reports = subparsers.add_parser(
        "generate-subject-reports",
        help="generate local Florence-2 descriptions, Grounding DINO boxes, and crops",
    )
    subject_reports.add_argument("--manifest", required=True)
    subject_reports.add_argument("--output-dir", default="artifacts/hierarchical_v2/subject_reports")
    subject_reports.add_argument("--split", choices=("train", "validation", "test"), default=None)
    subject_reports.add_argument("--max-images", type=int, default=None)
    subject_reports.add_argument("--device", default=None)
    subject_reports.add_argument("--no-florence", action="store_true")
    subject_reports.add_argument("--no-grounding", action="store_true")
    subject_reports.set_defaults(handler=_generate_subject_reports)

    category_subject_data = subparsers.add_parser(
        "generate-category-subject-data",
        help="generate/resume per-category subject annotations and crop files in curated class folders",
    )
    category_subject_data.add_argument(
        "--input-root",
        default=r"data\dataset\label layer data",
        help="root containing portraits, landscapes, still_life, and events",
    )
    category_subject_data.add_argument("--device", default=None)
    category_subject_data.add_argument("--limit-per-category", type=int, default=None)
    category_subject_data.add_argument("--checkpoint-every", type=int, default=10)
    category_subject_data.set_defaults(handler=_generate_category_subject_data)

    train_v2_parser = subparsers.add_parser(
        "train-hierarchy-v2",
        help="train the v2 independent person gate and non-person three-way head",
    )
    train_v2_parser.add_argument("--manifest", required=True)
    train_v2_parser.add_argument("--reports", default=None)
    train_v2_parser.add_argument("--output-dir", default="artifacts/hierarchical_v2")
    train_v2_parser.add_argument("--model", default="ViT-B-32")
    train_v2_parser.add_argument("--pretrained", default="openai")
    train_v2_parser.add_argument("--device", default=None)
    train_v2_parser.add_argument("--batch-size", type=int, default=16)
    train_v2_parser.add_argument("--epochs", type=int, default=60)
    train_v2_parser.add_argument("--patience", type=int, default=10)
    train_v2_parser.add_argument("--learning-rate", type=float, default=0.01)
    train_v2_parser.add_argument("--seed", type=int, default=1337)
    train_v2_parser.set_defaults(handler=_train_v2)

    classify_v2 = subparsers.add_parser(
        "classify-hierarchy-v2",
        help="classify a folder with the v2 cascade checkpoint",
    )
    classify_v2.add_argument("--input", required=True)
    classify_v2.add_argument("--checkpoint", default="artifacts/hierarchical_v2/hierarchical_v2_openclip_probe.pt")
    classify_v2.add_argument("--reports", default=None)
    classify_v2.add_argument("--predictions", default="artifacts/hierarchical_v2_inference/predictions.jsonl")
    classify_v2.add_argument("--decisions", default="artifacts/hierarchical_v2_inference/decisions.jsonl")
    classify_v2.add_argument("--csv", default="artifacts/hierarchical_v2_inference/review.csv")
    classify_v2.add_argument("--html", default="artifacts/hierarchical_v2_inference/review.html")
    classify_v2.add_argument("--device", default=None)
    classify_v2.set_defaults(handler=_classify_v2)

    classify_hierarchy = subparsers.add_parser(
        "classify-hierarchical",
        help="classify a folder using the trained three-stage cascade",
    )
    classify_hierarchy.add_argument("--input", required=True)
    classify_hierarchy.add_argument(
        "--checkpoint",
        default="artifacts/hierarchical_retrained_20260922/hierarchical_openclip_rbf_stage3.pt",
    )
    classify_hierarchy.add_argument("--predictions", default="artifacts/hierarchical_inference/predictions.jsonl")
    classify_hierarchy.add_argument("--decisions", default="artifacts/hierarchical_inference/decisions.jsonl")
    classify_hierarchy.add_argument("--csv", default="artifacts/hierarchical_inference/review.csv")
    classify_hierarchy.add_argument("--html", default="artifacts/hierarchical_inference/review.html")
    classify_hierarchy.add_argument("--device", default=None)
    classify_hierarchy.set_defaults(handler=_classify_hierarchical)

    apply_parser = subparsers.add_parser("apply", help="copy approved decisions without modifying originals")
    apply_parser.add_argument("--decisions", required=True)
    apply_parser.add_argument("--output", required=True)
    apply_parser.add_argument("--include-review", action="store_true")
    apply_parser.set_defaults(handler=_apply)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)
