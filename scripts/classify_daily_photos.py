"""Single daily entry point; uses local venv and ACTIVE_WORKFLOW.json."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from photo_classifier_agent.daily_workflow import ROOT, check_layout, load_assets, resolve_path, run_folder, scan_images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "daily_workflow_config.json")
    parser.add_argument("--input", type=Path, help="未标注的待分类照片文件夹")
    parser.add_argument("--output", type=Path, help="分类复制结果根目录，可为输入直接父目录（类别与输入并列）")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--check", action="store_true", help="核验环境和权重；不推理、不创建照片输出")
    parser.add_argument("--plan", action="store_true", help="扫描输入并预览配置，不加载模型推理")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    source = args.input or config.get("input_directory")
    destination = args.output or config.get("output_directory")
    threads = args.threads if args.threads is not None else config.get("cpu_threads", 8)
    recursive = config.get("recursive", True)
    if not isinstance(recursive, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("配置中的递归/线程数无效")
    if config.get("archive_policy", "cascade_prediction") != "cascade_prediction":
        raise ValueError("日常归档策略必须为 cascade_prediction：每张成功预测直接归档，不使用置信度门槛")
    workflow = ROOT / "ACTIVE_WORKFLOW.json"
    if args.check:
        _, _, audit = load_assets(workflow)
        print(json.dumps({"environment_and_models_ready": True, "input_configured": bool(source), "input_directory": str(source) if source else None,
                          "output_directory": str(resolve_path(destination)), "model_audit": audit,
                          "inference_started": False}, ensure_ascii=False, indent=2))
        return 0
    if not source:
        raise ValueError("尚未配置待分类文件夹。请提供 --input 路径，或在 daily_workflow_config.json 填写 input_directory；不会自动处理训练数据。")
    input_root, output_root = resolve_path(source), resolve_path(destination)
    if args.plan:
        check_layout(input_root, output_root)
        paths, skipped = scan_images(input_root, recursive)
        print(json.dumps({"input_directory": str(input_root), "output_directory": str(output_root), "images": len(paths),
                          "paths": [str(p) for p in paths], "skipped_links": skipped,
                          "archive_policy": "cascade_prediction", "confidence_threshold_enabled": False,
                          "copy_only": True, "inference_started": False}, ensure_ascii=False, indent=2))
        return 0
    summary = run_folder(input_root, output_root, workflow, recursive=recursive, threads=threads)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 2 if summary["statuses"].get("failed", 0) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"无法执行：{exc}", file=sys.stderr)
        raise SystemExit(1)
