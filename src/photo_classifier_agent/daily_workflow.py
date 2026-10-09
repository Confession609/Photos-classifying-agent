"""Local deployment of the active GD top-1 / original+crop two-head workflow.

This module never trains models, reads labels, or moves/deletes source images.
Photo output is copy-only, exclusive-create, hash-verified and collision-safe.
"""
from __future__ import annotations

import hashlib
import html
import importlib.util
import json
import math
import os
import shutil
import time
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .dataset import SUPPORTED_EXTENSIONS
from .hierarchy_v2 import V2_LEAF_LABELS, V2_NON_PERSON_LABELS, V2_PERSON_LABELS

ROOT = Path(__file__).resolve().parents[2]
CLIP_SHA = "e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31"
TZ = timezone(timedelta(hours=8))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def resolve_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def linked(path):
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def check_layout(input_root, output_root):
    input_root, output_root = input_root.resolve(), output_root.resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"待分类文件夹不存在：{input_root}")
    if input_root == output_root or output_root.is_relative_to(input_root) or (input_root.is_relative_to(output_root) and input_root.parent != output_root):
        raise ValueError("输出不能等于或位于输入内；若输出包含输入，只允许输入的直接父目录作为并列分类目录根")
    if input_root.parent == output_root and input_root.name in (*V2_LEAF_LABELS, "_待复核", "_运行记录"):
        raise ValueError("输入目录名与输出类别/记录目录冲突")
    if output_root == ROOT or ROOT.is_relative_to(output_root) or output_root == Path.home().resolve():
        raise ValueError("不能将工作区根目录、其父目录、磁盘根或用户主目录作为输出")
    for name in ("data", "algorithms", "artifacts", "src", "scripts", "tests", ".venv", ".git", ".codex", ".agents"):
        protected = (ROOT / name).resolve()
        if output_root == protected or output_root.is_relative_to(protected):
            raise ValueError(f"输出禁止写入受保护的项目目录：{name}")
    for path in (output_root, *output_root.parents):
        if (path / "SEALED_WORKFLOW.json").exists():
            raise ValueError("不能写入封存工作流")


def safe_directory(path, output_root):
    """Reject redirected category/report directories before creating or using."""
    path, output_root = Path(path), Path(output_root)
    if not path.is_relative_to(output_root):
        raise ValueError("目标目录不在输出范围内")
    current = path
    while current != output_root.parent:
        if linked(current):
            raise ValueError(f"输出目录不允许符号链接或junction：{current}")
        current = current.parent
    path.mkdir(parents=True, exist_ok=True)
    if not path.resolve().is_relative_to(output_root.resolve()):
        raise ValueError("目标目录解析到输出范围之外")


def scan_images(root, recursive=True):
    """Unlabeled scan; no category, description, JSON or crop sidecars read."""
    root = root.resolve()
    found, skipped = [], []
    for directory, dirs, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in list(dirs):
            if linked(base / name):
                dirs.remove(name)
                skipped.append(str(base / name))
        for name in files:
            path = base / name
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            if linked(path) or not path.resolve().is_relative_to(root):
                skipped.append(str(path))
            else:
                found.append(path)
        if not recursive:
            dirs.clear()
    return sorted(found, key=lambda p: p.relative_to(root).as_posix().casefold()), skipped


def copy_photo(source, directory, expected_hash, output_root):
    """Never overwrite: repeat identical copies are skipped, collisions renamed."""
    source, directory = Path(source), Path(directory)
    safe_directory(directory, output_root)
    if digest(source) != expected_hash:
        raise ValueError("原图在推理后发生变化，拒绝复制")
    candidates = [directory / source.name, directory / f"{source.stem}__{expected_hash[:12]}{source.suffix}"]
    index = 0
    while True:
        destination = candidates[index] if index < 2 else directory / f"{source.stem}__{expected_hash[:12]}_{index - 1}{source.suffix}"
        index += 1
        if destination.exists() or linked(destination):
            if not linked(destination) and destination.is_file() and digest(destination) == expected_hash:
                return {"destination_path": str(destination), "copy_status": "skipped_identical"}
            continue
        try:
            target = destination.open("xb")
        except FileExistsError:
            continue
        try:
            with source.open("rb") as stream, target:
                shutil.copyfileobj(stream, target, 8 * 1024 * 1024)
            if digest(destination) != expected_hash or digest(source) != expected_hash:
                raise ValueError("复制内容校验失败或原图已变化")
            shutil.copystat(source, destination)
        except BaseException:
            target.close()
            # Only the exclusive-created file from this call is removed.
            destination.unlink(missing_ok=True)
            raise
        return {"destination_path": str(destination), "copy_status": "copied" if index == 1 else "copied_collision_renamed"}


def validate_score(score):
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("无效分类分数")
    return score


def load_assets(workflow_path):
    missing = [name for name in ("torch", "transformers", "open_clip", "PIL", "safetensors") if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("环境缺少配置/依赖，请先补齐，不自动安装：" + ", ".join(missing))
    import torch
    workflow = json.loads(Path(workflow_path).read_text(encoding="utf-8-sig"))
    if workflow.get("status") != "active" or workflow.get("subject_selector_enabled") is not False or workflow.get("category_hint_used") is not False:
        raise ValueError("本入口只支持已确认的无标签、无主体选择层正式工作流")
    if workflow.get("prompt") != "main subject." or workflow.get("feature_dim") != 1024 or tuple(workflow.get("leaf_labels", [])) != V2_LEAF_LABELS:
        raise ValueError("正式工作流的提示词/特征/类别与本入口不一致")
    gate_threshold = float(workflow.get("person_gate_threshold", -1))
    if not 0 <= gate_threshold <= 1:
        raise ValueError("无效人像门控阈值")
    heads_path, gd_path = resolve_path(workflow["heads"]), resolve_path(workflow["gd_model"])
    for path, expected in ((heads_path, workflow["heads_sha256"]), (gd_path / "model.safetensors", workflow["gd_model_sha256"])):
        if not path.is_file():
            raise FileNotFoundError(f"环境缺少模型文件：{path}")
        if digest(path) != expected:
            raise ValueError(f"正式模型校验不符：{path}")
    checkpoint = torch.load(heads_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "v2_ablation_v1" or checkpoint.get("feature_dim") != 1024 or checkpoint.get("variant") != "original_auto_crop":
        raise ValueError("不是当前原图+自动裁剪分类头格式")
    for node, names in (("person_gate", V2_PERSON_LABELS), ("non_person_classifier", V2_NON_PERSON_LABELS)):
        if tuple(checkpoint["heads"][node]["class_names"]) != names:
            raise ValueError("分类头类别顺序错误")
    if checkpoint.get("model_name") != "ViT-B-32-quickgelu":
        raise ValueError("OpenCLIP骨干与已验证的工作流不一致")
    # A local override makes a copied repository portable without rewriting
    # trained weights. The exact verified CLIP content hash is still required.
    clip_path = resolve_path(workflow.get("clip_model") or checkpoint["pretrained_path"])
    if not clip_path.is_file():
        raise FileNotFoundError(f"环境缺少OpenCLIP权重：{clip_path}")
    if digest(clip_path) != CLIP_SHA:
        raise ValueError("OpenCLIP权重哈希不符")
    audit = {"workflow_path": str(Path(workflow_path).resolve()), "workflow_sha256": digest(workflow_path),
             "version": workflow["version"], "gd_model": str(gd_path), "gd_sha256": workflow["gd_model_sha256"],
             "heads": str(heads_path), "heads_sha256": workflow["heads_sha256"],
             "clip_path": str(clip_path), "clip_sha256": CLIP_SHA,
             "prompt": workflow["prompt"], "person_gate_threshold": gate_threshold,
             "labels": list(V2_LEAF_LABELS), "device": "cpu", "training": False, "label_hints": False,
             "subject_selector_enabled": False}
    return workflow, checkpoint, audit


class DailyModels:
    def __init__(self, workflow, checkpoint, audit, threads):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from PIL import Image, ImageDraw
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoContrastiveEmbedding
        from .hierarchy_v2_training import _open_clip
        torch.set_num_threads(threads)
        self.torch, self.Image, self.ImageDraw = torch, Image, ImageDraw
        self.processor = AutoProcessor.from_pretrained(audit["gd_model"], local_files_only=True)
        self.gd = AutoModelForZeroShotObjectDetection.from_pretrained(audit["gd_model"], local_files_only=True)
        self.gd.config.num_labels = int(self.gd.config.max_text_len)
        def finite_padding(module, inputs, result):
            if torch.isnan(result).any() or torch.isposinf(result).any():
                raise FloatingPointError("GD输出含无效数值")
            return result.masked_fill(torch.isneginf(result), -100.)
        for module in self.gd.modules():
            if isinstance(module, GroundingDinoContrastiveEmbedding):
                module.register_forward_hook(finite_padding)
        self.gd.to("cpu").eval()
        for param in self.gd.parameters():
            param.requires_grad_(False)
        _, _, self.clip, self.preprocess, _ = _open_clip(checkpoint["model_name"], audit["clip_path"], "cpu")
        self.gate, self.other = torch.nn.Linear(1024, 2), torch.nn.Linear(1024, 3)
        self.gate.load_state_dict(checkpoint["heads"]["person_gate"]["state_dict"])
        self.other.load_state_dict(checkpoint["heads"]["non_person_classifier"]["state_dict"])
        self.gate.eval(); self.other.eval()
        for head in (self.gate, self.other):
            for param in head.parameters():
                param.requires_grad_(False)
        self.threshold = workflow["person_gate_threshold"]

    def predict(self, source, run_dir, identity):
        torch, Image = self.torch, self.Image
        with Image.open(source) as opened:
            # Keep decoding/orientation identical to the verified training/evaluation path.
            image = opened.convert("RGB")
        try:
            inputs = self.processor(images=[image], text=["main subject."], return_tensors="pt", padding=True)
            phrase = self.processor.tokenizer("main subject", add_special_tokens=False)["input_ids"]
            tokens = inputs["input_ids"][0].tolist()
            token = next((i for i in range(len(tokens) - len(phrase) + 1) if tokens[i:i + len(phrase)] == phrase), None)
            if token is None or token >= self.gd.config.max_text_len:
                raise ValueError("固定提示词超出GD文本范围")
            with torch.inference_mode():
                output = self.gd(**inputs)
                scores = output.logits[0, :, token].sigmoid()
                if not torch.isfinite(scores).all() or not torch.isfinite(output.pred_boxes).all():
                    raise ValueError("GD输出无效")
                selected = int(scores.argmax())
                cx, cy, bw, bh = output.pred_boxes[0, selected].tolist()
            if bw <= 0 or bh <= 0:
                raise ValueError("GD没有合法主体框，不伪造整图裁剪")
            w, h = image.size
            left = max(0, min(w - 1, math.floor((cx - bw / 2) * w)))
            top = max(0, min(h - 1, math.floor((cy - bh / 2) * h)))
            right = max(left + 1, min(w, math.ceil((cx + bw / 2) * w)))
            bottom = max(top + 1, min(h, math.ceil((cy + bh / 2) * h)))
            crop_path = run_dir / "subject_crops" / f"{identity}.jpg"
            cropped_pixels = image.crop((left, top, right, bottom))
            try:
                cropped_pixels.save(crop_path, quality=95)
            finally:
                cropped_pixels.close()
            with Image.open(crop_path) as cropped:
                tensors = torch.stack((self.preprocess(image), self.preprocess(cropped.convert("RGB"))))
            with torch.inference_mode():
                parts = self.clip.encode_image(tensors).float().cpu()
                parts = parts / parts.norm(dim=1, keepdim=True).clamp_min(1e-12)
                features = torch.cat((parts[0], parts[1])).unsqueeze(0) / math.sqrt(2)
                if not torch.isfinite(features).all():
                    raise ValueError("OpenCLIP特征无效")
                gate = self.gate(features).softmax(dim=1)[0]
                other = self.other(features).softmax(dim=1)[0]
                if not torch.isfinite(gate).all() or not torch.isfinite(other).all():
                    raise ValueError("分类头输出无效")
                index = 0 if float(gate[0]) >= self.threshold else int(other.argmax()) + 1
                confidence = float(gate[0]) if index == 0 else float(gate[1] * other[index - 1])
            preview = image.copy()
            preview.thumbnail((560, 420))
            draw = self.ImageDraw.Draw(preview)
            draw.rectangle((left / w * preview.width, top / h * preview.height,
                            right / w * preview.width, bottom / h * preview.height), outline="red", width=3)
            preview.save(run_dir / "previews" / f"{identity}.jpg", quality=85)
            preview.close()
            return {"predicted_category": V2_LEAF_LABELS[index], "confidence_uncalibrated": confidence,
                    "person_probability": float(gate[0]), "non_person_probabilities": dict(zip(V2_NON_PERSON_LABELS, other.tolist())),
                    "gd_score": float(scores[selected]), "prediction_xywh_norm": [cx - bw / 2, cy - bh / 2, bw, bh],
                    "crop_box_xyxy_pixels": [left, top, right, bottom],
                    "crop_box_xywh_norm_clamped": [left / w, top / h, (right - left) / w, (bottom - top) / h],
                    "crop_path": str(crop_path), "crop_sha256": digest(crop_path),
                    "preview_path": str(run_dir / "previews" / f"{identity}.jpg")}
        finally:
            image.close()


def write_reports(run_dir, records, summary):
    # No spreadsheet dependency. JSONL is authoritative; TXT/HTML are views.
    with (run_dir / "classification_results.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    lines = [f'“{r["relative_path"]}”——“{r["predicted_category"]}”' for r in records if r.get("predicted_category")]
    (run_dir / "分类结果.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    rows = []
    for r in records:
        esc = lambda value: html.escape(str(value), quote=True)
        original = f'<a href="{esc(Path(r["source_path"]).as_uri())}">打开原图</a>'
        if r.get("preview_path"):
            original += f'<br><img loading="lazy" src="previews/{r["image_id"]}.jpg" alt="原图红框预览">'
        cropped = f'<a href="subject_crops/{r["image_id"]}.jpg"><img loading="lazy" src="subject_crops/{r["image_id"]}.jpg" alt="GD主体裁剪"></a>' if r.get("crop_path") else "—"
        score = f'{r["confidence_uncalibrated"]:.3f}' if r.get("confidence_uncalibrated") is not None else "—"
        status = {"classified": "已分类", "failed": "失败"}.get(r["status"], r["status"])
        rows.append(f'<tr><td>{esc(r["relative_path"])}</td><td>{original}</td><td>{cropped}</td><td>{esc(r.get("predicted_category", "—"))}</td><td>{score}</td><td>{status}<br>{esc(r.get("error", ""))}</td><td>{esc(r.get("destination_path", ""))}</td></tr>')
    content = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>照片分类结果</title><style>body{font-family:system-ui;margin:24px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px;text-align:left}img{max-width:290px;max-height:220px}td{word-break:break-all}</style><h1>照片分类结果</h1>'
    content += f'<p>共{summary["total"]}张；已分类{summary["statuses"].get("classified", 0)}；失败{summary["statuses"].get("failed", 0)}。每张成功照片按级联预测类别归档，无置信度门槛。原图保留，输出仅复制。</p>'
    content += '<p>分数未校准，仅供参考，不代表正确概率、不阻止归档。人像门控0.5保留，非人像取三个类别中分数最高的一类。红框为本轮实际GD裁剪位置；没有使用类别提示。TXT仅列预测类别，归档是否成功请查看本表或JSONL。</p><table><thead><tr><th>照片</th><th>原图/红框</th><th>主体裁剪</th><th>预测类别</th><th>未校准分数</th><th>状态</th><th>输出位置</th></tr></thead><tbody>'
    content += "".join(rows) + "</tbody></table></html>"
    (run_dir / "report.html").write_text(content, encoding="utf-8")


def run_folder(input_root, output_root, workflow_path, recursive=True, threads=8,
               model_factory=DailyModels):
    if threads < 1:
        raise ValueError("CPU线程数必须为正")
    check_layout(input_root, output_root)
    paths, skipped_links = scan_images(input_root, recursive)
    if not paths:
        raise ValueError("输入文件夹中没有支持的照片")
    workflow, checkpoint, audit = load_assets(workflow_path)
    run_id = datetime.now(TZ).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run_dir = output_root / "_运行记录" / run_id
    for directory in (output_root, *(output_root / name for name in V2_LEAF_LABELS),
                      run_dir, *(run_dir / name for name in ("records", "subject_crops", "previews"))):
        safe_directory(directory, output_root)
    write_json(run_dir / "run_config.json", audit | {"input_directory": str(input_root), "output_directory": str(output_root),
               "copy_only": True, "recursive": recursive, "archive_policy": "cascade_prediction",
               "confidence_threshold_enabled": False,
               "input_paths": [str(p) for p in paths], "skipped_links": skipped_links})
    started, records = time.monotonic(), []
    def progress(phase):
        return {"phase": phase, "processed": len(records), "total": len(paths),
                "statuses": dict(Counter(r["status"] for r in records)), "run_directory": str(run_dir),
                "elapsed_seconds": time.monotonic() - started, "source_images_modified": False}
    write_json(run_dir / "progress.json", progress("loading_models"))
    print(json.dumps(progress("loading_models"), ensure_ascii=False), flush=True)
    try:
        models = model_factory(workflow, checkpoint, audit, threads)
        for index, source in enumerate(paths):
            base = {"source_path": str(source), "relative_path": source.relative_to(input_root).as_posix()}
            identity = hashlib.sha256(str(source).encode()).hexdigest()
            record = base | {"image_id": identity}
            try:
                source_hash = digest(source)
                record["source_sha256"] = source_hash
                record.update(models.predict(source, run_dir, identity))
                if record["predicted_category"] not in V2_LEAF_LABELS:
                    raise ValueError("预测类别不在固定四类中")
                validate_score(record["confidence_uncalibrated"])
                record["status"] = "classified"
                directory = output_root / record["predicted_category"]
                record.update(copy_photo(source, directory, source_hash, output_root))
            except Exception as exc:
                record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            write_json(run_dir / "records" / f"{identity}.json", record)
            records.append(record)
            write_json(run_dir / "progress.json", progress("running"))
            print(json.dumps(progress("running"), ensure_ascii=False), flush=True)
        summary = progress("completed") | {"copy_statuses": dict(Counter(r.get("copy_status", "not_copied") for r in records)),
                    "predicted_categories": dict(Counter(r["predicted_category"] for r in records if r.get("predicted_category"))),
                    "report": str(run_dir / "report.html"), "copy_only": True, "model_audit": audit,
                    "archive_policy": "cascade_prediction", "confidence_threshold_enabled": False}
        write_reports(run_dir, records, summary)
        write_json(run_dir / "summary.json", summary)
        write_json(run_dir / "progress.json", summary)
        return summary
    except BaseException as exc:
        write_json(run_dir / "progress.json", progress("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed") | {"error": str(exc)})
        raise
