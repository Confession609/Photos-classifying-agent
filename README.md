# 照片分类 Agent

本地运行的摄影照片分类工作流。当前版本不使用类别提示或主体选择层：

`原图 → GroundingDINO 主体裁剪 → OpenCLIP 原图＋裁剪特征 → 人像/非人像分类头 → 非人像三分类头`

输出四类：**人像、风光摄影、静物摄影、活动事件摄影**。星空包含在风光摄影中；人物是否是主要摄影主体决定人像归属，不是“有人就算人像”。所有成功预测直接复制到类别目录，分数供查看，不设置归档置信度门槛。不会移动、删除或覆盖原图。

## 下载后运行（Windows / Python 3.12）

需要两部分：本仓库源码，以及同一版本 Release 中的 `photo-agent-models-20261009.zip`。只下载 GitHub 自动生成的源码 ZIP 不包含训练权重，不能直接分类。

1. 将源码解压到一个新的文件夹；将模型 ZIP 解压到这个源码根目录，得到 `models/`、可移植的 `ACTIVE_WORKFLOW.json`、`daily_workflow_config.json` 和 `MODEL_MANIFEST.json`。**不要把发布包解压覆盖已有工作区的私人配置。**
2. 安装 Python 3.12（64位），在源码根目录打开终端，安装环境：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[vision]"
.\.venv\Scripts\python.exe scripts\verify_model_bundle.py
.\.venv\Scripts\python.exe scripts\classify_daily_photos.py --check
```

安装依赖需要联网；模型就绪后的照片推理使用本地权重，不上传图片、不自动下载模型。默认 CPU 运行，首次模型加载和照片处理可能较慢。本轮没有在全新联网环境重新安装依赖；发行包会进行隔离路径的模型核验与推理验证，但不是所有 Windows 硬件兼容性保证。当前本机依赖版本见 `TESTED_ENVIRONMENT.md`。

3. 修改 `daily_workflow_config.json` 的 `input_directory` 为自己的待分类照片目录，`output_directory` 为输出根目录。也可以直接指定：

```powershell
.\.venv\Scripts\python.exe scripts\classify_daily_photos.py --input ".\inbox" --output ".\outputs\classified" --plan
.\.venv\Scripts\python.exe scripts\classify_daily_photos.py --input ".\inbox" --output ".\outputs\classified"
```

`--plan` 只扫描，不加载模型或创建照片输出；`--check` 检查依赖、格式和权重哈希，不推理。配置完成后可双击根目录 `开始照片分类.cmd`；执行结束会显示结果并等待按键，不会立即关闭窗口。也支持 `运行照片分类.ps1`。

如果自行配置模型而不是使用项目 Release，复制两个 `*.example.json` 为实际配置；GD 和分类头需为兼容的微调版本，填写实际 SHA-256。`clip_model` 可覆盖旧 checkpoint 内的本机路径，但内容必须匹配本项目验证的 OpenCLIP 权重；不能直接换成任意 CLIP/GD 后宣称相同效果。

## 输出和安全边界

输出根目录下生成四个类别目录与 `_运行记录/批次编号/`。每批包含 `分类结果.txt`、`classification_results.jsonl`、`report.html`、进度/摘要、主体裁剪和红框预览；TXT 采用 `“照片名字”——“类别”` 格式，失败详情看 HTML/JSONL。

- 同名相同内容跳过复制；同名不同内容增加哈希后缀，不覆盖。
- 可将输出设为输入的直接父目录，使四个类别与待分类目录并列；不能输出到输入内部、项目数据/模型目录或磁盘根目录。
- 单图解码/推理/复制失败只记录错误，不强行归档。HEIC/HEIF 是否可读取取决于解码器，不自动转换。
- 中断保留已生成文件；再次运行会新建批次并重新推理，相同目标副本仍可跳过。不提供日常推理断点恢复命令。
- 分类分数没有概率校准；四类边界可能模糊，建议人工抽查。

## 开发与训练

`src/photo_classifier_agent/` 保存核心代码；`scripts/classify_daily_photos.py` 是日常入口。旧 `photo-agent` CLI 和部分模块保留研究用途，并非当前生产模型的默认入口。

可复用脚本包括 GD 框微调、Florence 描述微调（可选研究）、固定 `main subject.` 提示生成自动裁剪、两个分类头训练，以及固定提示 GD 评估。用各脚本 `--help` 查看参数，不会因下载源码而启动训练；分类头训练入口本身会执行训练，请提供独立输出目录。

训练必须使用固定的 train/validation 清单，按拍摄事件/连拍/相似照片分组，禁止跨集合混用；test、examination 或最终泛化样本不得偷偷用于训练。GD 无类别提示训练时清单设置 `prompt_mode="fixed_main_subject"`，推理不读取标签、描述或人工框。个人照片、人工标注和既有清单不公开，因此本仓库不是完整训练数据的复现包。

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[vision,dev]"
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

测试不需要发布模型或真实照片；Windows 启动器测试需要按前文创建 `.venv`。

发布文件范围、许可证待确认事项和隐私说明见 [PUBLICATION.md](PUBLICATION.md)。模型来源与修改说明见 [MODEL_NOTICES.md](MODEL_NOTICES.md)。
