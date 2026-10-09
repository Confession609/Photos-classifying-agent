# 本次上传清单

## 1. GitHub 仓库：上传源码

上传 `github_upload/` **里面的全部内容**作为仓库根目录，不要把外层 `github_upload` 目录再套一层。它包含：

- `src/`：完整核心 Python 包。
- `scripts/`：日常入口、模型包核验、源码导出、可复用训练/裁剪/评估脚本。
- `tests/`：相关自动化测试。
- `algorithms/` 中的两个依赖清单，不含下载模型或包。
- `README.md`、`PUBLICATION.md`、`MODEL_NOTICES.md`、`TESTED_ENVIRONMENT.md`、本上传说明。
- `pyproject.toml`、`.gitignore`、`.gitattributes`、两个 `*.example.json`、CMD/PowerShell 启动入口。

GitHub 网页上传可能不显示点号开头的文件，请确认 `.gitignore` 与 `.gitattributes` 也提交。

## 2. 同版本 GitHub Release：上传模型附件

另上传这两个本地文件：

```text
release_assets/photo-agent-models-20261009.zip
release_assets/photo-agent-models-20261009.zip.sha256
```

**不要把模型 ZIP 或展开的权重放入普通源码提交。** 用户需要下载仓库源码和这个 Release 附件两部分；GitHub 自动生成的 Source code ZIP 不包含本项目的微调模型。Release 发布后，可把它的下载链接加到 README 首次运行段落；本轮尚未上传，因此没有编造下载链接。

## 3. 下载者的首次运行

将源码解压到新目录，模型包解压到源码根目录；安装 Python 3.12，按 README 创建 `.venv`、安装依赖、运行模型核验；填写自己的输入/输出路径，然后双击 `开始照片分类.cmd` 或调用日常入口。首次依赖安装需要联网；照片推理在本地完成。

不需要你的 `data/`、训练照片、个人复核、历史实验、`.venv/`、项目记忆或本机日常路径。模型附件提供的是同一生产网络的可移植副本；原权重和私人配置保持不变。

## 4. 公开前仍需确认

项目尚未选择自身开源许可证；确定后添加 LICENSE，并确认训练照片来源及微调模型权重的公开发布权限。上游许可与修改声明见 MODEL_NOTICES。打包和本机隔离路径验证不等于完成法律审核或在另一台全新电脑安装验证。
