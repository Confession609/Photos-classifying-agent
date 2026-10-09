# 已测试环境

本机：Windows、Python 3.12.14、CPU。下列版本来自本机已安装元数据，不代表在全新机器重新安装全部通过，也不要求复制整个 `.venv`：

```text
torch==2.14.0
torchvision==0.29.0
open-clip-torch==3.3.0
transformers==4.49.0
accelerate==1.15.0
peft==0.21.0
safetensors==0.8.0
huggingface-hub==0.29.3
Pillow==12.3.0
requests==2.34.2
einops==0.8.2
timm==1.0.29
numpy==2.5.3
tokenizers==0.21.1
scipy==1.18.1
ftfy==6.3.1
```

`pyproject.toml` 固定 Transformers/Hugging Face Hub 的关键版本；其他依赖保留兼容范围。如果包源不存在某个本机版本，不要照搬整个列表作为安装锁文件。先按 README 安装，再运行权重核验和小批量试用；兼容性问题应作为环境缺失明确报告，不静默换模型或训练。
