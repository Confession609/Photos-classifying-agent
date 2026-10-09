# 模型来源与发布说明

- GD 骨干及处理器来源：[IDEA-Research/grounding-dino-tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny)。官方模型卡标注 Apache-2.0。本项目发布候选使用经过主体框微调的权重，固定 `main subject.` 提示，不是未修改的官方权重。
- CLIP 权重来源：[timm/vit_base_patch32_clip_224.openai](https://huggingface.co/timm/vit_base_patch32_clip_224.openai)，官方模型卡标注 Apache-2.0；运行时由 OpenCLIP 加载为 `ViT-B-32-quickgelu`。本项目不修改视觉编码器权重。
- 两个分类头由本项目训练。发布副本只保留推理必需元数据、类别顺序和 tensor，去除原始训练清单路径及缓存信息；与正式模型逐 tensor 核验一致，原始权重文件不改写。GD 发布配置移除原机器 `_name_or_path`，不改变权重。
- Florence 与 LightGBM 主体选择层不属于本次生产模型包。

模型包附 `third_party/APACHE-2.0.txt` 供查看上游许可文本。以上是来源与修改声明，不替代对训练照片来源、个人数据权益及微调权重发布权限的核查。项目自身尚未选择开源许可证，公开发布前需确定；本轮只准备本地发布候选，不自动上传，也不宣称所有再分发事项已完成法律审核。
