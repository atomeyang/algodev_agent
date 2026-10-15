# qwen3.8-27b

- **来源 / 版本**：`Qwen3.8-27B`（`Qwen3_5ForConditionalGeneration`，多模态 vision-language）。
  本地权重目录：`/mnt/workspace/yangt/vlm/phonevlm/models/Qwen3.8-27B/`
- **参数规模与量化**：27B，部署用 GGUF `UD-Q4_K_XL`（约 17.9 GB）+ 视觉投影 `mmproj-F16.gguf`（约 0.93 GB）。
  另存原始 safetensors 18 分片（约 54 GB）作为参考权重，不参与常规运行。
- **端点与启动方式**：见 `serving/`。当前以 llama.cpp `llama-server` 提供 **OpenAI 兼容**接口，
  `http://127.0.0.1:8084/v1`，模型名 `qwen3.8-27b`。
- **权重引用与校验和**：见 `weights/REFERENCE.md`（含 sha256 与官方 `crc32.txt`）。
- **许可与使用限制**：Apache License 2.0（见权重目录 `LICENSE`）。仅本机/内网辅助使用，
  标注结果**必须经人工复核**后才能进入冻结评测集。
- **用途**：☑ 标注辅助　☑ 资料提取　☑ 诊断　☐ 在线推理（不参与主流程推理）

## 在 algo_agent 中的定位

它**不是**被测算法模型，也不参与验收指标计算。它只做两件事：

1. **数据构建阶段**：给候选图片做缺陷预标注（blur / over_exp / under_exp / flare / white_balance
   + severity 0–3），供人工复核后进入数据集版本；这是当前 `white_balance`、`flare` 正样本不足的直接解法。
2. **诊断阶段**：对失败样本给出自然语言归因，辅助 Agent 生成下一轮优化假设。

关键约束：**本地模型给出的标签是"待复核候选"，不是真值**。评测集标签仍以人工确认为准，
所有经本地模型参与的样本在数据集文档里标注 `prelabel=qwen3.8-27b, reviewed_by=<人>`。
