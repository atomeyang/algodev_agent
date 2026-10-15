# local_largemodel — 本地辅助大模型

存放本机运行、用于**资料提取 / 标注辅助 / 诊断**的本地大模型（例如 Qwen3-27B）。
它不参与主流程的在线推理；研究设计与评判仍由云端 DeepSeek 与冻结的 Evaluator 承担，
本地模型只作为可替换的辅助能力接入（见指导说明 §3 provider 与 §4 目录表）。

> 目录名取自指导说明 4 章目录表：`local_largemodel/<model_id>/`。
> `local_auxmodels/` 作为别名同样受支持：`Settings.local_models_dir`
> 优先解析本目录，不存在时回退到 `local_auxmodels/`。

## 目录约定

每个本地模型独立一个目录，命名用模型 id（例如 `qwen3-27b/`）：

```
local_largemodel/<model_id>/
  weights/     # 权重本体，或指向权重的引用文件（引用必须可解析并带校验和）
  serving/     # 启动/服务脚本（vLLM、llama.cpp、OpenAI 兼容端点等）
  configs/     # 端点、上下文长度、量化、采样参数等
  adapters/    # LoRA / 适配器权重或引用
  MODEL.md     # 说明：来源、版本、能力、许可与校验和
```

`weights/` 与 `adapters/` 里的二进制大文件**不进普通 Git**（见 `.gitignore`），
只提交引用文件、配置和 `MODEL.md`。

## 快速开始

```bash
cp -r local_largemodel/_template local_largemodel/qwen3-27b
# 放入权重或写引用文件，然后按 serving/README.md 启动服务
```

在 `.env` 中配置端点（`init` 生成的 `.env.example` 已列出）：

```
ALGO_LOCAL_BASE_URL=http://127.0.0.1:8000/v1
ALGO_LOCAL_MODEL=qwen3-27b
```

之后 `algo-agent doctor` 会检查该端点是否可达，并报告当前是 OFFLINE 还是 LIVE 模式。

## 数据构建方案中的角色（IQA 实例）

本地辅助大模型是**数据构建方案的一部分**，不是可选装饰。在 IQA 任务里它承担
「预标注候选生成 + 提示词迭代 + 类别诊断」，流程固定在 `algorithm_plugins/iqa/annotation.py`
与 `scripts/iqa_annotate.py`：

1. `annotation.annotate(path)` 让本地 VLM 输出结构化 JSON 缺陷向量
   （5 类缺陷 present/severity + wb direction + 0-100 质量分 + confidence）；
2. 结果写入 `<task>_experiment/data/annotations/runs/<version>/<run_id>/prelabels.jsonl`，
   方案写入同任务的 `data/annotations/plans/<version>/`。脚本用 `--task`、`--version` 指定归属，
   不再创建顶层 `iqa_annotations/`。每行带
   `model_id / prompt_version / provenance`，且 `review_required` 恒为 `true`；
3. `scripts/iqa_annotate.py --split eval|dev` 同时产出 `.agreement.json`，
   与冻结 eval 的 GT 对比得到各类 P/R/F1、severity MAE、wb 方向准确率；
4. **候选标签必须经人工复核**才能进入某个 `dataset_version`；罕见类
   （flare / white_balance）100% 复核，其他类按 confidence 分层抽检 ≥30%。

诊断规则（来自 20 张 eval 图实跑）：

- 强信号类可直接加速人工标注（over_exp F1 0.857）；
- blur 弱（F1 0.222），需 prompt v2 增加边缘能量/中心-角落对比判据；
- flare / white_balance 误报明显（20 张里分别 4 / 6 个 FP），v1 输出必须加
  「描述光源/色偏方向」的负向约束，并只用作**缺失正样本的候选生成器**。

当前实例：`local_largemodel/qwen3.8-27b/`（llama.cpp，OpenAI 兼容端点
`http://127.0.0.1:8084/v1`，多模态，Apache-2.0）。启动见
`qwen3.8-27b/serving/start_llama_server.sh`，端点配置见
`qwen3.8-27b/configs/endpoint.yaml`。

> 重要边界：本地模型输出**永远不是 ground truth**；它不改变冻结 Evaluator 的判定，
> 也不能在没有人工复核的情况下写入数据集版本。
