# 模型评估模块说明

本目录提供 TrafficLLM 的模型评估能力，支持：

- 检测任务（`detection`）：计算分类指标（Accuracy / Precision / Recall / F1）、混淆矩阵与分类报告。
- 生成任务（`generation`）：按样本类别聚合生成结果并写出 JSON。
- 单模型评估与批量 checkpoint 评估：可自动遍历 `checkpoint-*` 目录，按约束筛选并自动选择最优 checkpoint。

评估入口脚本为项目根目录下的 `evaluation.py`。

---

## 一、文件结构

### 1. 评估入口脚本
- `../evaluation.py`
    - 统一评估主入口（基于 `fire` 命令行）。
    - 支持 GLM4 `messages` 格式测试集（推荐）与旧版 `instruction/output` 兼容格式。
    - 支持 LoRA adapter、P-Tuning 以及基础模型三种加载路径。

### 2. 评估结果文件（示例）
- `*_batch_eval_results.json`
    - 检测任务的批量评估输出，包含：
        - 评估约束（`min_accuracy/min_precision/min_recall`）
        - 最优 checkpoint 指标
        - 全部 checkpoint 评估结果
- `*_eval_metrics.txt`
    - 历史评估摘要文本（便于快速查看）。

---

## 二、评估工作流

### 检测任务（detection）

输入：
- 测试集（推荐 `GLM4_*_test_with_label.jsonl`，每行含 `messages`）
- 标签映射文件（`*_label.json`）

主要流程：
1. 读取测试样本，提取 prompt 消息与真实标签。
2. 对单模型或多个 checkpoint 逐一推理。
3. 解析模型输出标签并与真实标签对齐。
4. 计算指标：
     - `accuracy`
     - `precision`（weighted）
     - `recall`（weighted）
     - `f1`（weighted）
     - `confusion_matrix`
     - `classification_report`
5. 若设置约束（最小精度/召回/准确率），优先选择满足约束且 F1 最高的 checkpoint；若无满足项，回退到全局 F1 最高项。
6. 写出汇总结果到 `output_result_file`（默认 `batch_eval_results.json`）。

### 生成任务（generation）

输入：
- 测试集（`messages` 或兼容格式）

主要流程：
1. 执行采样生成（`top_p=0.8, temperature=0.8`）。
2. 按 user 文本中的类别字段聚合模型生成内容。
3. 写出结果：
     - 单模型：`generation.json`
     - 批量 checkpoint：`generation_checkpoint-xxx.json`

---

## 三、命令行参数说明

`evaluation.py` 主要参数：

- `model_name`: 基座模型路径或模型名（必填）。
- `test_file`: 测试集文件路径（必填）。
- `traffic_task`: 任务类型，`detection` 或 `generation`（必填）。
- `label_file`: 标签映射文件，仅 `detection` 必填。
- `ptuning_path`: P-Tuning 权重目录（可选）。
- `checkpoint_root`: LoRA checkpoint 根目录（可选，启用批量评估）。
- `checkpoint_pattern`: checkpoint 匹配关键字，默认 `checkpoint-`。
- `max_samples`: 最大评估样本数，默认 `1000`。
- `max_new_tokens`: 最大生成长度，默认 `128`。
- `min_accuracy`: 最小准确率约束，默认 `0.0`。
- `min_precision`: 最小精度约束，默认 `0.0`。
- `min_recall`: 最小召回率约束，默认 `0.0`。
- `output_result_file`: 检测任务汇总输出路径，默认 `batch_eval_results.json`。
- `print_response`: 是否打印每条模型输出，默认 `False`。

说明：
- 当 `checkpoint_root` 未提供时，执行单模型评估。
- 当 `checkpoint_root` 提供时，自动扫描并排序 `checkpoint-*` 子目录进行批量评估。

---

## 四、使用说明

### 1. 检测任务：单模型评估

```bash
python evaluation.py \
    --model_name ./models/glm-4-9b-chat-lora \
    --test_file ./datasets/csic-2010/GLM4_csic-2010_detection_packet_test_with_label.jsonl \
    --label_file ./datasets/csic-2010/csic-2010_label.json \
    --traffic_task detection \
    --max_samples 1000 \
    --output_result_file ./evaluation/csic-2010_batch_eval_results.json
```

### 2. 检测任务：批量 checkpoint 评估

```bash
python evaluation.py \
    --model_name ./models/glm-4-9b-chat \
    --test_file ./datasets/dapt-2020/GLM4_dapt-2020_detection_packet_test_with_label.jsonl \
    --label_file ./datasets/dapt-2020/dapt-2020_label.json \
    --traffic_task detection \
    --checkpoint_root ./FT/Logs/dapt-2020 \
    --checkpoint_pattern checkpoint- \
    --min_accuracy 0.90 \
    --min_precision 0.90 \
    --min_recall 0.90 \
    --output_result_file ./evaluation/dapt-2020_batch_eval_results.json
```

### 3. 生成任务评估

```bash
python evaluation.py \
    --model_name ./models/glm-4-9b-chat-lora \
    --test_file ./datasets/iscx-vpn-2016/GLM4_iscx-vpn-2016_generation_packet_test_with_label.jsonl \
    --traffic_task generation \
    --max_samples 200
```

### 4. 使用 P-Tuning 权重

```bash
python evaluation.py \
    --model_name ./models/glm-4-9b-chat \
    --ptuning_path ./models/ptuning_ckpt \
    --test_file ./datasets/csic-2010/GLM4_csic-2010_detection_packet_test_with_label.jsonl \
    --label_file ./datasets/csic-2010/csic-2010_label.json \
    --traffic_task detection
```

---

## 五、输入与输出约定

### 1. 测试集格式

推荐使用 GLM4 对话格式（JSONL，每行一条）：

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"标签"}]}
```

兼容旧格式（JSONL）：

```json
{"instruction":"...","output":"标签"}
```

说明：
- 检测任务评估必须包含真实标签，因此应使用带标签测试集（通常为 `*_test_with_label.jsonl`）。
- 若使用去标签测试集（assistant 为 `-`），将无法得到有效检测指标。

### 2. 标签文件格式（检测任务）

`label_file` 为 JSON 字典，示例：

```json
{
  "benign": 0,
  "attack": 1
}
```

### 3. 检测任务输出文件

`output_result_file`（JSON）结构示例：

```json
{
  "constraints": {
    "min_accuracy": 0.9,
    "min_precision": 0.9,
    "min_recall": 0.9
  },
  "best_checkpoint": {
    "checkpoint": "...",
    "passed_constraints": true,
    "accuracy": 0.98,
    "precision": 0.98,
    "recall": 0.98,
    "f1": 0.98,
    "confusion_matrix": [[52, 1], [1, 46]],
    "classification_report": "..."
  },
  "all_results": [
    {
      "checkpoint": "...",
      "passed_constraints": true,
      "accuracy": 0.98,
      "precision": 0.98,
      "recall": 0.98,
      "f1": 0.98,
      "confusion_matrix": [[52, 1], [1, 46]],
      "classification_report": "..."
    }
  ]
}
```

### 4. 生成任务输出文件

输出为按类别聚合的 JSON，示例：

```json
{
  "benign": ["生成结果1", "生成结果2"],
  "attack": ["生成结果3"]
}
```

---

## 六、模型加载策略说明

`evaluation.py` 按以下优先级加载模型：

1. 若 `checkpoint_path/adapter_config.json` 存在：
     - 视为 LoRA adapter，加载基座模型后再挂载 adapter。
2. 否则若提供 `ptuning_path`：
     - 按 P-Tuning 方式加载 prefix encoder 权重。
3. 否则：
     - 直接加载基座模型进行评估。

说明：
- LoRA 模式依赖 `peft`，若未安装会报错。
- 推理默认使用 GPU（`.half().cuda()`）。

---

## 七、常见问题与排查

### 1. 报错：`label_file is required for detection task`

原因：
- 检测任务未提供标签映射文件。

处理：
- 补充 `--label_file ./datasets/xxx/xxx_label.json`。

### 2. 报错：`Target label not found in label file`

原因：
- 测试集中的真实标签与 `label_file` 映射不一致。

处理：
1. 检查测试集 `assistant` 字段标签拼写。
2. 检查 `*_label.json` 是否来自同一批预处理结果。

### 3. 没有发现 checkpoint，自动回退单模型评估

现象：
- 日志提示 `No checkpoints found in ... fallback to single model evaluation.`

处理：
1. 检查 `--checkpoint_root` 路径是否正确。
2. 检查子目录命名是否匹配 `checkpoint_pattern` 且形如 `checkpoint-*`。

### 4. 显存不足（CUDA OOM）

处理建议：
1. 降低 `--max_samples`。
2. 减小 `--max_new_tokens`。
3. 评估时关闭 `print_response`，避免额外开销。

### 5. 使用了去标签 test 文件导致指标异常

现象：
- 指标无意义或标签相关异常。

处理：
- 检测评估请使用 `*_test_with_label.jsonl`，不要使用去标签 `*_test.jsonl`。

---

## 八、推荐执行顺序

1. 确认测试集与标签文件来自同一预处理批次。
2. 先做单模型小样本冒烟评估（如 `max_samples=100`）。
3. 再做批量 checkpoint 全量评估并设置最小指标约束。
4. 根据 `best_checkpoint` 结果选择部署。