# 测试数据生成模块说明（仅用于模型演示）

本模块用于基于现有任务配置与测试集数据，自动生成 TrafficLLM 两阶段推理演示所需的测试样本。

- 目标：快速构造“任务识别 + 任务回答”的联动演示输入。
- 范围：仅用于模型演示与联调，不作为正式评测集构建工具。
- 入口脚本：`generate_test_data.py`（项目根目录）。

---

## 一、文件结构

### 1. 入口脚本
- `generate_test_data.py`
    - 主入口脚本。
    - 按 `config.json` 中配置的任务自动收集数据源。
    - 支持按“每个标签固定采样数”生成多任务演示样本。

### 2. 依赖配置
- `config.json`
    - `tasks`: 任务名与任务键（task key）映射。
    - `peft_set`: 各任务键对应的微调路径（用于反推数据集名称）。

### 3. 输入数据（由预处理阶段产出）
- `datasets/<dataset_name>/GLM4_<dataset_name>_detection_packet_test_with_label.jsonl`
    - 带标签测试集，默认用于生成可校验答案的样本。
- `datasets/<dataset_name>/GLM4_<dataset_name>_detection_packet_test.jsonl`
    - 去标签测试集，可通过 `--no-label` 选项使用。
- `datasets/<dataset_name>/<dataset_name>_label.json`
    - 标签映射文件，用于保证按标签均衡采样。

### 4. 输出目录
- `generated_test_data/`
    - 每个任务输出一个 JSON 文件。
    - 文件名格式：`{task_key}_{task_name}_test_data.json`（task_name 会做小写与符号清洗）。

---

## 二、工作流程

### 阶段一：读取任务配置

1. 读取 `config.json`，获取 `tasks` 与 `peft_set`。
2. 从 `peft_set` 路径中反推数据集名（如 `ustc-tfc-2016_detection_packet/...` -> `ustc-tfc-2016`）。

### 阶段二：按标签均衡采样并生成样本

1. 读取各任务对应的 GLM4 测试 JSONL 与标签映射。
2. 按 `label.json` 中的标签列表分组流量样本。
3. 每个标签采样 `N` 条（`N = --samples-per-label`）：
     - 若样本充足：无放回采样。
     - 若样本不足：有放回采样补齐。
4. 基于流量记录组装两阶段格式（两个阶段共用同一条 `traffic_data`）：
  - `user_input.traffic_data`
     - `expected_output.stage1_task_*`
     - `expected_output.stage2_answer`
5. 写出每任务独立 JSON 文件，并在 `meta` 记录采样参数。

---

## 三、使用说明

### 1. 基本命令

```bash
python generate_test_data.py
```

默认行为：
- 每个标签采样 3 条。
- 随机种子 42。
- 输出目录 `generated_test_data/`。
- 使用带标签测试集 `*_test_with_label.jsonl`。

### 2. 常用命令示例

#### 指定每标签采样数与随机种子

```bash
python generate_test_data.py \
        --samples-per-label 5 \
        --seed 2026
```

#### 输出到自定义目录

```bash
python generate_test_data.py \
        --output-dir demo_samples
```

#### 使用去标签测试集（演示占位答案场景）

```bash
python generate_test_data.py --no-label
```

### 3. 参数说明

- `--samples-per-label`
    - 类型：`int`
    - 默认：`3`
    - 含义：每个任务中每个标签抽取的样本数，必须大于 0。

- `--seed`
    - 类型：`int`
    - 默认：`42`
    - 含义：随机种子，保证可复现采样结果。

- `--output-dir`
    - 类型：`str`
    - 默认：`generated_test_data`
    - 含义：输出目录（相对项目根目录）。

- `--no-label`
    - 类型：开关参数
    - 默认：关闭
    - 含义：改用 `*_test.jsonl`（非带标签版本）作为流量输入源。

---

## 四、输出格式说明

每个任务输出文件结构如下：

```json
{
  "meta": {
    "seed": 42,
    "samples_per_label": 3,
    "task_name": "Web Attack Detection",
    "task_key": "WAD",
    "total_samples": 15,
    "traffic_source": "*_test_with_label.jsonl"
  },
  "samples": [
    {
      "sample_id": "WAD_0001",
      "task_name": "Web Attack Detection",
      "task_key": "WAD",
      "user_input": {
        "traffic_data": "<packet>: ..."
      },
      "expected_output": {
        "stage1_task_name": "Web Attack Detection",
        "stage1_task_key": "WAD",
        "stage2_answer": "SQL Injection"
      }
    }
  ]
}
```

字段说明：
- `meta`: 记录本次生成参数与来源信息。
- `samples`: 样本列表。
- `sample_id`: 任务键 + 四位递增编号。
- `user_input.traffic_data`: 来自 GLM4 测试集中的 user 消息内容，供两阶段共用。
- `expected_output.stage1_task_*`: 期望阶段一任务识别结果。
- `expected_output.stage2_answer`: 期望阶段二回答（标签或占位内容）。

---

## 五、适用边界与注意事项

1. 本模块定位为演示样本生成，不替代正式评估集构建流程。
2. 若使用 `--no-label`，`stage2_answer` 可能为占位符，不能直接用于精确指标评测。
3. 样本总量与标签分布受原始测试集规模限制：
     - 当某标签样本过少时，会触发有放回采样，可能出现重复流量记录。
4. 输入仅保留 `traffic_data`，不再额外拼接人类指令字段。

---

## 六、常见问题与排查

### 1. 报错找不到数据文件

现象：
- `Dataset file not found` 或 `Label file not found`。

排查步骤：
1. 检查 `config.json` 中 `peft_set` 路径格式是否为：`<dataset>_detection_packet/...`。
2. 检查对应目录下是否存在：
     - `GLM4_<dataset>_detection_packet_test_with_label.jsonl`（或 `test.jsonl`）
     - `<dataset>_label.json`
3. 确认已完成 `preprocess_stage2.py` 生成 GLM4 测试文件。

### 2. 报错无可用流量记录

现象：
- `No traffic records found for task`

排查步骤：
1. 检查目标数据集测试文件是否为空或格式异常。
2. 检查数据文件中 `messages` 是否包含 `user` 内容。

### 3. 标签采样时报错“某标签无记录”

原因：
- 标签文件中存在类别，但测试集中没有对应 assistant 内容。

建议：
1. 对齐 `*_label.json` 与 `GLM4_*_test_with_label.jsonl` 的标签命名。
2. 重新检查阶段一标签映射与阶段二转换过程。

---

## 七、推荐执行顺序

1. 先完成预处理并生成 GLM4 测试文件（含带标签版本）。
2. 检查 `config.json` 中任务配置与 `peft_set` 路径格式。
3. 执行测试数据生成：`python generate_test_data.py`。
4. 抽查 `generated_test_data/*.json` 中 `meta` 与样本字段是否符合预期。
5. 将生成文件用于模型演示或联调验证。