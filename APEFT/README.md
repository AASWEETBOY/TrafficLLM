# APEFT 自适应参数高效微调模块说明

本目录提供 TrafficLLM 的 APEFT（Adaptive Parameter-Efficient Fine-Tuning）两阶段训练编排脚本，统一管理：

- 阶段一（Stage1）：任务指令微调（instructions 数据集）
- 阶段二（Stage2）：面向具体流量数据集的任务适配微调（如 `csic-2010`、`dapt-2020`）

该模块通过 `apeft.py` 对 `FT/train_stage1.sh` 与 `FT/train_stage2.sh` 进行封装，支持：

- 一键串行执行 Stage1 + Stage2
- Stage1 已完成自动跳过（可强制重跑）
- 自动列出可用 Stage2 数据集
- 兼容历史参数名（`task_name` / `dataset` / `datasets`）

---

## 一、文件结构

### 1. APEFT 入口脚本
- `apeft.py`
    - 模块主入口（基于 `fire` 命令行接口）。
    - 负责训练流程编排、数据集合法性校验、阶段执行与日志输出。

### 2. 训练执行脚本（位于 `FT/`）
- `../FT/train_stage1.sh`
    - 执行 Stage1 指令微调。
    - 单卡默认使用 `python finetune.py`（避免单卡 `torchrun` 潜在崩溃）。
- `../FT/train_stage2.sh`
    - 执行指定数据集的 Stage2 微调。
    - 会基于模板配置自动生成运行时配置：`configs/lora_stage2.<dataset>.runtime.yaml`。

### 3. 配置文件（位于 `FT/configs/`）
- `../FT/configs/lora_stage1.yaml`
    - Stage1 默认配置模板。
- `../FT/configs/lora_stage2.yaml`
    - Stage2 默认配置模板。
- `../FT/configs/lora_stage2.<dataset>.runtime.yaml`
    - Stage2 运行时动态配置（脚本自动生成）。

### 4. 模型与日志目录
- `../models/glm-4-9b-chat/`
    - 基座模型默认路径。
- `../models/glm-4-9b-chat-lora/instructions/`
    - Stage1 LoRA 输出目录（用于 Stage1 完成检测）。
- `../models/glm-4-9b-chat-lora/<dataset>_detection_packet/`
    - Stage2 LoRA 输出目录。
- `../FT/Logs/`
    - 训练日志输出目录（Stage1/Stage2 分目录保存）。

---

## 二、两阶段工作流

### 阶段一：任务指令微调（Stage1）

输入：
- 数据目录：`datasets/instructions`
- 基座模型：`models/glm-4-9b-chat`
- 配置：`FT/configs/lora_stage1.yaml`

输出：
- LoRA 适配器目录：`models/glm-4-9b-chat-lora/instructions`
- 日志：`FT/Logs/GLM4-TASK-tuning_stage1/`

主要流程：
1. 检查 Stage1 是否已完成（检测以下任一条件）：
     - `adapter_config.json` 存在
     - `trainer_state.json` 存在
     - 存在 `checkpoint-*`
2. 若已完成且未开启强制模式，则跳过 Stage1。
3. 否则调用 `train_stage1.sh` 执行训练。

### 阶段二：数据集适配微调（Stage2）

输入：
- 数据目录：`datasets/<dataset>/`
- 关键文件：
    - `GLM4_<dataset>_detection_packet_train.jsonl`
    - `GLM4_<dataset>_detection_packet_test_with_label.jsonl`
- 配置模板：`FT/configs/lora_stage2.yaml`

输出：
- 运行时配置：`FT/configs/lora_stage2.<dataset>.runtime.yaml`
- LoRA 适配器目录：`models/glm-4-9b-chat-lora/<dataset>_detection_packet`
- 日志：`FT/Logs/GLM4-TASK-tuning_stage2-<dataset>/`

主要流程：
1. 列举 `datasets/` 下全部子目录并过滤 `instructions`，得到可用 Stage2 数据集。
2. 校验目标数据集是否合法。
3. 校验 Stage2 所需 JSONL 文件是否存在。
4. 生成针对该数据集的运行时 YAML。
5. 调用 `train_stage2.sh` 执行训练。

---

## 三、使用说明

### 1. 常用命令

```bash
# 列出所有可用 Stage2 数据集
python APEFT/apeft.py --list_datasets=True

# 默认执行 Stage1，然后执行指定数据集 Stage2
python APEFT/apeft.py --dataset=csic-2010

# 跳过 Stage1，仅执行 Stage2
python APEFT/apeft.py --dataset=dapt-2020 --run_stage1=False

# 强制重跑 Stage1，再执行 Stage2
python APEFT/apeft.py --dataset=iscx-vpn-2016 --force_stage1=True

# 使用历史参数名（向后兼容）
python APEFT/apeft.py --task_name=ustc-tfc-2016
```

### 2. 主要参数说明

- `--model_name`
    - 基座模型路径，默认：`models/glm-4-9b-chat`
- `--tuning_data`
    - 数据集根目录，默认：`datasets`
- `--dataset`
    - Stage2 目标数据集名称（推荐参数）
- `--datasets`
    - 历史兼容参数，可替代 `dataset`
- `--task_name`
    - 历史兼容参数，也可用于指定 Stage2 数据集
- `--run_stage1`
    - 是否执行 Stage1，默认 `True`
- `--force_stage1`
    - 是否强制重跑 Stage1，默认 `False`
- `--list_datasets`
    - 是否仅列出可用 Stage2 数据集并退出
- `--stage1_config`
    - Stage1 配置文件，默认：`configs/lora_stage1.yaml`
- `--stage2_config`
    - Stage2 配置模板，默认：`configs/lora_stage2.yaml`

### 3. 参数优先级与兼容逻辑

在指定 Stage2 数据集时，`apeft.py` 的选择顺序为：

1. `dataset`
2. `datasets`
3. `task_name`

若以上均未指定：
- 仅执行（或跳过）Stage1，并打印可用数据集列表提示。

---

## 四、目录与数据准备要求

### 1. 数据目录结构

```text
datasets/
    instructions/
        GLM4_instructions_train.jsonl
        GLM4_instructions_test_with_label.jsonl
        ...
    csic-2010/
        GLM4_csic-2010_detection_packet_train.jsonl
        GLM4_csic-2010_detection_packet_test_with_label.jsonl
    dapt-2020/
        GLM4_dapt-2020_detection_packet_train.jsonl
        GLM4_dapt-2020_detection_packet_test_with_label.jsonl
```

说明：
- Stage2 仅支持非 `instructions` 数据集。
- Stage2 默认按 `detection_packet` 命名规则读取 JSONL 文件。

### 2. 训练产物目录

```text
models/
    glm-4-9b-chat/
    glm-4-9b-chat-lora/
        instructions/
        csic-2010_detection_packet/
        dapt-2020_detection_packet/
```

---

## 五、日志与运行时配置

### 1. 日志位置

- Stage1：`FT/Logs/GLM4-TASK-tuning_stage1/`
- Stage2：`FT/Logs/GLM4-TASK-tuning_stage2-<dataset>/`

每次运行会生成：
- `train_YYYYmmdd_HHMMSS.log`
- `train_latest.log`（软链接，指向最近一次日志）

### 2. 运行时配置

Stage2 启动前会自动生成：
- `FT/configs/lora_stage2.<dataset>.runtime.yaml`

其核心变更包括：
- `data_config.train_file`
- `data_config.val_file`
- `data_config.test_file`
- `training_args.output_dir`

用于确保同一模板可复用于不同数据集。

---

## 六、环境依赖

建议环境：
- Python 3.9
- 可用 CUDA 环境（按实际 GPU 配置）

关键依赖：
- `fire`
- `ruamel.yaml`
- `torch`
- `transformers`
- `datasets`
- `peft`
- 其余依赖见项目根目录 `requirements.txt`

说明：
- 当前训练脚本在单卡场景默认走 `python finetune.py`，避免单卡 `torchrun` 在部分环境下出现稳定性问题。

---

## 七、常见问题与排查

### 1. Stage1 被跳过

现象：
- 控制台提示：`[SKIP] stage1 already done: ...`

原因：
- Stage1 输出目录中已存在 `adapter_config.json`、`trainer_state.json` 或 `checkpoint-*`。

处理：
- 强制重跑：`python APEFT/apeft.py --force_stage1=True --dataset=<name>`

### 2. 提示数据集非法

现象：
- `Invalid dataset '<name>'...`

原因：
- 指定的数据集不在 `datasets/` 子目录列表中，或为 `instructions`。

处理：
1. 运行 `python APEFT/apeft.py --list_datasets=True` 查看可用集合。
2. 确认数据目录位于 `datasets/<name>/`。

### 3. Stage2 报缺少 JSONL 文件

现象：
- `expected file not found: ...`

原因：
- 缺失以下任一文件：
    - `GLM4_<dataset>_detection_packet_train.jsonl`
    - `GLM4_<dataset>_detection_packet_test_with_label.jsonl`

处理：
1. 先完成预处理阶段二，确保 GLM4 格式文件生成。
2. 检查文件名是否与脚本约定一致。

### 4. 训练启动后中断或显存不足

排查建议：
1. 检查 `FT/configs/lora_stage1.yaml` / `lora_stage2.yaml` 的 batch size、max length、max steps。
2. 优先降低 `per_device_train_batch_size` 或序列长度。
3. 查看对应 `Logs/.../train_latest.log` 获取第一条报错位置。

---

## 八、推荐执行顺序

1. 确认 `datasets/` 下 Stage1 与目标 Stage2 数据已准备完成。
2. 列出可用数据集：`python APEFT/apeft.py --list_datasets=True`。
3. 首次建议执行：`python APEFT/apeft.py --dataset=<name>`（自动处理 Stage1 + Stage2）。
4. 多数据集迭代时，可复用 Stage1：`python APEFT/apeft.py --dataset=<name> --run_stage1=False`。
5. 训练后检查 `models/glm-4-9b-chat-lora/` 产物与 `FT/Logs/` 日志，进入评估/推理流程。