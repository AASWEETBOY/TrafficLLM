# 数据集预处理模块说明

本目录提供 TrafficLLM 的两阶段预处理流程：

- 阶段一：从原始 pcap/pcapng 中提取流量特征，构建 instruction-output 训练/测试数据。
- 阶段二：将阶段一产出的 train/test 数据集统一转换为 GLM4 的 messages 对话格式。

---

## 一、文件结构

### 1. 配置与参数
- `config.py`
    - `FlowConfig`: 流级别特征提取参数（每流包数、包长度等）。
    - `PacketConfig`: 包级别特征提取参数（tshark 字段、超时等）。
    - `DatasetConfig`: 采样与划分参数（每类采样上限、训练集比例、随机种子等）。
    - `TaskConfig`: 任务类型、粒度与指令模板配置。

### 2. 阶段一入口
- `preprocess_stage1.py`
    - 主入口脚本。
    - 支持三类任务：`detection`、`generation`、`understanding`。
    - 输入为按类别组织的原始流量目录。

### 3. 阶段一核心处理
- `preprocess_utils.py`
    - 数据集拆分（采样、打乱、训练/测试划分）。
    - instruction-output 样本构建。
    - train/test/label 的写出。
- `flow_data_preprocess.py`
    - 流级特征提取（默认 `flow bytes`）。
- `packet_data_preprocess.py`
    - 包级特征提取（默认 `traffic words`，基于 tshark 字段）。
- `specfic_dataset_utils.py`
    - USTC-TFC2016 的专用处理逻辑（Benign/Malware 双层目录结构）。

### 4. 阶段二入口
- `preprocess_stage2.py`
    - 将阶段一生成的 train/test JSON 转为 GLM4 `messages` JSONL。
    - test 集会同时生成：
        - 带标签版本（用于评估）
        - 去标签版本（assistant 输出为 `-`，用于推理）
    - 额外生成统一包级任务数据集：
        - `datasets/instructions/GLM4_instructions_train.jsonl`
        - `datasets/instructions/GLM4_instructions_test_with_label.jsonl`
        - `datasets/instructions/GLM4_instructions_test.jsonl`
        - `datasets/instructions/instructions_label.json`

### 5. 工具脚本
- `check_tshark_fields.py`
    - 检查 `config.py` 中配置字段是否受本机 tshark 支持。
- `TSHARK_FIELD_CHECK.md`
    - tshark 字段诊断说明与使用手册。

---

## 二、两阶段工作流

### 阶段一：原始流量 -> 任务数据集

输入：按类别组织的目录，每个类别下包含 pcap/pcapng 文件。  
输出：`*_train.json`、`*_test.json`（JSON Lines）以及检测任务的 `*_label.json`。

主要流程：
1. 读取参数与输入目录。
2. 按任务类型选择处理分支：
     - `detection`: 构造分类任务 instruction-output。
     - `generation`: 构造“生成某类流量”的 instruction-output。
     - `understanding`: 构造字段理解问答样本（强制 `packet`）。
3. 按 `granularity` 调用底层特征提取：
     - `flow`: 流级特征提取。
     - `packet`: 包级特征提取（默认 tshark 字段）。
4. 采样、打乱、训练/测试切分。
5. 写出训练/测试数据文件（检测任务额外写出标签映射）。

默认采样与划分（可在 `config.py` 中修改）：

- 每类最大采样：`MAX_SAMPLING_NUMBER = 1000`
- 训练集比例：`TRAINING_SAMPLE_RATIO = 0.90`

#### 任务指令样本说明

`preprocess_stage1.py` 当前不再直接处理 `instructions.json`。若需要任务指令样本（如 `instructions_train.json`、`instructions_test.json`），请先通过独立脚本或已有数据文件准备好，再交给阶段二继续转换。

### 阶段二：任务数据集 -> GLM4 messages

输入：阶段一生成的 train/test json 文件。  
输出：`GLM4_*.jsonl`，以及统一包级 instructions 数据集。

主要流程：
1. 读取阶段一样本中的 `instruction` 与 `output`。
2. 解析 instruction，拆成：
     - `system`: 任务描述
     - `user`: `<packet>:...`
3. 生成 GLM4 格式：
     - `messages[0]`: system
     - `messages[1]`: user
     - `messages[2]`: assistant
4. 对 test 集额外生成无标签版本（assistant 置为 `-`）。
5. 基于五个检测数据集的 GLM4 数据，额外生成包级统一分类 instructions：
    - train：每个任务类别随机无放回抽取 4000 条
    - test：每个任务类别随机无放回抽取 500 条
    - system 使用统一分类提示词
    - assistant 使用任务类别名（如 `Malware Traffic Detection`）

---

## 三、阶段一使用说明

### 1. 基本命令

```bash
python preprocess_stage1.py \
        --input ../datasets/backdoor \
        --dataset_name backdoor \
        --traffic_task detection \
        --granularity packet \
        --output_path ./datasets \
        --output_name backdoor
```

### 2. 参数说明

- `--input`: 原始数据集路径。
- `--dataset_name`: 数据集名称（会影响 detection 任务的指令模板选择）。
- `--traffic_task`: `detection` / `generation` / `understanding`。
- `--granularity`: `flow` / `packet`。
- `--output_path`: 输出目录。
- `--output_name`: 输出文件名前缀。

### 3. 输出文件命名

阶段一输出文件名格式：

- 训练集：`{output_name}_{traffic_task}_{granularity}_train.json`
- 测试集：`{output_name}_{traffic_task}_{granularity}_test.json`
- 标签映射（仅 detection）：`{output_name}_label.json`

### 4. 任务差异

- `detection`
    - output 是类别标签（如应用类别/行为类别）。
    - 若 `dataset_name=ustc-tfc-2016`，会走专用目录解析流程。
- `generation`
    - instruction 为“生成某类别某粒度流量”。
    - output 为真实流量特征文本。
- `understanding`
    - 强制使用 `packet` 粒度。
    - 会在最终阶段限制样本上限（训练 20000、测试 200）。

---

## 四、阶段二使用说明

`preprocess_stage2.py` 会遍历脚本目录下的 `datasets` 子目录，自动处理阶段一生成的 train/test 分割文件：文件名包含 `train` 或 `test` 且后缀为 `.json` 的文件。

### 1. 运行方式

```bash
python preprocess_stage2.py
```

### 2. 处理逻辑

#### 处理流量类 train/test 分割文件
- 解析 instruction，拆成 system 与 user 两部分
- 生成 GLM4 messages 格式：
    - `messages[0]`: system（任务描述）
    - `messages[1]`: user（`<packet>:...` 格式的流量特征）
    - `messages[2]`: assistant（输出标签）
- test 集额外生成去标签版本（assistant 置为 `-`）

#### 生成统一包级 instructions 数据集
- 数据来源：5 个检测数据集的 GLM4 格式文件
    - 训练来源：`GLM4_*_detection_packet_train.jsonl`
    - 测试来源：`GLM4_*_detection_packet_test_with_label.jsonl`
- 采样策略：
    - 训练集：每个任务类别随机无放回抽取 4000 条
    - 测试集：每个任务类别随机无放回抽取 500 条
- 样本改写规则：
    - System 统一改为分类指令提示词
    - Assistant 改为任务类别名
    - 测试集另生成 assistant 为 `-` 的去标签版本
- 任务与数据集映射来源：项目根目录 `config.json` 中的 `tasks` 与 `peft_set`

### 3. 输出结果

#### 流量类 train/test 分割文件输出
- 训练集：
    - `GLM4_xxx_train.jsonl`
- 测试集：
    - `GLM4_xxx_test_with_label.jsonl`（带标签评估）
    - `GLM4_xxx_test.jsonl`（去标签推理）

#### 统一包级 instructions 输出（`datasets/instructions`）
- `GLM4_instructions_train.jsonl`
- `GLM4_instructions_test_with_label.jsonl`
- `GLM4_instructions_test.jsonl`
- `instructions_label.json`

---

## 五、输入数据组织建议

推荐目录结构（通用 detection/generation/understanding）：

```text
dataset_root/
    class_a/
        1.pcap
        2.pcapng
    class_b/
        1.pcap
```

USTC-TFC2016 目录结构需为：

```text
USTC-TFC2016/
    Benign/
        category_x/
            *.pcap
    Malware/
        category_y/
            *.pcap
```

---

## 六、环境依赖

建议环境：
- Python 3.9
- 已安装依赖（见项目根目录 `requirements.txt`）
- 可执行的 `tshark`（用于 packet 默认模式 `traffic words`）

关键依赖：
- `scapy`
- `flowcontainer`
- `tqdm`
- `pandas`（阶段二转换使用）

---

## 七、常见问题与排查

### 1. 提取结果为 0 或样本数异常偏低

现象：
- 日志提示“成功提取 0 个数据包特征”。
- 训练/测试样本远低于预期。

排查步骤：
1. 确认 `tshark` 可用：
     - 运行 `python check_tshark_fields.py`
2. 检查字段兼容性：
     - 若有不兼容字段，执行 `python check_tshark_fields.py --update-config`
3. 检查数据目录结构：
     - 类别目录下是否确实存在 `.pcap/.pcapng`
4. 检查采样上限：
     - `MAX_SAMPLING_NUMBER` 是否设置过小

说明：
- 当前 `packet_data_preprocess.py` 中对 tshark 字段数量不一致采用了容错解析（允许小幅偏差，按可用字段处理），可提升跨数据集兼容性。

### 2. 阶段二没有输出文件

原因通常是输入文件位置不符合脚本约定。  
`preprocess_stage2.py` 默认扫描其同级目录下的 `datasets` 文件夹，请确认：
- 阶段一输出文件已放入该目录（或按需修改脚本中的输入路径）

---

## 八、推荐执行顺序

1. 先验证 tshark 字段兼容：`python check_tshark_fields.py`
2. 运行阶段一生成任务数据：`python preprocess_stage1.py ...`
3. 检查输出样本数量与标签文件是否符合预期
4. 运行阶段二生成 GLM4 格式：`python preprocess_stage2.py`
5. 用 `GLM4_*_train.jsonl` 和 `GLM4_*_test*.jsonl` 进入训练/评估流程
