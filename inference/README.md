# 模型推理模块说明

本目录提供 TrafficLLM 的模型推理与实测能力，重点用于：

- 在评估模块筛选出表现较优的模型（或 checkpoint）后，进行实际分类任务推理验证。
- 对指定测试集执行逐样本推理，输出预测结果并计算分类指标。
- 记录推理日志与指标文件，便于复现与对比不同模型的实际落地表现。

与评估模块的定位差异：
- 评估模块（`evaluation.py`）偏向“模型筛选”，支持批量 checkpoint 遍历、约束过滤与最优模型选择。
- 推理模块（本目录）偏向“模型实测”，针对已选模型执行完整推理流程，重点关注各分类任务的实际完成效果与最终指标。

---

## 一、文件结构

### 1. 推理入口脚本
- `inference.py`
    - 推理主脚本（基于 `typer` 命令行）。
    - 支持基础模型与 LoRA adapter 自动识别加载。
    - 读取 GLM4 `messages` 测试数据并逐条生成预测。
    - 对预测结果执行分类指标统计并写出评估结果文件。

### 2. 一键运行脚本
- `inference.sh`
    - Bash 启动脚本，封装运行参数、日志目录与日志落盘。
    - 自动创建 `Logs/GLM4-inference/`，并生成时间戳日志。
    - 维护软链接 `inference_latest.log` 指向最近一次运行日志。

### 3. 日志目录
- `Logs/GLM4-inference/`
    - 保存每次推理执行日志（包含模型路径、输入路径、逐条输出与指标打印）。

---

## 二、模块工作流

### 推理实测流程：已选模型 -> 实际任务指标

输入：
- 已筛选模型路径（基座模型或 LoRA checkpoint）
- 去标签测试集（用于推理输入）
- 带标签测试集（用于真实标签对齐）
- 标签映射文件（类别 -> 数字 ID）

主要流程：
1. 加载模型与 tokenizer：
     - 若检测到 `adapter_config.json`，按 LoRA 方式加载基座模型并挂载 adapter。
     - 否则按基础模型路径直接加载。
2. 读取测试数据：
     - `test_file`：读取待推理 `messages`。
     - `target_path`：读取带标签样本中的真实类别。
3. 执行逐样本生成：
     - 使用 chat template 构造输入。
     - 关闭采样（`do_sample=False`），稳定输出分类标签。
4. 指标计算：
     - `accuracy`
     - `precision_weighted`
     - `recall_weighted`
     - `f1_weighted`
     - `confusion_matrix`
     - `classification_report`
5. 结果写出：
     - 将上述指标写入 `evaluation_result_file`（文本格式）。
     - 推理过程与结果同步打印到控制台/日志。

说明：
- 当前脚本默认对前 100 条样本进行推理与统计（见 `inference.py` 中切片逻辑）。

---

## 三、与评估模块的关系与差异

建议流程：
1. 先用评估模块（`evaluation.py`）对多个 checkpoint 做批量评估与约束筛选。
2. 选出性能较优模型（如约束下 F1 最优 checkpoint）。
3. 再用本推理模块对该模型做实际任务推理实测，观察最终分类任务完成指标与日志细节。

差异对比：
- 评估模块：
    - 面向模型比较与筛选。
    - 支持批量 checkpoint 自动遍历、最优模型选择。
    - 输出批量对比结果（JSON 汇总）。
- 推理模块：
    - 面向单个已选模型的实测验证。
    - 强调推理过程记录与实际分类表现确认。
    - 输出单次推理指标文本与运行日志。

---

## 四、使用说明

### 1. 使用 Bash 脚本运行（推荐）

```bash
cd inference
bash inference.sh \
    ../models/glm-4-9b-chat-lora \
    ../datasets/csic-2010/GLM4_csic-2010_detection_packet_test.jsonl \
    ../datasets/csic-2010/GLM4_csic-2010_detection_packet_test_with_label.jsonl \
    ../datasets/csic-2010/csic-2010_label.json \
    ../evaluation/csic-2010_inference_metrics.txt
```

参数顺序与含义：
- `$1` `MODEL_DIR`: 模型目录（基座或 LoRA）。
- `$2` `TEST_FILE`: 推理输入测试集（通常为去标签版本）。
- `$3` `TARGET_PATH`: 带标签测试集（用于读取真实标签）。
- `$4` `LABEL_FILE`: 标签映射文件。
- `$5` `EVALUATION_RESULT_FILE`: 指标输出文件路径。

### 2. 直接运行 Python 脚本

```bash
cd inference
python inference.py \
    ../models/glm-4-9b-chat-lora \
    ../datasets/dapt-2020/GLM4_dapt-2020_detection_packet_test.jsonl \
    ../datasets/dapt-2020/GLM4_dapt-2020_detection_packet_test_with_label.jsonl \
    ../datasets/dapt-2020/dapt-2020_label.json \
    ../evaluation/dapt-2020_inference_metrics.txt
```

---

## 五、输入与输出约定

### 1. 输入文件

#### 推理输入测试集（`test_file`）
推荐使用去标签版本（assistant 为 `-`）：

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"-"}]}
```

#### 带标签测试集（`target_path`）
用于提取真实标签：

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"benign"}]}
```

#### 标签映射（`label_file`）

```json
{
  "benign": 0,
  "attack": 1
}
```

### 2. 输出文件

#### 指标文件（`evaluation_result_file`）
文本内容包含：
- 错误标签输出记录（若有）
- `acc`
- `precision`
- `recall`
- `f1`
- `metrics_json`
- `confusion matrix`
- `classification report`

#### 运行日志（`Logs/GLM4-inference/`）
每次运行生成：
- `inference_YYYYMMDD_HHMMSS.log`
- `inference_latest.log`（软链接到最近一次日志）

---

## 六、环境依赖

建议环境：
- Python 3.9
- 可用 CUDA GPU（推荐）
- 已安装项目依赖（见项目根目录 `requirements.txt`）

关键依赖：
- `torch`
- `transformers`
- `peft`
- `pandas`
- `scikit-learn`
- `typer`
- `tqdm`

---

## 七、常见问题与排查

### 1. 报错：测试文件不存在

现象：
- `Provided Test file does not exist ...`

排查：
1. 检查 `test_file` 路径是否正确。
2. 确认运行目录是否为 `inference/`。
3. 建议使用绝对路径或从仓库根目录重新组织相对路径。

### 2. 指标异常偏低或标签无法匹配

现象：
- 日志出现 `generated mistake labels`。

原因：
- 模型输出标签文本与 `label_file` 中键值不一致。

排查：
1. 检查模型是否输出了额外前后缀或解释文本。
2. 检查 `label_file` 是否与当前测试集同源。
3. 必要时在推理后增加输出清洗规则（如去除多余描述，仅保留类别标签）。

### 3. 显存不足（OOM）

排查：
1. 降低并发或切换更小模型。
2. 确认 `CUDA_VISIBLE_DEVICES` 设置正确。
3. 关闭其他占用 GPU 的进程。

### 4. LoRA 模型加载失败

排查：
1. 检查模型目录下是否存在 `adapter_config.json`。
2. 检查 `adapter_config.json` 中 `base_model_name_or_path` 是否可访问。
3. 确认 `peft` 版本与训练环境兼容。

---

## 八、推荐执行顺序

1. 先在评估模块中完成批量 checkpoint 评估与约束筛选。
2. 选定表现较优模型后，使用本模块执行推理实测。
3. 对比不同任务（AAD/BND/EVD/MTD/WAD）的推理日志与指标文件。
4. 将最终指标结果汇总到实验记录，作为部署前依据。