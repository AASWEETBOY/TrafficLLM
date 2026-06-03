# 模型可视化模块说明

本目录对应 TrafficLLM 的统一 Web 控制台（Streamlit），不仅包含双阶段推理，还整合了预处理、APEFT 训练、评估与测试数据生成能力，重点用于：

- 通过可视化页面统一管理 TrafficLLM 全流程（数据准备 -> 训练 -> 评估 -> 推理）。
- 以低门槛方式执行项目内核心脚本，并在页面中回显 stdout/stderr 与关键结果。
- 支持双阶段 LoRA 推理（任务路由 + 任务专用模型）及批量/单条联动分析。

---

## 一、文件结构

### 1. 可视化入口脚本
- `../trafficllm_server.py`
    - Streamlit 主入口，启动时会优先尝试自动切换到 `trafficllm` Conda 环境。
    - 页面内置 6 个栏目：模块总览、数据预处理、APEFT 训练、模型评估、测试数据生成、双阶段推理。
    - 支持目录树可视化、命令预览、一键执行、结果回显、评估指标图表展示。

### 2. 运行配置文件
- `../config.json`
    - `model_path`: 基座模型路径。
    - `peft_path`: LoRA 权重根路径。
    - `peft_set`: NLP 路由模型与各下游任务模型的 adapter 相对路径。
    - `tasks`: 任务名称到任务缩写（MTD/BND/WAD/AAD/EVD）映射，用于阶段一路由解析。

### 3. 目录文档
- `ReadMe.md`
    - 本模块使用与维护说明。

---

## 二、模块工作流

### 双阶段可视化推理流程：用户输入 -> 任务路由 -> 预测结果

输入：
- 预处理后的流量文本（手动模式建议以 `<packet>` 起始）

主要流程：
1. 页面初始化：
     - 检查并切换 Conda 环境（目标环境：`trafficllm`）。
     - 加载 `config.json`，初始化 tokenizer，设置主题与导航。
2. 阶段一（任务理解）：
    - 加载 `peft_set["NLP"]` 对流量进行任务分类。
    - 任务集合固定为：Malware/Botnet/Web Attack/APT/VPN 五类。
3. 任务解析与映射：
     - 通过 `_resolve_task` 将阶段一输出映射到 `tasks` 里的任务缩写。
     - 支持严格匹配 + 忽略大小写子串匹配 + 任务缩写匹配。
4. 阶段二（任务专用判别）：
     - 根据任务缩写加载对应 LoRA adapter。
     - 拼接 `preprompt(task, traffic_data)`，执行任务专用分类/检测。
5. 页面展示：
     - 自动批量模式：按任务分组展示样本结果、预期对比与正确率。
     - 手动单条模式：展示“下游任务”和“预测结果”，支持从批量结果一键带入。

说明：
- 当前推理解码模式固定为 `do_sample=True`，可在侧边栏调节 `max_length`、`top_p`、`temperature`、`repetition_penalty`。

---

## 三、使用说明

### 1. 启动命令（推荐）

在项目根目录执行：

```bash
python trafficllm_server.py
```

说明：
- 脚本会优先使用 `conda run -n trafficllm streamlit run trafficllm_server.py` 方式自举启动。
- 若未找到 `conda` 可执行文件，会回退到 `bash -lc` + `conda activate trafficllm` 方式启动。
- 也可手动启动：

```bash
streamlit run trafficllm_server.py --server.port 8501
```

启动后浏览器访问：

```text
http://localhost:8501
```

### 2. 输入要求

页面的输入在不同模块含义不同：
- 数据预处理：输入/输出目录、任务类型、数据集名、粒度等参数。
- APEFT 训练：模型路径、数据目录、Stage1/Stage2 配置与目标数据集。
- 模型评估：评估/推理脚本参数、checkpoint 路径、指标约束与结果文件路径。
- 双阶段推理（手动）：
    - `流量数据`：粘贴预处理后的单条流量文本。

在双阶段推理中，提交后返回：
- `下游任务`: 阶段一路由结果
- `预测结果`: 阶段二模型输出

### 3. 侧边栏参数

- `界面主题`
    - 支持 `海洋` / `经典` 两种样式。
- `最大长度 max_length`
    - 仅在双阶段推理模块生效，控制生成长度上限。
- `核采样 top_p`
    - 控制采样概率质量范围，默认 `0.8`。
- `温度 temperature`
    - 控制采样随机性，默认 `0.8`。
- `重复惩罚 repetition_penalty`
    - 抑制重复输出，默认 `1.2`。

---

## 四、配置说明

### 1. 模型路径约定

`config.json` 中关键字段示例：

```json
{
  "model_path": "models/glm-4-9b-chat",
  "peft_path": "models/glm-4-9b-chat-lora",
  "peft_set": {
    "NLP": "instructions/checkpoint-8500",
    "MTD": "ustc-tfc-2016_detection_packet/checkpoint-11000",
    "BND": "iscx-botnet-2014_detection_packet/checkpoint-10000",
    "WAD": "csic-2010_detection_packet/checkpoint-17000",
    "AAD": "dapt-2020_detection_packet/checkpoint-12000",
    "EVD": "iscx-vpn-2016_detection_packet/checkpoint-9000"
  },
  "tasks": {
    "Malware Traffic Detection": "MTD",
    "Botnet Detection": "BND",
    "Web Attack Detection": "WAD",
    "APT Attack Detection": "AAD",
    "Encrypted VPN Detection": "EVD"
  }
}
```

要求：
- 每个 adapter 目录下需包含 `adapter_config.json`。
- `adapter_config.json` 中的 `base_model_name_or_path` 应可被当前环境访问。

### 2. 任务映射机制

- 优先精确匹配：阶段一输出直接命中 `tasks` 的键。
- 其次宽松匹配：忽略大小写后匹配任务名子串。
- 再次缩写匹配：阶段一输出包含 `MTD/BND/WAD/AAD/EVD` 时也可解析。
- 仍失败时抛错：`Unknown downstream task response: ...`。

---

## 五、输入与输出示例

### 1. 输入示例

流量数据：

```text
<packet>: frame.len=1514 ip.src=10.0.0.1 ip.dst=10.0.0.2 tcp.dstport=443 payload=...
```

### 2. 输出示例

```text
下游任务: Malware Traffic Detection
预测结果: Zeus
```

批量模式表格示例字段：

```text
批次 | 输入流量数据 | 实际输出结果 | 预期输出结果 | 与预期值是否符合
```

---

## 六、环境依赖

建议环境：
- Linux + Conda（推荐）
- Python 3.9+
- 可用 CUDA GPU（默认 `CUDA_VISIBLE_DEVICES=0`）
- 已安装项目依赖（见项目根目录 `requirements.txt`）

关键依赖：
- `torch`
- `transformers`
- `peft`
- `streamlit`
- `altair`
- `pandas`

---

## 七、常见问题与排查

### 1. 页面启动失败

现象：
- 启动后未进入页面，或提示找不到 Streamlit/Conda。

排查：
1. 确认 Conda 环境 `trafficllm` 已创建并可激活。
2. 检查 `conda` 是否在 PATH 中，或手动执行 `conda activate trafficllm` 后再启动。
3. 安装依赖：`pip install -r requirements.txt`。

### 2. LoRA 加载失败

现象：
- 报错找不到 `adapter_config.json` 或 adapter 目录。

排查：
1. 检查 `config.json` 的 `peft_path` + `peft_set[任务]` 拼接路径。
2. 确认对应 checkpoint 目录完整。
3. 验证磁盘挂载与访问权限。

### 3. 下游任务解析失败

现象：
- `Unknown downstream task response: ...`

排查：
1. 检查阶段一模型输出是否含五类任务名或任务缩写。
2. 检查 `config.json` 中 `tasks` 键值对与训练定义是否一致。
3. 在数据或提示词层面限制阶段一输出格式，避免冗长自由文本。

### 4. 显存不足（OOM）

排查：
1. 在侧边栏调低 `max_length`。
2. 减少并发任务，关闭其他占用 GPU 的进程。
3. 确认 `CUDA_VISIBLE_DEVICES` 与实际可用 GPU 一致。

### 5. 输入后无结果或结果异常

排查：
1. 手动模式下确认 `流量数据` 非空。
2. 批量模式下确认 JSON 文件格式为 `{"samples": [...]}`，且每条样本包含 `traffic_data`。
3. 检查任务数据与 LoRA 任务是否匹配（例如 VPN 样本不要用 WAD 模型）。

---

## 八、推荐执行顺序

1. 在数据预处理模块完成阶段一/阶段二预处理，并检查生成文件预览。
2. 在 APEFT 模块完成 Stage1/Stage2 训练（或确认已有可用 checkpoint）。
3. 在模型评估模块运行 evaluation 与 inference，并可视化最佳 checkpoint 指标。
4. 在测试数据生成模块构造演示样本后，进入双阶段推理模块做批量与单条验证。