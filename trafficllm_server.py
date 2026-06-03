import os
import sys
import subprocess
import shutil
import shlex
import re


def _ensure_conda_environment():
    # 确保当前脚本运行在 trafficllm conda 环境中，避免依赖不一致。
    target_env = "trafficllm"
    bootstrap_flag = "TRAFFICLLM_CONDA_BOOTSTRAPPED"

    if os.environ.get(bootstrap_flag) == "1":
        return

    current_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if current_env == target_env:
        os.environ[bootstrap_flag] = "1"
        return

    script_path = os.path.abspath(__file__)
    env = os.environ.copy()
    env[bootstrap_flag] = "1"

    conda_executable = shutil.which("conda")
    if conda_executable:
        # 优先使用 conda run 直接重启当前脚本。
        os.execvpe(
            conda_executable,
            [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                target_env,
                "streamlit",
                "run",
                script_path,
            ],
            env,
        )

    bootstrap_command = (
        # 兼容不同 conda 安装路径，找不到则按 bash 激活方式兜底。
        f'source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || '
        f'source "$HOME/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || '
        f'source "$HOME/mambaforge/etc/profile.d/conda.sh" 2>/dev/null || true; '
        f'conda activate {target_env} && streamlit run {shlex.quote(script_path)}'
    )
    os.execvpe("bash", ["bash", "-lc", bootstrap_command], env)


_ensure_conda_environment()

from transformers import AutoModel, AutoTokenizer
from peft import PeftModelForCausalLM
import streamlit as st
import altair as alt
import torch
import json
import inspect
import pandas as pd
from pathlib import Path
import zipfile

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
SERVER_WORKDIR = str(Path(__file__).resolve().parent)

with open("config.json", "r", encoding="utf-8") as fin:
    config = json.load(fin)

st.set_page_config(
    page_title="TrafficLLM AI 系统",
    page_icon=":robot:",
    layout='wide'
)


def _resolve_task(task_response):
    # Stage1 可能返回自然语言，先精确匹配，再做模糊匹配。
    if task_response in config["tasks"]:
        return config["tasks"][task_response]

    normalized = task_response.strip().lower()
    for task_name, task_key in config["tasks"].items():
        if task_name.lower() in normalized:
            return task_key

    for task_name, task_key in config["tasks"].items():
        if task_key.lower() in normalized:
            return task_key

    raise ValueError(f"Unknown downstream task response: {task_response}")


def _supports_kwarg(fn, kwarg_name):
    # 不同模型 chat 签名可能不同，这里做可选参数兼容判断。
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False

    for param in signature.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return kwarg_name in signature.parameters


def _infer_with_model(model, prompt, max_new_tokens=512, repetition_penalty=1.2):
    # 分支1：支持 chat 接口的模型（如部分 GLM 实现）。
    if hasattr(model, "chat"):
        chat_kwargs = {
            "history": [],
            "max_length": max_length,
        }
        if do_sample:
            chat_kwargs["top_p"] = top_p
            chat_kwargs["temperature"] = temperature
        else:
            # 关闭采样时强制使用确定性解码。
            chat_kwargs["top_p"] = 1.0
            chat_kwargs["temperature"] = 0.0

        if _supports_kwarg(model.chat, "repetition_penalty"):
            chat_kwargs["repetition_penalty"] = repetition_penalty
        eos_token_id = getattr(model.config, "eos_token_id", None)
        if eos_token_id is not None and _supports_kwarg(model.chat, "eos_token_id"):
            chat_kwargs["eos_token_id"] = eos_token_id

        response, _ = model.chat(
            tokenizer,
            prompt,
            **chat_kwargs
        )
        return response

    messages = [{"role": "user", "content": prompt}]
    # 分支2：通用 generate 流程，使用 chat template 构造输入。
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    )

    model_device = next(model.parameters()).device
    inputs = inputs.to(model_device)

    generate_kwargs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": repetition_penalty,
        "eos_token_id": model.config.eos_token_id,
    }
    if do_sample:
        generate_kwargs["top_p"] = top_p
        generate_kwargs["temperature"] = temperature

    generated_ids = model.generate(
        **generate_kwargs
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return response


def preprompt(task, traffic_data):
    """Preprompts in LLMs for downstream traffic pattern learning"""
    prepromt_set = {
        "MTD": "Given the following traffic data <packet> that contains protocol fields, traffic features, and "
               "payloads. Please conduct the ENCRYPTED MALWARE DETECTION TASK to determine which application "
               "category the encrypted beign or malicious traffic belongs to. The categories include 'BitTorrent, "
               "FTP, Facetime, Gmail, MySQL, Outlook, SMB, Skype, Weibo, WorldOfWarcraft,Cridex, Geodo, Htbot, Miuref, "
               "Neris, Nsis-ay, Shifu, Tinba, Virut, Zeus'.\n",
        "BND": "Given the following traffic data <packet> that contains protocol fields, traffic features, "
               "and payloads. Please conduct the BOTNET DETECTION TASK to determine which type of network the "
               "traffic belongs to. The categories include 'IRC, Neris, RBot, Virut, normal'.\n",
        "WAD": "Classify the given HTTP request into benign and malicious categories. Each HTTP request will consist "
               "of three parts: method, URL, and body, presented in JSON format. If a web attack is detected in an "
               "HTTP request, please output an 'exception'. Only output 'malicious' or 'benign', no additional output "
               "is required. The given HTTP request is as follows:\n",
        "AAD": "Given the following traffic data <packet> that contains protocol fields, traffic features, "
               "and payloads. Please conduct the APT DETECTION TASK to determine which behavior or application category "
               "the traffic belongs to under the APT attacks. The categories include 'APT and normal'.\n",
        "EVD": "Given the following traffic data <packet> that contains protocol fields, traffic features, "
               "and payloads. Please conduct the encrypted VPN detection task to determine which behavior or "
               "application category the VPN encrypted traffic belongs to. The categories include 'aim, bittorrent, "
               "email, facebook, ftps, hangout, icq, netflix, sftp, skype, spotify, vimeo, voipbuster, youtube'.\n"
    }

    prompt = prepromt_set[task] + traffic_data

    return prompt


def dual_stage_inference(traffic_data):

    # 阶段1：先判断当前流量属于哪个下游任务。
    nlp_prompts = (
        "Given the following traffic data <packet>. Please conduct the CLASSIFICATION TASK to determine "
        "which category the traffic belongs to. The categories include 'Malware Traffic Detection, Botnet "
        "Detection, Web Attack Detection, APT Attack Detection, Encrypted VPN Detection'."
    )
    stage1_prompt = f"{nlp_prompts}\n{traffic_data}".strip()
    model_nlp = get_lora_model(config["peft_set"]["NLP"])
    task_response = _infer_with_model(
        model_nlp,
        stage1_prompt,
        max_new_tokens=512,
        repetition_penalty=repetition_penalty,
    )
    print("Downstream task: " + task_response)

    # 阶段2：根据任务路由到对应 LoRA 适配器进行分类/识别。
    task = _resolve_task(task_response)
    model_downstream = get_lora_model(config["peft_set"][task])

    traffic_prompt = preprompt(task, traffic_data)
    final_response = _infer_with_model(
        model_downstream,
        traffic_prompt,
        max_new_tokens=512,
        repetition_penalty=repetition_penalty,
    )
    print("Predicted result: " + final_response)

    return task_response, final_response


@st.cache_resource
def get_tokenizer():
    # 分词器较重，使用 Streamlit 资源缓存避免重复加载。
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_path"],
        trust_remote_code=True,
        encode_special_tokens=True,
        use_fast=False
    )

    return tokenizer


@st.cache_resource
def get_lora_model(adapter_relative_path):
    # 以 adapter_relative_path 作为缓存键，同一适配器只加载一次。
    adapter_path = os.path.join(config["peft_path"], adapter_relative_path)
    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")

    if not os.path.exists(adapter_config_path):
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")

    with open(adapter_config_path, "r", encoding="utf-8") as fin:
        adapter_config = json.load(fin)

    # 优先读取适配器中记录的基座模型，缺省时回退到全局配置。
    base_model_name_or_path = adapter_config.get("base_model_name_or_path", config["model_path"])
    model = AutoModel.from_pretrained(
        base_model_name_or_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model = PeftModelForCausalLM.from_pretrained(
        model=model,
        model_id=adapter_path,
        trust_remote_code=True,
    )
    model = model.eval()

    return model


tokenizer = get_tokenizer()
do_sample = True
max_length = 8192
top_p = 0.8
temperature = 0.8
repetition_penalty = 1.2


def _normalize_path(path_text):
    # 将输入路径统一规范为绝对路径，便于后续跨模块复用。
    if not path_text:
        return os.getcwd()
    return os.path.abspath(os.path.expanduser(path_text.strip()))


def _resolve_path_relative_to_base(base_dir, path_text):
    if not path_text:
        return os.path.abspath(base_dir)

    expanded = os.path.expanduser(path_text.strip())
    if os.path.isabs(expanded):
        return os.path.abspath(expanded)
    return os.path.abspath(os.path.join(base_dir, expanded))


def _build_tree_lines(root_path, max_depth=3):
    # 构造目录树文本，用于页面中可视化展示。
    root = Path(root_path)
    if not root.exists():
        return [f"[不存在] {root_path}"]

    lines = [f"{root.name}/"]

    def walk(current_path, prefix, depth):
        if depth > max_depth:
            return

        try:
            children = sorted(current_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            lines.append(prefix + "└── [无权限访问]")
            return

        for idx, child in enumerate(children):
            is_last = idx == len(children) - 1
            connector = "└── " if is_last else "├── "
            display_name = f"{child.name}/" if child.is_dir() else child.name
            lines.append(prefix + connector + display_name)

            if child.is_dir() and depth < max_depth:
                next_prefix = prefix + ("    " if is_last else "│   ")
                walk(child, next_prefix, depth + 1)

    walk(root, "", 1)
    return lines


def _run_shell_command(command, cwd, input_text=None):
    # 统一命令执行入口，确保 stdout/stderr 捕获格式一致。
    return subprocess.run(
        command,
        cwd=cwd,
        shell=True,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        input=input_text,
    )


def _is_preview_text_file(file_name):
    lower_name = file_name.lower()
    return lower_name.endswith((".jsonl", ".json", ".txt", ".csv"))


def _collect_stage1_preview_files(output_root):
    # 阶段一主要预览包含 train/test 的输出文件。
    if not os.path.isdir(output_root):
        return []

    preview_files = []
    for root, _, files in os.walk(output_root):
        for name in files:
            lower_name = name.lower()
            if ("train" in lower_name or "test" in lower_name) and _is_preview_text_file(name):
                preview_files.append(os.path.join(root, name))

    return sorted(preview_files)


def _collect_stage2_preview_files(datasets_root):
    # 阶段二主要预览 GLM4 开头的 jsonl 文件。
    if not os.path.isdir(datasets_root):
        return []

    preview_files = []
    for root, _, files in os.walk(datasets_root):
        for name in files:
            lower_name = name.lower()
            if lower_name.startswith("glm4") and lower_name.endswith(".jsonl"):
                preview_files.append(os.path.join(root, name))

    return sorted(preview_files)


def _collect_generated_test_data_preview_files(output_root):
    if not os.path.isdir(output_root):
        return []

    preview_files = []
    for root, _, files in os.walk(output_root):
        for name in files:
            if _is_preview_text_file(name):
                preview_files.append(os.path.join(root, name))

    return sorted(preview_files)


def _load_generated_test_data_samples(json_file_path):
    # 校验生成测试数据的 JSON 结构，避免页面渲染时报错。
    with open(json_file_path, "r", encoding="utf-8", errors="replace") as fin:
        payload = json.load(fin)

    if not isinstance(payload, dict):
        raise ValueError("generated test data JSON must be an object")

    samples = payload.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError("generated test data JSON missing samples array")

    return {
        "meta": payload.get("meta", {}),
        "samples": samples,
    }


def _collect_generated_test_data_json_files(output_root):
    if not os.path.isdir(output_root):
        return []

    json_files = []
    for root, _, files in os.walk(output_root):
        for name in files:
            if name.lower().endswith(".json"):
                json_files.append(os.path.join(root, name))

    return sorted(json_files)


def _truncate_display_text(text, max_len=120):
    if text is None:
        return ""
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + " ..."


def _read_file_head_lines(file_path, head_lines):
    lines = []
    with open(file_path, "r", encoding="utf-8", errors="replace") as fin:
        for _ in range(head_lines):
            line = fin.readline()
            if not line:
                break
            lines.append(line.rstrip("\n"))
    return "\n".join(lines)


def _is_supported_capture_file(file_name):
    lower_name = file_name.lower()
    return lower_name.endswith(".pcap") or lower_name.endswith(".pcapng")


def _collect_capture_files(folder_path):
    capture_files = []
    for root, _, files in os.walk(folder_path):
        for name in files:
            if _is_supported_capture_file(name):
                capture_files.append(os.path.join(root, name))
    return sorted(capture_files)


def _copy_folder_contents(src_root, dst_root):
    # 递归复制目录，若重名则自动追加序号避免覆盖。
    copied_files = []
    for root, _, files in os.walk(src_root):
        rel_dir = os.path.relpath(root, src_root)
        target_dir = dst_root if rel_dir == "." else os.path.join(dst_root, rel_dir)
        os.makedirs(target_dir, exist_ok=True)

        for name in files:
            src_path = os.path.join(root, name)
            dst_path = os.path.join(target_dir, name)
            if os.path.exists(dst_path):
                stem, ext = os.path.splitext(name)
                counter = 1
                while os.path.exists(dst_path):
                    dst_path = os.path.join(target_dir, f"{stem}_{counter}{ext}")
                    counter += 1
            shutil.copy2(src_path, dst_path)
            copied_files.append(dst_path)
    return copied_files


def _build_capture_save_path(upload_root, capture_file_name):
    base_name = os.path.basename(capture_file_name)
    parent_name = os.path.splitext(base_name)[0] or "capture"
    parent_dir = os.path.join(upload_root, parent_name)
    os.makedirs(parent_dir, exist_ok=True)

    dst_path = os.path.join(parent_dir, base_name)
    if not os.path.exists(dst_path):
        return dst_path

    stem, ext = os.path.splitext(base_name)
    counter = 1
    while True:
        candidate = os.path.join(parent_dir, f"{stem}_{counter}{ext}")
        if not os.path.exists(candidate):
            return candidate
        counter += 1


def _list_visualizable_dirs(project_root):
    root = Path(project_root)
    dirs = [str(root)]
    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if child.is_dir() and not child.name.startswith("."):
            dirs.append(str(child))
    return dirs


def _list_module_roots(project_root):
    candidates = [
        ("项目根目录", project_root),
        ("APEFT 模块", os.path.join(project_root, "APEFT")),
        ("训练模块 FT", os.path.join(project_root, "FT")),
        ("数据集目录 datasets", os.path.join(project_root, "datasets")),
        ("评估目录 evaluation", os.path.join(project_root, "evaluation")),
        ("推理目录 inference", os.path.join(project_root, "inference")),
        ("测试数据目录 generated_test_data", os.path.join(project_root, "generated_test_data")),
        ("模型目录 models", os.path.join(project_root, "models")),
    ]
    return [(label, path) for label, path in candidates if os.path.exists(path)]


def _list_apeft_module_roots(project_root):
    allowed_labels = {
        "项目根目录",
        "APEFT 模块",
        "训练模块 FT",
        "数据集目录 datasets",
        "模型目录 models",
    }
    return [
        (label, path)
        for label, path in _list_module_roots(project_root)
        if label in allowed_labels
    ]


def _list_available_stage2_datasets(datasets_root):
    root = Path(datasets_root)
    if not root.is_dir():
        return []

    names = []
    for child in root.iterdir():
        if child.is_dir() and child.name != "instructions":
            names.append(child.name)
    return sorted(names)


def _list_evaluation_result_json_files(evaluation_root):
    root = Path(evaluation_root)
    if not root.is_dir():
        return []

    return sorted(
        [str(path) for path in root.glob("*.json") if path.is_file() and "batch_eval" in path.name],
        key=lambda p: Path(p).name.lower(),
    )


def _load_best_checkpoint_from_eval_json(json_file_path):
    # 仅提取可视化模块需要的关键字段。
    with open(json_file_path, "r", encoding="utf-8") as fin:
        payload = json.load(fin)

    best_checkpoint = payload.get("best_checkpoint")
    if not isinstance(best_checkpoint, dict) or not best_checkpoint:
        raise ValueError("结果文件中未找到 best_checkpoint 字段。")

    return {
        "source_file": json_file_path,
        "constraints": payload.get("constraints", {}),
        "checkpoint": best_checkpoint.get("checkpoint", ""),
        "passed_constraints": best_checkpoint.get("passed_constraints"),
        "accuracy": best_checkpoint.get("accuracy"),
        "precision": best_checkpoint.get("precision"),
        "recall": best_checkpoint.get("recall"),
        "f1": best_checkpoint.get("f1"),
        "confusion_matrix": best_checkpoint.get("confusion_matrix"),
        "classification_report": best_checkpoint.get("classification_report", ""),
    }


def _extract_float_from_line(text, field_name):
    match = re.search(rf"^{re.escape(field_name)}\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$", text, flags=re.MULTILINE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def _parse_confusion_matrix_block(text):
    # 从纯文本评估结果中提取混淆矩阵数字块。
    block_match = re.search(r"confusion matrix:\s*(.*?)(?:\nclassification report:|$)", text, flags=re.DOTALL | re.IGNORECASE)
    if not block_match:
        return []

    matrix_lines = []
    for line in block_match.group(1).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        numbers = re.findall(r"-?\d+", stripped)
        if numbers:
            matrix_lines.append([int(num) for num in numbers])

    return matrix_lines


def _list_inference_result_txt_files(evaluation_root):
    root = Path(evaluation_root)
    if not root.is_dir():
        return []

    return sorted(
        [str(path) for path in root.glob("*.txt") if path.is_file() and "eval_metrics" in path.name],
        key=lambda p: Path(p).name.lower(),
    )


def _load_inference_metrics_from_txt(txt_file_path):
    # 兼容解析标量指标行与 metrics_json 扩展字段。
    with open(txt_file_path, "r", encoding="utf-8", errors="replace") as fin:
        content = fin.read()

    metrics_json = {}
    metrics_json_match = re.search(r"^metrics_json\s*:\s*(\{.*\})\s*$", content, flags=re.MULTILINE)
    if metrics_json_match:
        try:
            metrics_json = json.loads(metrics_json_match.group(1))
        except json.JSONDecodeError:
            metrics_json = {}

    report_match = re.search(r"classification report:(.*)$", content, flags=re.DOTALL | re.IGNORECASE)
    if report_match:
        # Keep leading spaces in the first report line so columns stay aligned in monospace rendering.
        classification_report = report_match.group(1).lstrip("\r\n").rstrip("\n")
    else:
        classification_report = ""

    return {
        "source_file": txt_file_path,
        "acc": _extract_float_from_line(content, "acc"),
        "precision": _extract_float_from_line(content, "precision"),
        "recall": _extract_float_from_line(content, "recall"),
        "f1": _extract_float_from_line(content, "f1"),
        "metrics_json": metrics_json,
        "confusion_matrix": _parse_confusion_matrix_block(content),
        "classification_report": classification_report,
    }


def _show_shell_result(result_key):
    # 标准化展示命令执行结果，减少重复 UI 代码。
    result = st.session_state.get(result_key)
    if not result:
        return

    st.markdown("##### 执行结果")
    st.write(f"执行目录: `{result['cwd']}`")
    st.write(f"命令: `{result['cmd']}`")
    if result["returncode"] == 0:
        st.success("执行成功 (return code = 0)")
    else:
        st.error(f"执行失败 (return code = {result['returncode']})")

    out_tab, err_tab = st.tabs(["标准输出 stdout", "错误输出 stderr"])
    with out_tab:
        st.code(result["stdout"] if result["stdout"].strip() else "(无输出)", language="text")
    with err_tab:
        st.code(result["stderr"] if result["stderr"].strip() else "(无输出)", language="text")


def _set_active_module(module_key):
    st.session_state.active_module = module_key


def _render_module_navigation(module_labels):
    st.markdown("#### 栏目切换")
    st.caption("点击栏目按钮切换模块，当前栏目会高亮显示。")

    nav_columns = st.columns(len(module_labels))
    for idx, (label, module_key) in enumerate(module_labels):
        with nav_columns[idx]:
            st.button(
                label,
                key=f"module_nav_btn_{module_key}",
                type="primary" if st.session_state.active_module == module_key else "secondary",
                use_container_width=True,
                on_click=_set_active_module,
                args=(module_key,),
            )


def _render_module_overview():
    st.subheader("模块总览")
    st.caption("点击卡片中的跳转按钮，可直接切换到对应模块。")

    cards = [
        ("数据预处理", "把原始流量转换为训练、测试的 GLM4 消息格式数据。", "preprocess/", "preprocess"),
        ("APEFT 训练", "两阶段参数高效微调，支持 Stage1 指令微调与 Stage2 数据集适配。", "APEFT/ + FT/", "apeft"),
        ("模型评估", "批量评估 checkpoint，筛选出最佳 checkpoint并生成相关报告。", "evaluation.py + evaluation/", "evaluation"),
        ("测试数据生成", "按任务和标签自动采样，生成演示的测试数据。", "generate_test_data.py + generated_test_data/", "generation"),
        ("双阶段推理", "在线服务模块，完成任务路由和下游推理。", "trafficllm_server.py", "inference"),
    ]

    cols = st.columns(2)
    for idx, (title, description, location, module_key) in enumerate(cards):
        with cols[idx % 2]:
            with st.container(border=True):
                st.markdown(f"#### {title}")
                st.write(description)
                st.caption(f"位置: {location}")
                st.button(
                    f"跳转到{title}",
                    key=f"overview_jump_{module_key}",
                    on_click=_set_active_module,
                    args=(module_key,),
                    use_container_width=True,
                )


def _render_apeft_module():
    st.subheader("APEFT 自适应参数高效微调模块")
    st.caption("支持 Stage1 指令微调、Stage2 数据集适配、可用数据集列表查询和命令预览执行。")

    project_root = os.getcwd()
    module_roots = _list_apeft_module_roots(project_root)
    module_root_map = {label: path for label, path in module_roots}

    with st.container(border=True):
        col_a, col_b = st.columns([1.5, 1])
        with col_a:
            st.markdown("#### 目录可视化")
            selected_root_label = st.selectbox(
                "选择目录",
                list(module_root_map.keys()),
                index=0,
                key="apeft_tree_root",
            )
            tree_depth = st.selectbox("目录层级", [2, 3, 4, 5], index=1, key="apeft_tree_depth")
        with col_b:
            st.info("推荐先查看 APEFT、FT、datasets 和 models 的目录结构，再填写下方参数。")

        selected_root = module_root_map[selected_root_label]
        tree_lines = _build_tree_lines(selected_root, max_depth=tree_depth)
        st.code("\n".join(tree_lines), language="text")

    available_datasets = _list_available_stage2_datasets(Path(project_root) / "datasets")
    default_dataset = available_datasets[0] if available_datasets else ""

    with st.container(border=True):
        st.markdown("#### 命令参数配置")

        col1, col2 = st.columns(2)
        with col1:
            apeft_workdir = st.text_input(
                "执行目录",
                value=SERVER_WORKDIR,
                help="固定为 trafficllm_server.py 的上级目录绝对路径。",
                disabled=True,
                key="apeft_workdir",
            )

            model_name = st.text_input(
                "基座模型路径 --model_name",
                value="./models/glm-4-9b-chat",
                key="apeft_model_name",
            )
            tuning_data = st.text_input(
                "数据根目录 --tuning_data",
                value="./datasets",
                key="apeft_tuning_data",
            )

        with col2:
            dataset_selector = st.selectbox(
                "Stage2 目标数据集 --dataset",
                ["(请选择)"] + available_datasets if available_datasets else ["(暂无可用数据集)"],
                index=1 if available_datasets else 0,
                key="apeft_dataset_selector",
            )
            stage1_config = st.text_input(
                "Stage1 配置文件 --stage1_config",
                value="./FT/configs/lora_stage1.yaml",
                key="apeft_stage1_config",
            )
            stage2_config = st.text_input(
                "Stage2 配置文件 --stage2_config",
                value="./FT/configs/lora_stage2.yaml",
                key="apeft_stage2_config",
            )

        col3, col4, col5 = st.columns(3)
        with col3:
            run_stage1 = st.checkbox("执行 Stage1", value=True, key="apeft_run_stage1")
        with col4:
            force_stage1 = st.checkbox("强制重跑 Stage1", value=False, key="apeft_force_stage1")
        with col5:
            list_datasets = st.checkbox("仅列出可用数据集", value=False, key="apeft_list_datasets")

        if list_datasets:
            command = "python APEFT/apeft.py --list_datasets=True"
        else:
            command_parts = ["python", "APEFT/apeft.py"]
            command_parts.append(f"--model_name={model_name}")
            command_parts.append(f"--tuning_data={tuning_data}")
            selected_dataset = dataset_selector if dataset_selector != "(请选择)" else ""
            if selected_dataset:
                command_parts.append(f"--dataset={selected_dataset}")
            command_parts.append(f"--run_stage1={'True' if run_stage1 else 'False'}")
            command_parts.append(f"--force_stage1={'True' if force_stage1 else 'False'}")
            command_parts.append(f"--stage1_config={stage1_config}")
            command_parts.append(f"--stage2_config={stage2_config}")
            command = " ".join(shlex.quote(part) for part in command_parts)

        st.markdown("##### 命令预览")
        st.code(command, language="bash")

        col_run, col_clear = st.columns([1, 1])
        with col_run:
            run_cmd = st.button("开始执行 APEFT", type="primary", key="apeft_run_cmd")
        with col_clear:
            clear_output = st.button("清空结果", key="apeft_clear_result")

        if clear_output:
            st.session_state.pop("apeft_last_result", None)

        if run_cmd:
            if not os.path.isdir(apeft_workdir):
                st.error(f"执行目录不存在: {apeft_workdir}")
            elif not list_datasets and not selected_dataset:
                st.error("请选择或输入一个 Stage2 数据集名称。")
            else:
                with st.spinner("APEFT 命令执行中，请稍候..."):
                    result = _run_shell_command(command, cwd=apeft_workdir)
                st.session_state["apeft_last_result"] = {
                    "cmd": command,
                    "cwd": apeft_workdir,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }

        _show_shell_result("apeft_last_result")


def _render_evaluation_module():
    st.subheader("模型评估模块")
    st.caption("先用 evaluation.py 按 F1 主指标筛选最佳 checkpoint，再可视化评估结果。")

    project_root = os.getcwd()
    available_datasets = _list_available_stage2_datasets(Path(project_root) / "datasets")
    default_dataset = available_datasets[0] if available_datasets else ""
    module_root_map = {label: path for label, path in _list_module_roots(project_root)}

    with st.container(border=True):
        st.markdown("#### 目录可视化")
        eval_root_label = st.selectbox(
            "选择目录",
            [label for label in module_root_map.keys() if label in {"项目根目录", "评估目录 evaluation", "数据集目录 datasets", "模型目录 models"}],
            index=0,
            key="eval_tree_root",
        )
        st.code("\n".join(_build_tree_lines(module_root_map[eval_root_label], max_depth=3)), language="text")

    with st.container(border=True):
        st.markdown("#### 1.1 evaluation.py：按 F1 及约束筛选最佳 checkpoint")
        st.caption("说明：`checkpoint_root` 为空时执行单模型评估；提供 `checkpoint_root` 时会自动扫描并排序 `checkpoint-*` 子目录执行批量评估。")

        col1, col2 = st.columns(2)
        with col1:
            eval_workdir = st.text_input(
                "执行目录",
                value=SERVER_WORKDIR,
                help="固定为 trafficllm_server.py 的上级目录绝对路径。",
                disabled=True,
                key="eval_workdir",
            )

            model_name = st.text_input(
                "model_name",
                value="./models/glm-4-9b-chat",
                help="基座模型路径或模型名（必填）。",
                key="eval_model_name",
            )
            traffic_task = st.selectbox(
                "traffic_task",
                ["detection", "generation"],
                index=0,
                help="任务类型，detection 或 generation（必填）。",
                key="eval_traffic_task",
            )
            test_file = st.text_input(
                "test_file",
                value=(
                    f"./datasets/{default_dataset}/GLM4_{default_dataset}_detection_packet_test_with_label.jsonl"
                    if default_dataset
                    else "./datasets/xxx/GLM4_xxx_detection_packet_test_with_label.jsonl"
                ),
                help="测试集文件路径（必填）。",
                key="eval_test_file",
            )
            label_file = st.text_input(
                "label_file",
                value=(
                    f"./datasets/{default_dataset}/{default_dataset}_label.json"
                    if default_dataset
                    else "./datasets/xxx/xxx_label.json"
                ),
                help="标签映射文件，仅 detection 必填。",
                key="eval_label_file",
                disabled=(traffic_task != "detection"),
            )

        with col2:
            checkpoint_root = st.text_input(
                "checkpoint_root",
                value="./models/glm-4-9b-chat-lora/csic-2010_detection_packet",
                help="LoRA checkpoint 根目录，可选；填写后启用批量评估。",
                key="eval_checkpoint_root",
            )
            checkpoint_pattern = st.text_input(
                "checkpoint_pattern",
                value="checkpoint-",
                help="checkpoint 匹配关键字，默认 checkpoint-。",
                key="eval_checkpoint_pattern",
            )
            max_samples = st.number_input(
                "max_samples",
                min_value=1,
                max_value=50000,
                value=1000,
                step=100,
                help="最大评估样本数，默认 1000。",
                key="eval_max_samples",
            )
            max_new_tokens = st.number_input(
                "max_new_tokens",
                min_value=1,
                max_value=8192,
                value=128,
                step=16,
                help="最大生成长度，默认 128。",
                key="eval_max_new_tokens",
            )
            output_result_file = st.text_input(
                "output_result_file",
                value=f"./evaluation/{default_dataset}_batch_eval_results.json" if default_dataset else "./evaluation/batch_eval_results.json",
                help="检测任务汇总输出路径，默认 batch_eval_results.json。",
                key="eval_output_result_file",
            )
            print_response = st.checkbox(
                "print_response",
                value=False,
                help="是否打印每条模型输出，默认 False。",
                key="eval_print_response",
            )

        col4, col5 = st.columns(2)
        with col4:
            min_precision = st.number_input(
                "min_precision",
                min_value=0.0,
                max_value=1.0,
                value=0.95,
                step=0.01,
                help="最小精度约束，默认 0.95。",
                key="eval_min_precision",
            )
        with col5:
            min_recall = st.number_input(
                "min_recall",
                min_value=0.0,
                max_value=1.0,
                value=0.95,
                step=0.01,
                help="最小召回率约束，默认 0.95。",
                key="eval_min_recall",
            )

        eval_cmd_parts = [
            "python",
            "evaluation.py",
            f"--model_name={model_name}",
            f"--test_file={test_file}",
            f"--traffic_task={traffic_task}",
            f"--max_samples={int(max_samples)}",
            f"--max_new_tokens={int(max_new_tokens)}",
            f"--min_precision={float(min_precision):.4f}",
            f"--min_recall={float(min_recall):.4f}",
            f"--output_result_file={output_result_file}",
            f"--print_response={'True' if print_response else 'False'}",
        ]
        if traffic_task == "detection":
            eval_cmd_parts.append(f"--label_file={label_file}")
        if checkpoint_root.strip():
            eval_cmd_parts.append(f"--checkpoint_root={checkpoint_root.strip()}")
            eval_cmd_parts.append(f"--checkpoint_pattern={checkpoint_pattern}")
        eval_command = " ".join(shlex.quote(part) for part in eval_cmd_parts)

        st.markdown("##### 命令预览")
        st.code(eval_command, language="bash")

        col_run_eval, col_clear_eval = st.columns([1, 1])
        with col_run_eval:
            run_eval = st.button("执行第1部分：筛选最佳 checkpoint", type="primary", key="run_eval_stage1")
        with col_clear_eval:
            clear_eval = st.button("清空第1部分结果", key="clear_eval_stage1")

        if clear_eval:
            st.session_state.pop("eval_stage1_last_result", None)

        if run_eval:
            if traffic_task == "detection" and not label_file.strip():
                st.error("detection 任务必须填写 label_file。")
            else:
                if not os.path.isdir(eval_workdir):
                    st.error(f"执行目录不存在: {eval_workdir}")
                else:
                    with st.spinner("evaluation.py 执行中，请稍候..."):
                        result = _run_shell_command(eval_command, cwd=eval_workdir)
                    st.session_state["eval_stage1_last_result"] = {
                        "cmd": eval_command,
                        "cwd": eval_workdir,
                        "returncode": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }

        _show_shell_result("eval_stage1_last_result")

    with st.container(border=True):
        st.markdown("#### 1.2 evaluation 结果可视化：展示最佳 checkpoint 与关键指标")
        st.caption("读取 evaluation 目录下生成的 xxx.json（如 *_batch_eval_results.json），可视化 best_checkpoint 信息。")

        evaluation_root = os.path.join(project_root, "evaluation")
        available_result_files = _list_evaluation_result_json_files(evaluation_root)
        default_result_file = output_result_file if output_result_file.strip() else ""
        default_result_path = _normalize_path(default_result_file) if default_result_file else ""

        result_file_options = ["(手动填写路径)"] + available_result_files if available_result_files else ["(手动填写路径)"]
        default_index = 0
        if default_result_path and default_result_path in available_result_files:
            default_index = result_file_options.index(default_result_path)

        col_vis_1, col_vis_2 = st.columns([1.2, 1])
        with col_vis_1:
            selected_result_file = st.selectbox(
                "选择 evaluation 结果 JSON",
                result_file_options,
                index=default_index,
                key="eval_best_ckpt_json_select",
            )
        with col_vis_2:
            manual_result_file = st.text_input(
                "或手动填写 JSON 路径",
                value="",
                key="eval_best_ckpt_json_manual",
                help="支持相对项目根目录或绝对路径。",
            )

        if selected_result_file != "(手动填写路径)":
            resolved_result_file = selected_result_file
        else:
            resolved_result_file = _normalize_path(manual_result_file) if manual_result_file.strip() else ""

        col_vis_btn_1, col_vis_btn_2 = st.columns([1, 1])
        with col_vis_btn_1:
            visualize_btn = st.button("可视化最佳 checkpoint", type="primary", key="eval_visualize_best_ckpt")
        with col_vis_btn_2:
            clear_visualize = st.button("清空可视化结果", key="eval_clear_best_ckpt")

        if clear_visualize:
            st.session_state.pop("eval_best_ckpt_view", None)

        if visualize_btn:
            if not resolved_result_file:
                st.error("请先选择或填写 evaluation 结果 JSON 文件路径。")
            elif not os.path.isfile(resolved_result_file):
                st.error(f"结果文件不存在: {resolved_result_file}")
            else:
                try:
                    st.session_state["eval_best_ckpt_view"] = _load_best_checkpoint_from_eval_json(resolved_result_file)
                except (OSError, json.JSONDecodeError, ValueError) as exc:
                    st.error(f"读取或解析失败: {exc}")

        best_ckpt_view = st.session_state.get("eval_best_ckpt_view")
        if best_ckpt_view:
            st.markdown("##### 最佳 checkpoint 信息")
            st.write(f"结果文件: `{best_ckpt_view['source_file']}`")
            st.write(f"最佳 checkpoint 路径: `{best_ckpt_view['checkpoint']}`")

            passed_constraints = best_ckpt_view.get("passed_constraints")
            if passed_constraints is True:
                st.success("已满足约束条件 (passed_constraints=True)")
            elif passed_constraints is False:
                st.warning("未满足约束条件 (passed_constraints=False)")

            metric_col_2, metric_col_3, metric_col_4 = st.columns(3)
            with metric_col_2:
                st.metric("Precision", f"{float(best_ckpt_view['precision']):.4f}" if best_ckpt_view.get("precision") is not None else "N/A")
            with metric_col_3:
                st.metric("Recall", f"{float(best_ckpt_view['recall']):.4f}" if best_ckpt_view.get("recall") is not None else "N/A")
            with metric_col_4:
                st.metric("F1", f"{float(best_ckpt_view['f1']):.4f}" if best_ckpt_view.get("f1") is not None else "N/A")

            constraints = best_ckpt_view.get("constraints") or {}
            if constraints:
                st.caption(
                    "约束条件: "
                    f"min_precision={constraints.get('min_precision', 'N/A')}, "
                    f"min_recall={constraints.get('min_recall', 'N/A')}"
                )

            st.markdown("##### 指标小图表")

            precision_value = best_ckpt_view.get("precision")
            recall_value = best_ckpt_view.get("recall")
            f1_value = best_ckpt_view.get("f1")

            min_precision_val = constraints.get("min_precision") if isinstance(constraints, dict) else None
            min_recall_val = constraints.get("min_recall") if isinstance(constraints, dict) else None

            metric_rows = []
            for metric_name, value, target in [
                ("Precision", precision_value, min_precision_val),
                ("Recall", recall_value, min_recall_val),
            ]:
                if value is not None:
                    metric_rows.append(
                        {
                            "metric": metric_name,
                            "value": float(value),
                            "target": float(target) if target is not None else 0.0,
                        }
                    )

            f1_ref_candidates = [v for v in [min_precision_val, min_recall_val] if v is not None]
            f1_reference = float(min(f1_ref_candidates)) if f1_ref_candidates else 0.0

            chart_col_1, chart_col_2 = st.columns(2)

            with chart_col_1:
                if f1_value is not None:
                    f1_data = [{"metric": "F1", "value": float(f1_value)}]
                    f1_bar = alt.Chart(alt.Data(values=f1_data)).mark_bar(size=66, color="#2f7fb8").encode(
                        x=alt.X("metric:N", title=""),
                        y=alt.Y("value:Q", title="F1", scale=alt.Scale(domain=[0, 1])),
                        tooltip=[alt.Tooltip("value:Q", title="F1", format=".4f")],
                    )
                    f1_rule = alt.Chart(alt.Data(values=[{"threshold": f1_reference}])).mark_rule(color="#e85d04", strokeDash=[6, 4]).encode(
                        y=alt.Y("threshold:Q", scale=alt.Scale(domain=[0, 1]))
                    )
                    f1_label = alt.Chart(alt.Data(values=[{"x": "F1", "threshold": f1_reference, "text": f"参考线 {f1_reference:.4f}"}])).mark_text(
                        dy=-8,
                        color="#e85d04",
                        fontSize=12,
                    ).encode(
                        x=alt.X("x:N"),
                        y=alt.Y("threshold:Q", scale=alt.Scale(domain=[0, 1])),
                        text="text:N",
                    )
                    st.altair_chart((f1_bar + f1_rule + f1_label).properties(height=260, title="F1 柱状图 + 约束参考线"), use_container_width=True)
                else:
                    st.info("F1 不存在，无法绘制 F1 小图表。")

            with chart_col_2:
                if metric_rows:
                    metric_bar = alt.Chart(alt.Data(values=metric_rows)).mark_bar(size=36, color="#49a39d").encode(
                        x=alt.X("metric:N", title=""),
                        y=alt.Y("value:Q", title="值", scale=alt.Scale(domain=[0, 1])),
                        tooltip=[
                            alt.Tooltip("metric:N", title="指标"),
                            alt.Tooltip("value:Q", title="实际值", format=".4f"),
                            alt.Tooltip("target:Q", title="约束阈值", format=".4f"),
                        ],
                    )
                    metric_rule = alt.Chart(alt.Data(values=metric_rows)).mark_tick(color="#e85d04", thickness=2, size=22).encode(
                        x=alt.X("metric:N"),
                        y=alt.Y("target:Q", scale=alt.Scale(domain=[0, 1])),
                    )
                    st.altair_chart((metric_bar + metric_rule).properties(height=260, title="Precision/Recall 对比（含阈值刻度）"), use_container_width=True)
                else:
                    st.info("Precision/Recall 缺失，无法绘制对比图。")

                st.caption("说明：F1 参考线使用 min_precision/min_recall 的最小值，作为保守阈值参考。")

            confusion_matrix = best_ckpt_view.get("confusion_matrix")
            if confusion_matrix:
                st.markdown("##### 混淆矩阵")
                st.dataframe(confusion_matrix, use_container_width=True)

            classification_report = best_ckpt_view.get("classification_report", "")
            if classification_report.strip():
                with st.expander("查看 classification_report", expanded=True):
                    st.code(classification_report, language="text")

def _render_generate_test_data_module():
    st.subheader("测试数据生成模块")
    st.caption("按任务与标签随机采样，批量生成演示用测试数据。")

    with st.container(border=True):
        st.markdown("#### 命令参数配置")
        col1, col2 = st.columns(2)
        with col1:
            gen_workdir = st.text_input(
                "执行目录",
                value=SERVER_WORKDIR,
                help="固定为 trafficllm_server.py 的上级目录绝对路径。",
                disabled=True,
                key="gen_workdir",
            )

            samples_per_label = st.number_input("每个标签采样数 --samples-per-label", min_value=1, max_value=1000, value=3, step=1, key="gen_samples")
            seed = st.number_input("随机种子 --seed", min_value=0, max_value=2**31 - 1, value=42, step=1, key="gen_seed")
        with col2:
            output_dir = st.text_input("输出目录 --output-dir", value="generated_test_data", key="gen_output_dir")
            preview_sample_count = st.selectbox(
                "预览前 n 个 samples（执行成功后展示生成文件）",
                list(range(1, int(samples_per_label) + 1)),
                index=0,
                key="gen_preview_lines",
            )

        command_parts = [
            "python",
            "generate_test_data.py",
            f"--samples-per-label={int(samples_per_label)}",
            f"--seed={int(seed)}",
            f"--output-dir={output_dir}",
        ]
        command = " ".join(shlex.quote(part) for part in command_parts)

        st.markdown("##### 命令预览")
        st.code(command, language="bash")

        col_run, col_clear = st.columns([1, 1])
        with col_run:
            run_cmd = st.button("开始生成数据", type="primary", key="gen_run_cmd")
        with col_clear:
            clear_output = st.button("清空结果", key="gen_clear_result")

        if clear_output:
            st.session_state.pop("gen_last_result", None)

        if run_cmd:
            if not os.path.isdir(gen_workdir):
                st.error(f"执行目录不存在: {gen_workdir}")
            else:
                with st.spinner("测试数据生成中，请稍候..."):
                    result = _run_shell_command(command, cwd=gen_workdir)
                st.session_state["gen_last_result"] = {
                    "cmd": command,
                    "cwd": gen_workdir,
                    "output_dir": output_dir,
                    "preview_sample_count": int(preview_sample_count),
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }

        _show_shell_result("gen_last_result")

        result = st.session_state.get("gen_last_result")
        if result and result.get("returncode") == 0:
            preview_root = _resolve_path_relative_to_base(gen_workdir, result.get("output_dir", "generated_test_data"))
            preview_sample_count = int(result.get("preview_sample_count", 1))
            preview_sample_count = max(preview_sample_count, 1)
            preview_files = _collect_generated_test_data_preview_files(preview_root)

            st.markdown("##### 生成文件内容预览")
            st.caption(f"目录: `{preview_root}`，每个文件展示前 {preview_sample_count} 个 samples。")

            if not preview_files:
                st.info("未在输出目录下找到可预览的文本文件（jsonl/json/txt/csv）。")
            else:
                for preview_file in preview_files[:10]:
                    relative_path = os.path.relpath(preview_file, preview_root)
                    st.markdown(f"**{relative_path}**")
                    try:
                        preview_payload = _load_generated_test_data_samples(preview_file)
                    except OSError as exc:
                        st.error(f"读取失败: {preview_file} ({exc})")
                        continue
                    except (json.JSONDecodeError, ValueError) as exc:
                        st.error(f"解析失败: {preview_file} ({exc})")
                        continue

                    samples = preview_payload.get("samples", [])[:preview_sample_count]
                    if not samples:
                        st.code("(无 samples)", language="text")
                    else:
                        st.code(
                            json.dumps(samples, ensure_ascii=False, indent=2),
                            language="json",
                        )

                if len(preview_files) > 10:
                    st.caption(f"共匹配 {len(preview_files)} 个文件，仅展示前 10 个。")


def _render_inference_module():
    global do_sample, max_length, top_p, temperature, repetition_penalty

    st.subheader("TrafficLLM 双阶段推理模块")
    st.caption("支持自动批量模式与手动单条模式。")

    do_sample = True
    st.sidebar.write("解码模式: do_sample=True (采样)")

    max_length = st.sidebar.slider(
        '最大长度 max_length', 0, 128000, 25600, step=1
    )
    top_p = st.sidebar.slider(
        '核采样 top_p', 0.0, 1.0, 0.8, step=0.01
    )
    temperature = st.sidebar.slider(
        '温度 temperature', 0.0, 1.0, 0.8, step=0.01
    )
    repetition_penalty = st.sidebar.slider(
        '重复惩罚 repetition_penalty', 1.0, 2.0, 1.2, step=0.01
    )

    if "inference_mode" not in st.session_state:
        st.session_state.inference_mode = "自动批量模式"
    if "manual_traffic_data" not in st.session_state:
        st.session_state.manual_traffic_data = ""
    if "manual_auto_run" not in st.session_state:
        st.session_state.manual_auto_run = False
    if "manual_expected_output" not in st.session_state:
        st.session_state.manual_expected_output = ""
    if "manual_expected_output_from_batch" not in st.session_state:
        st.session_state.manual_expected_output_from_batch = False
    if "manual_expected_output_skip_clear" not in st.session_state:
        st.session_state.manual_expected_output_skip_clear = False

    def _clear_batch_context_on_manual_switch():
        # 手动模式切换时清理批量模式带入的预期结果，避免脏状态。
        if st.session_state.get("manual_expected_output_skip_clear"):
            st.session_state.manual_expected_output_skip_clear = False
            return
        st.session_state.manual_expected_output = ""
        st.session_state.manual_expected_output_from_batch = False

    # 在渲染 radio 前先处理待切换模式，保证页面状态一致。
    pending_mode = st.session_state.pop("inference_mode_next", None)
    if pending_mode in {"自动批量模式", "手动单条模式"}:
        st.session_state.inference_mode = pending_mode

    st.radio(
        "推理方式",
        ["自动批量模式", "手动单条模式"],
        key="inference_mode",
        horizontal=True,
        on_change=_clear_batch_context_on_manual_switch,
    )

    if st.session_state.inference_mode == "自动批量模式":
        st.markdown("#### 自动批量模式")
        st.caption("数据来源于测试数据生成模块产生的 JSON，点击一键进行批量推理。")

        gen_last_result = st.session_state.get("gen_last_result")
        if gen_last_result and gen_last_result.get("output_dir"):
            default_batch_root = _resolve_path_relative_to_base(
                gen_last_result.get("cwd", SERVER_WORKDIR),
                gen_last_result.get("output_dir", "generated_test_data"),
            )
        else:
            default_batch_root = _resolve_path_relative_to_base(SERVER_WORKDIR, "generated_test_data")

        batch_root_input = st.text_input(
            "批量数据目录",
            value=default_batch_root,
            key="inference_batch_root",
            help="默认使用测试数据生成模块的输出目录。",
        )
        batch_root = _normalize_path(batch_root_input)

        batch_files = _collect_generated_test_data_json_files(batch_root)
        if not batch_files:
            st.info("当前目录下未找到可用 JSON 文件，请先在测试数据生成模块生成测试数据。")
        else:
            selected_batch_files = st.multiselect(
                "选择参与批量推理的文件",
                options=batch_files,
                default=batch_files,
                key="inference_batch_files",
                format_func=lambda p: os.path.relpath(p, batch_root),
            )

            run_batch = st.button("一键批量处理", type="primary", key="run_batch_inference")

            if run_batch:
                if not selected_batch_files:
                    st.warning("请至少选择一个测试数据文件。")
                else:
                    # 批量执行时逐文件、逐样本推理，并记录对比结果。
                    with st.spinner("批量推理中，请稍候..."):
                        batch_rows = []
                        for data_file in selected_batch_files:
                            try:
                                payload = _load_generated_test_data_samples(data_file)
                            except (OSError, json.JSONDecodeError, ValueError) as exc:
                                batch_rows.append(
                                    {
                                        "task_name": "读取失败",
                                        "batch": os.path.basename(data_file),
                                        "traffic_full": "",
                                        "traffic_preview": f"读取失败: {exc}",
                                        "actual_output": "读取失败",
                                        "expected_output_display": "(无预期结果)",
                                        "matched_bool": False,
                                    }
                                )
                                continue

                            for sample in payload.get("samples", []):
                                traffic_data = str(sample.get("user_input", {}).get("traffic_data", "")).strip()
                                sample_id = sample.get("sample_id", "")
                                expected = sample.get("expected_output", {})
                                expected_task_key = str(expected.get("stage1_task_key", "")).strip()
                                expected_task_name = str(expected.get("stage1_task_name", "")).strip()
                                expected_stage2_raw = str(expected.get("stage2_answer", "")).strip()
                                expected_stage2 = expected_stage2_raw.lower()
                                task_name = str(sample.get("task_name", "未知任务"))
                                expected_output_display = f"下游任务: {expected_task_name or expected_task_key or 'N/A'} | 结果: {expected_stage2_raw or 'N/A'}"

                                if not traffic_data:
                                    batch_rows.append(
                                        {
                                            "task_name": task_name,
                                            "batch": sample_id or "(缺少 sample_id)",
                                            "traffic_full": "",
                                            "traffic_preview": "(空流量)",
                                            "actual_output": "输入为空",
                                            "expected_output_display": expected_output_display,
                                            "matched_bool": False,
                                        }
                                    )
                                    continue

                                task_response, final_response = dual_stage_inference(traffic_data)
                                try:
                                    predicted_task_key = _resolve_task(task_response)
                                except ValueError:
                                    predicted_task_key = ""

                                stage1_ok = bool(expected_task_key) and predicted_task_key == expected_task_key
                                stage2_ok = str(final_response).strip().lower() == expected_stage2
                                # 仅当阶段1任务与阶段2结果都匹配时判定为命中。
                                matched = stage1_ok and stage2_ok

                                batch_rows.append(
                                    {
                                        "task_name": task_name,
                                        "batch": sample_id or "(缺少 sample_id)",
                                        "traffic_full": traffic_data,
                                        "traffic_preview": _truncate_display_text(traffic_data, max_len=120),
                                        "actual_output": f"下游任务: {task_response} | 结果: {final_response}",
                                        "expected_output_display": expected_output_display,
                                        "matched_bool": matched,
                                    }
                                )

                    st.session_state["inference_batch_results"] = batch_rows

            batch_results = st.session_state.get("inference_batch_results", [])
            if batch_results:
                grouped = {}
                for row in batch_results:
                    grouped.setdefault(row.get("task_name", "未知任务"), []).append(row)

                st.markdown("#### 批量结果表")
                st.caption("每个分类任务单独一个表；如需跳转到手动模式，请先选择批次再点击跳转按钮。")

                for task_name, rows in grouped.items():
                    st.markdown(f"##### {task_name}")

                    table_rows = []
                    for item in rows:
                        table_rows.append(
                            {
                                "批次": item["batch"],
                                "输入流量数据": item["traffic_preview"],
                                "实际输出结果": item["actual_output"],
                                "预期输出结果": item.get("expected_output_display", ""),
                                "与预期值是否符合": "是" if item["matched_bool"] else "否",
                            }
                        )

                    df = pd.DataFrame(table_rows)

                    def _style_batch_table(styler):
                        return styler.set_properties(
                            **{
                                "white-space": "normal",
                                "word-break": "break-word",
                                "vertical-align": "top",
                            }
                        ).set_table_styles(
                            [
                                {
                                    "selector": "th",
                                    "props": [
                                        ("white-space", "normal"),
                                        ("word-break", "break-word"),
                                        ("text-align", "left"),
                                    ],
                                },
                                {
                                    "selector": "td",
                                    "props": [
                                        ("white-space", "normal"),
                                        ("word-break", "break-word"),
                                    ],
                                },
                            ]
                        ).apply(
                            lambda col: [
                                "color: #1f7a1f; font-weight: 700;" if str(value) == "是" else "color: #c62828; font-weight: 700;"
                                for value in col
                            ] if col.name == "与预期值是否符合" else ["" for _ in col],
                            axis=0,
                        )

                    styled_df = _style_batch_table(df.style)

                    selected_rows = []
                    try:
                        # 新版 streamlit 支持 on_select，可用于点击行跳转手动模式。
                        selected = st.dataframe(
                            styled_df,
                            use_container_width=True,
                            hide_index=True,
                            on_select="rerun",
                            selection_mode="single-row",
                            key=f"batch_table_{task_name}",
                        )
                        selected_rows = selected.selection.get("rows", []) if hasattr(selected, "selection") else []
                    except TypeError:
                        st.dataframe(
                            styled_df,
                            use_container_width=True,
                            hide_index=True,
                        )

                    total_count = len(rows)
                    correct_count = sum(1 for r in rows if r["matched_bool"])
                    st.write(f"最终正确率: {correct_count}/{total_count}")

                    if selected_rows:
                        idx = selected_rows[0]
                        if 0 <= idx < len(rows):
                            selected_item = rows[idx]
                            st.session_state.inference_mode_next = "手动单条模式"
                            st.session_state.manual_traffic_data = selected_item.get("traffic_full", "")
                            st.session_state.manual_expected_output = selected_item.get("expected_output_display", "")
                            st.session_state.manual_expected_output_from_batch = True
                            st.session_state.manual_expected_output_skip_clear = True
                            st.rerun()

    else:
        st.markdown("#### 手动单条模式")
        st.caption("手动输入单条流量，或从批量模式点击批次自动带入。")

        if st.session_state.get("manual_expected_output_from_batch") and st.session_state.get("manual_expected_output"):
            st.info(f"预期结果: {st.session_state.manual_expected_output}")

        manual_traffic_data = st.text_area(
            label="流量数据",
            height=220,
            key="manual_traffic_data",
            placeholder="请输入由 TrafficLLM 预处理代码提取的流量特征数据，需以 <packet> 标识开头。",
        )

        run_manual = st.button("提交", key="predict", type="primary")
        if run_manual:
            if not manual_traffic_data.strip():
                st.error("`流量数据` 不能为空，请输入流量数据。")
            else:
                with st.spinner("推理中，请稍候..."):
                    task_response, final_response = dual_stage_inference(manual_traffic_data.strip())

                st.markdown("##### 推理结果")
                st.success(f"下游任务: {task_response}")
                st.info(f"预测结果: {final_response}")


def _render_preprocess_module():
    st.subheader("数据预处理模块")
    st.caption("支持 tshark 字段检查与阶段一/阶段二预处理，同时参数可视化配置后自动生成命令并点击执行。")

    project_root = os.getcwd()
    visualizable_dirs = _list_visualizable_dirs(project_root)
    default_preprocess = os.path.join(project_root, "preprocess")
    default_index = visualizable_dirs.index(default_preprocess) if default_preprocess in visualizable_dirs else 0

    with st.container(border=True):
        col_a, col_b, col_c = st.columns([2.2, 1.8, 1])
        with col_a:
            st.markdown("#### 目录可视化")
        with col_b:
            selected_root = st.selectbox(
                "选择可视化目录",
                visualizable_dirs,
                index=default_index,
                key="tree_root",
                format_func=lambda p: "项目根目录 (.)" if os.path.abspath(p) == os.path.abspath(project_root) else os.path.relpath(p, project_root),
            )
            st.write(f"当前目录: `{selected_root}`")
        with col_c:
            tree_depth = st.selectbox("目录深度", [2, 3, 4, 5], index=1, key="tree_depth")

        tree_lines = _build_tree_lines(selected_root, max_depth=tree_depth)
        st.code("\n".join(tree_lines), language="text")

    with st.container(border=True):
        st.markdown("#### 命令参数配置")

        mode = st.selectbox(
            "请选择预处理步骤",
            [
                "第一步：检查 tshark 字段",
                "第二步：上传数据集",
                "第三步：数据预处理阶段一——原始流量 -> 任务数据集",
                "第四步：数据预处理阶段二——任务数据集 -> GLM4格式",
            ],
            index=0,
            key="preprocess_mode",
        )

        workdir = st.text_input(
            "执行目录（命令执行路径）",
            value=SERVER_WORKDIR,
            help="固定为 trafficllm_server.py 的上级目录绝对路径。",
            disabled=True,
            key="preprocess_workdir",
        )

        command = ""
        command_desc = ""

        if mode == "第一步：检查 tshark 字段":
            # 第一步用于验证当前环境下 tshark 字段兼容性。
            st.markdown("##### tshark 字段检查参数")
            update_config = st.checkbox("自动更新 config.py 中不兼容字段 (--update-config)", value=False, key="s1_update_cfg")
            command_desc = "检查当前环境下 tshark 字段可用性。"
            command = "python preprocess/check_tshark_fields.py"
            update_confirm = True
            if update_config:
                update_confirm = st.checkbox(
                    "二次确认：我已知晓该操作会修改 config.py，确认继续执行",
                    value=False,
                    key="s1_update_cfg_confirm",
                )
                command += " --update-config"
                st.info("执行更新时，如果命令行出现 y/n 选择，系统会自动输入 Y。")
                if not update_confirm:
                    st.warning("已启用配置更新，但尚未完成二次确认，当前不会执行该命令。")

        elif mode == "第二步：上传数据集":
            # 上传步骤支持三种输入来源：文件、zip、服务器本地目录。
            st.markdown("##### 数据集上传参数")
            with st.expander("查看推荐目录结构", expanded=True):
                st.markdown("通用 detection/generation/understanding 推荐结构：")
                st.code(
                    """dataset_root（数据集名称）/
    class_a（类别1）/
        class_a_1.pcap
        class_a_2.pcapng
    class_b（类别2）/
        class_b.pcap""",
                    language="text",
                )
            st.info(
                "上传后会自动按文件名创建上一级目录。"
                "例如上传 bot.pcap，最终会保存为 目标目录/bot/bot.pcap。"
            )

            upload_target_dir = st.text_input(
                "上传保存目录",
                value="./uploaded",
                help="支持相对项目根目录或绝对路径。",
                key="dataset_upload_target",
            )
            resolved_upload_target = _normalize_path(upload_target_dir)
            upload_mode = st.radio(
                "上传方式",
                ["上传 pcap/pcapng 文件", "上传文件夹压缩包 (zip)", "导入服务器本地文件夹"],
                horizontal=True,
                key="dataset_upload_mode",
            )

            upload_files = None
            upload_zip = None
            source_folder = ""

            if upload_mode == "上传 pcap/pcapng 文件":
                upload_files = st.file_uploader(
                    "选择数据集文件（支持多文件上传）",
                    accept_multiple_files=True,
                    type=["pcap", "pcapng"],
                    key="dataset_upload_files",
                )
                st.caption("仅支持上传 pcap/pcapng 格式数据文件。")
            elif upload_mode == "上传文件夹压缩包 (zip)":
                upload_zip = st.file_uploader(
                    "选择文件夹压缩包（zip）",
                    accept_multiple_files=False,
                    type=["zip"],
                    key="dataset_upload_zip",
                )
                st.caption("请将整个数据集文件夹打包为 zip 上传，系统会自动提取其中的 pcap/pcapng 文件。")
            else:
                source_folder = st.text_input(
                    "服务器本地文件夹路径",
                    value="../raw",
                    help="用于导入服务器上已存在的数据集目录，系统会按原目录结构递归复制（相当于复制粘贴）。",
                    key="dataset_source_folder",
                )
                st.caption("此方式会直接复制整个文件夹内容到目标目录。")

            command_desc = "上传步骤通过前端保存文件，不执行命令。"
            command = "(上传步骤无需命令执行)"

            col_upload, col_hint = st.columns([1, 1])
            with col_upload:
                upload_btn = st.button("开始上传", type="primary", key="upload_dataset_btn")
            with col_hint:
                st.write(f"目标目录: `{resolved_upload_target}`")

            if upload_btn:
                os.makedirs(resolved_upload_target, exist_ok=True)

                if upload_mode == "上传 pcap/pcapng 文件":
                    if not upload_files:
                        st.warning("请先选择至少一个文件再上传。")
                    else:
                        saved_files = []
                        for uploaded in upload_files:
                            if not _is_supported_capture_file(uploaded.name):
                                continue
                            save_path = _build_capture_save_path(resolved_upload_target, uploaded.name)
                            with open(save_path, "wb") as fout:
                                fout.write(uploaded.getbuffer())
                            saved_files.append(save_path)

                        if saved_files:
                            st.success(f"上传完成，共保存 {len(saved_files)} 个文件。")
                            with st.expander("查看已保存文件", expanded=False):
                                st.code("\n".join(saved_files), language="text")
                        else:
                            st.warning("未检测到可保存的 pcap/pcapng 文件。")

                elif upload_mode == "上传文件夹压缩包 (zip)":
                    if upload_zip is None:
                        st.warning("请先选择 zip 文件再上传。")
                    else:
                        temp_zip_path = os.path.join(resolved_upload_target, upload_zip.name)
                        temp_extract_root = os.path.join(resolved_upload_target, "_tmp_zip_extract")
                        with open(temp_zip_path, "wb") as fout:
                            fout.write(upload_zip.getbuffer())

                        os.makedirs(temp_extract_root, exist_ok=True)
                        with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
                            zip_ref.extractall(temp_extract_root)

                        capture_files = _collect_capture_files(temp_extract_root)
                        saved_files = []
                        for src_path in capture_files:
                            dst_path = _build_capture_save_path(resolved_upload_target, src_path)
                            shutil.copy2(src_path, dst_path)
                            saved_files.append(dst_path)

                        if os.path.exists(temp_zip_path):
                            os.remove(temp_zip_path)
                        if os.path.exists(temp_extract_root):
                            shutil.rmtree(temp_extract_root)

                        if saved_files:
                            st.success(f"文件夹上传并解压完成，共导入 {len(saved_files)} 个 pcap/pcapng 文件。")
                            with st.expander("查看已导入文件", expanded=False):
                                st.code("\n".join(saved_files), language="text")
                        else:
                            st.warning("zip 中未检测到 pcap/pcapng 文件。")

                else:
                    resolved_source_folder = _normalize_path(source_folder)
                    if not os.path.isdir(resolved_source_folder):
                        st.error(f"本地目录不存在: {resolved_source_folder}")
                    else:
                        copied_files = _copy_folder_contents(resolved_source_folder, resolved_upload_target)
                        if not copied_files:
                            st.warning("指定目录为空，没有可复制文件。")
                        else:
                            st.success(f"目录复制完成，共复制 {len(copied_files)} 个文件。")
                            with st.expander("查看已复制文件", expanded=False):
                                st.code("\n".join(copied_files), language="text")

        elif mode == "第三步：数据预处理阶段一——原始流量 -> 任务数据集":
            # 阶段一：从原始流量抽取任务数据，生成 train/test。
            st.markdown("##### 阶段一参数")
            c1, c2 = st.columns(2)
            with c1:
                input_dir = st.text_input("输入数据目录 --input", value="./uploaded/USTC-TFC2016", key="s1_input")
                dataset_name = st.text_input("数据集名称 --dataset_name", value="ustc-tfc-2016", key="s1_dataset")
                traffic_task = st.selectbox(
                    "任务类型 --traffic_task",
                    ["detection", "generation", "understanding"],
                    index=0,
                    key="s1_task",
                )
            with c2:
                granularity = st.selectbox("粒度 --granularity", ["packet"], index=0, key="s1_granularity")
                output_path = st.text_input("输出目录 --output_path", value="./datasets/USTC-TFC2016", key="s1_output_path")
                output_name = st.text_input("输出名前缀 --output_name", value="USTC-TFC2016", key="s1_output_name")

            preview_lines = st.selectbox(
                "预览前 n 行（执行成功后展示 train/test 文件）",
                [1, 2, 3, 4, 5],
                index=2,
                key="s1_preview_lines",
            )

            command_desc = "执行阶段一预处理并生成 train/test 数据。"
            command = (
                "python preprocess/preprocess_stage1.py "
                f"--input {shlex.quote(input_dir)} "
                f"--dataset_name {shlex.quote(dataset_name)} "
                f"--traffic_task {shlex.quote(traffic_task)} "
                f"--granularity {shlex.quote(granularity)} "
                f"--output_path {shlex.quote(output_path)} "
                f"--output_name {shlex.quote(output_name)}"
            )

        elif mode == "第四步：数据预处理阶段二——任务数据集 -> GLM4格式":
            # 阶段二：将任务数据转为 GLM4 对话消息格式。
            st.markdown("##### 阶段二参数")
            st.info("阶段二默认扫描项目根目录下的 datasets 子目录。")
            stage2_preview_lines = st.selectbox(
                "预览前 n 行（执行成功后展示 datasets 下 GLM4*.jsonl 文件）",
                [1, 2, 3, 4, 5],
                index=2,
                key="s2_preview_lines",
            )
            command_desc = "执行阶段二转换，生成 GLM4 消息格式数据。"
            command = "python preprocess/preprocess_stage2.py"

        st.markdown("##### 命令预览")
        st.code(command, language="bash")
        st.caption(command_desc)

        if mode != "第二步：上传数据集":
            col_run, col_clear = st.columns([1, 1])
            with col_run:
                run_cmd = st.button("开始执行", type="primary", key="run_preprocess_cmd")
            with col_clear:
                clear_output = st.button("清空日志", key="clear_preprocess_log")

            if clear_output:
                st.session_state.pop("preprocess_last_result", None)

            if run_cmd:
                if not os.path.isdir(workdir):
                    st.error(f"工作目录不存在: {workdir}")
                elif mode == "第一步：检查 tshark 字段" and update_config and not update_confirm:
                    st.error("请先完成二次确认后再执行配置更新。")
                else:
                    with st.spinner("命令执行中，请稍候..."):
                        input_text = "Y\n" if mode == "第一步：检查 tshark 字段" and update_config else None
                        result = _run_shell_command(command, cwd=workdir, input_text=input_text)
                    stage1_preview = None
                    if mode == "第三步：数据预处理阶段一——原始流量 -> 任务数据集":
                        stage1_preview = {
                            "output_path": output_path,
                            "output_name": output_name,
                            "head_lines": int(preview_lines),
                        }
                    stage2_preview = None
                    if mode == "第四步：数据预处理阶段二——任务数据集 -> GLM4格式":
                        stage2_preview = {
                            "datasets_root": os.path.join(workdir, "datasets"),
                            "head_lines": int(stage2_preview_lines),
                        }
                    st.session_state["preprocess_last_result"] = {
                        "cmd": command,
                        "cwd": workdir,
                        "mode": mode,
                        "stage1_preview": stage1_preview,
                        "stage2_preview": stage2_preview,
                        "returncode": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }

            result = st.session_state.get("preprocess_last_result")
            if result:
                result_mode = result.get("mode", mode)
                st.markdown("##### 执行结果")
                st.write(f"执行目录: `{result['cwd']}`")
                st.write(f"命令: `{result['cmd']}`")
                if result["returncode"] == 0:
                    st.success("执行成功 (return code = 0)")
                else:
                    st.error(f"执行失败 (return code = {result['returncode']})")

                display_stdout = result["stdout"]
                display_stderr = result["stderr"]
                if result_mode == "第三步：数据预处理阶段一——原始流量 -> 任务数据集" and result["returncode"] == 0:
                    if not display_stdout.strip() and display_stderr.strip():
                        display_stdout = display_stderr
                        display_stderr = ""
                if result_mode == "第四步：数据预处理阶段二——任务数据集 -> GLM4格式" and result["returncode"] == 0:
                    if display_stderr.strip():
                        display_stdout = f"{display_stdout.rstrip()}\n{display_stderr.lstrip()}" if display_stdout.strip() else display_stderr
                        display_stderr = ""
                if result_mode == "第一步：检查 tshark 字段" and result["returncode"] != 0:
                    if not display_stderr.strip() and display_stdout.strip():
                        display_stderr = display_stdout
                        display_stdout = ""

                out_tab, err_tab = st.tabs(["标准输出 stdout", "错误输出 stderr"])
                with out_tab:
                    st.code(display_stdout if display_stdout.strip() else "(无输出)", language="text")
                with err_tab:
                    st.code(display_stderr if display_stderr.strip() else "(无输出)", language="text")

                if result_mode == "第三步：数据预处理阶段一——原始流量 -> 任务数据集" and result["returncode"] == 0:
                    preview_cfg = result.get("stage1_preview") or {}
                    preview_root = _normalize_path(preview_cfg.get("output_path", ""))
                    head_lines = int(preview_cfg.get("head_lines", 3))
                    head_lines = min(max(head_lines, 1), 5)
                    preview_files = _collect_stage1_preview_files(preview_root)

                    st.markdown("##### 生成文件内容预览")
                    st.caption(f"目录: `{preview_root}`，每个文件展示前 {head_lines} 行。")

                    if not preview_files:
                        st.info("未在输出目录中找到包含 train/test 的文本文件（jsonl/json/txt/csv）。")
                    else:
                        for preview_file in preview_files[:10]:
                            relative_path = os.path.relpath(preview_file, preview_root)
                            st.markdown(f"**{relative_path}**")
                            try:
                                preview_text = _read_file_head_lines(preview_file, head_lines)
                            except OSError as exc:
                                st.error(f"读取失败: {preview_file} ({exc})")
                                continue
                            st.code(preview_text if preview_text else "(空文件)", language="text")

                        if len(preview_files) > 10:
                            st.caption(f"共匹配 {len(preview_files)} 个文件，仅展示前 10 个。")

                if result_mode == "第四步：数据预处理阶段二——任务数据集 -> GLM4格式" and result["returncode"] == 0:
                    preview_cfg = result.get("stage2_preview") or {}
                    datasets_root = _normalize_path(preview_cfg.get("datasets_root", "datasets"))
                    head_lines = int(preview_cfg.get("head_lines", 3))
                    head_lines = min(max(head_lines, 1), 5)
                    preview_files = _collect_stage2_preview_files(datasets_root)

                    st.markdown("##### 生成文件内容预览")
                    st.caption(f"目录: `{datasets_root}`，匹配规则: GLM4*.jsonl，每个文件展示前 {head_lines} 行。")

                    if not preview_files:
                        st.info("未在 datasets 目录下找到 GLM4 开头的 jsonl 文件。")
                    else:
                        for preview_file in preview_files[:10]:
                            relative_path = os.path.relpath(preview_file, datasets_root)
                            st.markdown(f"**{relative_path}**")
                            try:
                                preview_text = _read_file_head_lines(preview_file, head_lines)
                            except OSError as exc:
                                st.error(f"读取失败: {preview_file} ({exc})")
                                continue
                            st.code(preview_text if preview_text else "(空文件)", language="text")

                        if len(preview_files) > 10:
                            st.caption(f"共匹配 {len(preview_files)} 个文件，仅展示前 10 个。")


def _apply_ui_theme(theme_name):
    if theme_name == "海洋":
        css = """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 8% 12%, #d8efff 0%, transparent 32%),
                radial-gradient(circle at 92% 88%, #ffe9cd 0%, transparent 34%),
                linear-gradient(120deg, #f6fbff 0%, #fffaf1 100%);
        }
        .block-container {
            max-width: 1180px;
            margin: 0 auto;
            padding-top: 3.25rem;
            padding-right: 1.1rem;
            padding-bottom: 1.6rem;
            padding-left: 1.1rem;
        }
        .hero-card {
            background: linear-gradient(135deg, #145f8f 0%, #2f7fb8 55%, #49a39d 100%);
            color: #ffffff;
            border-radius: 16px;
            padding: 18px 22px;
            box-shadow: 0 10px 30px rgba(20, 95, 143, 0.24);
            margin-top: 0.35rem;
            margin-bottom: 14px;
        }
        .hero-title {
            font-size: 1.8rem;
            font-weight: 800;
            line-height: 1.2;
            margin: 0;
        }
        .hero-subtitle {
            margin-top: 8px;
            font-size: 0.98rem;
            opacity: 0.95;
        }
        div[data-testid="stTextArea"] textarea {
            border-radius: 12px;
            border: 1px solid #bfd8e8;
            background: #f9fcff;
        }
        div[data-testid="stChatMessage"] {
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid rgba(105, 152, 180, 0.22);
            box-shadow: 0 8px 20px rgba(17, 77, 110, 0.08);
        }
        div.stButton > button {
            border-radius: 999px;
            border: 1px solid rgba(20, 95, 143, 0.20);
            background: rgba(255, 255, 255, 0.80);
            color: #184e71;
            font-weight: 700;
            padding: 0.45rem 1.2rem;
        }
        div.stButton > button[kind="primary"] {
            border: none;
            background: linear-gradient(90deg, #145f8f 0%, #2f7fb8 100%);
            color: #ffffff;
            box-shadow: 0 8px 18px rgba(20, 95, 143, 0.18);
        }
        </style>
        """
    else:
        css = """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f8f9fb 0%, #f3f5f8 100%);
        }
        .block-container {
            max-width: 1180px;
            margin: 0 auto;
            padding-top: 3.25rem;
            padding-right: 1.1rem;
            padding-bottom: 1.6rem;
            padding-left: 1.1rem;
        }
        .hero-card {
            background: linear-gradient(135deg, #343a40 0%, #495057 100%);
            color: #ffffff;
            border-radius: 14px;
            padding: 16px 20px;
            margin-top: 0.35rem;
            margin-bottom: 14px;
        }
        .hero-title {
            font-size: 1.7rem;
            font-weight: 800;
            margin: 0;
        }
        .hero-subtitle {
            margin-top: 6px;
            font-size: 0.95rem;
            opacity: 0.92;
        }
        div[data-testid="stTextArea"] textarea {
            border-radius: 10px;
            border: 1px solid #d4d8dd;
            background: #ffffff;
        }
        div[data-testid="stChatMessage"] {
            border-radius: 10px;
            border: 1px solid #e5e8ec;
            background: #ffffff;
        }
        div.stButton > button {
            border-radius: 999px;
            border: 1px solid #c9d1d8;
            background: #ffffff;
            color: #34424f;
            font-weight: 700;
            padding: 0.45rem 1.2rem;
        }
        div.stButton > button[kind="primary"] {
            border: none;
            background: #2f6aa3;
            color: #ffffff;
            box-shadow: 0 8px 18px rgba(47, 106, 163, 0.18);
        }
        </style>
        """

    st.markdown(css, unsafe_allow_html=True)


theme_name = st.sidebar.selectbox("界面主题", ["海洋", "经典"], index=0)
_apply_ui_theme(theme_name)

st.markdown(
    """
    <div class="hero-card">
        <p class="hero-title">TrafficLLM AI 系统</p>
        <p class="hero-subtitle">数据预处理、APEFT 训练、模型评估、测试数据生成与双阶段推理统一入口</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "active_module" not in st.session_state:
    st.session_state.active_module = "overview"

module_labels = [
    ("模块总览", "overview"),
    ("数据预处理", "preprocess"),
    ("APEFT 训练", "apeft"),
    ("模型评估", "evaluation"),
    ("测试数据生成", "generation"),
    ("双阶段推理", "inference"),
]

_render_module_navigation(module_labels)

if st.session_state.active_module == "overview":
    _render_module_overview()
elif st.session_state.active_module == "preprocess":
    _render_preprocess_module()
elif st.session_state.active_module == "apeft":
    _render_apeft_module()
elif st.session_state.active_module == "evaluation":
    _render_evaluation_module()
elif st.session_state.active_module == "generation":
    _render_generate_test_data_module()
elif st.session_state.active_module == "inference":
    _render_inference_module()
