"""
数据集预处理工具模块

该模块提供数据集的分割、文本数据集构建、标签写入等功能。
支持流量检测、流量生成、流量理解等多种任务类型。
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import random
import json
import os
from pathlib import Path

from flow_data_preprocess import build_flow_data
from packet_data_preprocess import build_packet_data
from config import dataset_config, task_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置随机种子以确保可重复性
random.seed(dataset_config.RANDOM_SEED)


def split_dataset(
    build_data: List[Any], 
    sampling: bool = True,
    max_samples: Optional[int] = None,
    train_ratio: Optional[float] = None
) -> Tuple[List[Any], List[Any]]:
    """
    按指定比例分割数据集为训练集和测试集。
    
    参数：
        build_data: 原始数据列表
        sampling: 是否使用采样，默认为 True
        max_samples: 最大采样数量（None 时使用配置文件默认值）
        train_ratio: 训练集比例（None 时使用配置文件默认值）
    
    返回值：
        (train_data, test_data) 训练集和测试集的元组
    
    异常：
        ValueError: 当 train_ratio 不在 (0, 1) 范围内时
    
    示例：
        >>> train, test = split_dataset(data, sampling=True)
        >>> train, test = split_dataset(data, sampling=False, train_ratio=0.8)
    """
    if not build_data:
        logger.warning("输入数据为空")
        return [], []
    
    # 使用配置文件的默认值
    max_samples = max_samples or dataset_config.MAX_SAMPLING_NUMBER
    train_ratio = train_ratio or dataset_config.TRAINING_SAMPLE_RATIO
    
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio 必须在 (0, 1) 范围内，当前值: {train_ratio}")
    
    # 打乱数据
    data_copy = build_data.copy()
    random.shuffle(data_copy)
    
    # 计算训练集和测试集大小
    if sampling:
        total_samples = min(max_samples, len(data_copy))
    else:
        total_samples = len(data_copy)
    
    train_nb = int(total_samples * train_ratio)
    test_nb = total_samples - train_nb
    
    train_data = data_copy[:train_nb]
    test_data = data_copy[train_nb:train_nb + test_nb]
    
    logger.info(f"数据集分割完成: 训练集 {len(train_data)} 样本, 测试集 {len(test_data)} 样本")

    return train_data, test_data


def write_dataset(dataset: List[Dict[str, Any]], output_path: str) -> None:
    """
    序列化数据集为 JSONL 文件。
    
    每一行是一个完整的 JSON 对象，便于流式读取和处理大数据集。
    
    参数：
        dataset: 包含数据样本的列表，每个元素为字典
        output_path: 输出文件路径
    
    返回值：
        无，直接写入文件
    
    异常：
        IOError: 当文件写入失败时
    
    文件格式：
        JSONL (JSON Lines) 格式，每一行是一个 JSON 对象
    """
    if not dataset:
        logger.warning(f"数据集为空，跳过写入: {output_path}")
        return
    
    # 创建输出目录（如果不存在）
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 打乱数据集
    dataset_copy = dataset.copy()
    random.shuffle(dataset_copy)
    
    try:
        with open(output_path, "w", encoding="utf-8") as fin:
            for data in dataset_copy:
                json.dump(data, fin, ensure_ascii=False)
                fin.write("\n")
        
        logger.info(f"成功写入 {len(dataset_copy)} 条数据到: {output_path}")
        
    except Exception as e:
        raise IOError(f"写入数据集文件失败: {e}")


def write_labels(labels: List[str], output_path: str) -> None:
    """
    将类别标签序列编码为索引映射，保存为 JSON 文件。
    
    参数：
        labels: 类别标签列表
        output_path: 输出文件路径 (JSON 格式)
    
    返回值：
        无，直接写入文件
    
    异常：
        IOError: 当文件写入失败时
    
    输出格式：
        JSON 对象，键为类别标签，值为递增的整数
        例子: {"YouTube": 0, "Facebook": 1, ...}
    """
    if not labels:
        logger.warning(f"标签列表为空，跳过写入: {output_path}")
        return
    
    # 创建输出目录（如果不存在）
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建标签到索引的映射
    label_dict = {label: i for i, label in enumerate(labels)}
    
    try:
        with open(output_path, "w", encoding="utf-8") as fin:
            json.dump(label_dict, fin, indent=4, separators=(',', ': '), ensure_ascii=False)
        
        logger.info(f"成功写入 {len(label_dict)} 个标签到: {output_path}")
        
    except Exception as e:
        raise IOError(f"写入标签文件失败: {e}")


def build_dataset(
    args, 
    path: str, 
    file: str
) -> Tuple[List[str], List[str]]:
    """
    载入所有 PCAP 文件并提取数据包特征，然后分割训练集及测试集。
    
    根据 granularity 参数选择流级或数据包级特征提取。
    
    参数：
        args: 参数对象，包含 granularity ('flow' 或 'packet')
        path: 父目录路径
        file: 类别文件夹名
    
    返回值：
        (train_data, test_data) 训练集及测试集的元组
    
    异常：
        FileNotFoundError: 当路径不存在时
        ValueError: 当 granularity 参数无效时
    """
    files_path = Path(path) / file
    
    if not files_path.exists():
        raise FileNotFoundError(f"路径不存在: {files_path}")
    
    if not files_path.is_dir():
        raise ValueError(f"路径不是目录: {files_path}")
    
    build_data = []
    pcaps = list(files_path.glob("*.pcap")) + list(files_path.glob("*.pcapng"))
    
    if not pcaps:
        logger.warning(f"未找到 PCAP 文件: {files_path}")
        return [], []
    
    logger.info(f"处理 {len(pcaps)} 个 PCAP 文件 (类别: {file})")
    
    for pcap in pcaps:
        try:
            if args.granularity == "flow":
                pcap_data = build_flow_data(str(pcap))
            elif args.granularity == "packet":
                pcap_data = build_packet_data(str(pcap))
            else:
                raise ValueError(
                    f"不支持的粒度: {args.granularity}. "
                    f"支持的值: 'flow', 'packet'"
                )
            
            build_data.extend(pcap_data)
            
        except Exception as e:
            logger.warning(f"处理 PCAP 文件失败 {pcap}: {e}")
            continue

    train_data, test_data = split_dataset(build_data)
    return train_data, test_data


def save_dataset(
    args, 
    train_dataset: List[Dict[str, Any]], 
    test_dataset: List[Dict[str, Any]]
) -> None:
    """
    保存训练集及测试集为 JSONL 文件。
    
    参数：
        args: 命令行参数对象。必须包含：
            - output_path: 输出目录
            - output_name: 输出文件名前缀
            - traffic_task: 任务类型
            - granularity: 粒度
        train_dataset: 训练集
        test_dataset: 测试集
    
    返回值：
        无，直接写入文件
    
    输出文件名: 
        - {output_name}_{traffic_task}_{granularity}_train.json
        - {output_name}_{traffic_task}_{granularity}_test.json
    """
    # 构建文件名
    base_name = f"{args.output_name}_{args.traffic_task}_{args.granularity}"
    
    train_path = os.path.join(args.output_path, f"{base_name}_train.json")
    test_path = os.path.join(args.output_path, f"{base_name}_test.json")
    
    # 写入文件
    write_dataset(train_dataset, train_path)
    write_dataset(test_dataset, test_path)


def build_td_text_dataset(
    traffic_data: List[str],
    first_label: Optional[str] = None,
    second_label: Optional[str] = None,
    task_name: Optional[str] = None,
    granularity: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    流量检测任务的文本数据集构建。
    
    根据不同的检测任务类型生成对应的指令和输出，用于 LLM 监督学习。
    
    任务类型说明：
        - EMD: 加密恶意软件检测 (Encrypted Malware Detection)
        - EAC: 加密应用分类 (Encrypted App Classification)
        - BND: 僵尸网络检测 (Botnet Detection)
        - EVD: 加密 VPN 检测 (Encrypted VPN Detection)
        - MDD: 恶意 DoH 检测 (Malicious DoH Detection)
        - TBD: Tor 网络行为检测 (Tor Behavior Detection)
        - APT: APT 攻击检测 (APT Detection)
    
    参数：
        traffic_data: 流量特征数据列表
        first_label: 第一级标签，如 'benign', 'malware'（保留，兼容旧代码）
        second_label: 第二级标签或类别名称
        task_name: 任务类型标识符
        granularity: 粒度 ('flow' 或 'packet')
    
    返回值：
        描述-标签数据序列，每个元素是一个字典：
        {"instruction": "操作前缀描述", "output": "类别标签"}
    
    异常：
        ValueError: 当 task_name 不支持时
    """
    if not traffic_data:
        logger.warning("流量数据为空，返回空数据集")
        return []
    
    if not task_name:
        raise ValueError("task_name 不能为空")
    
    if not second_label:
        raise ValueError("second_label 不能为空")
    
    # 从配置文件获取指令模板
    instruction_template = task_config.TASK_INSTRUCTIONS.get(task_name)
    
    if instruction_template is None:
        raise ValueError(
            f"不支持的任务类型: {task_name}. "
            f"支持的类型: {list(task_config.TASK_INSTRUCTIONS.keys())}"
        )
    
    # 格式化指令（替换 granularity）
    instruction = instruction_template.format(granularity=granularity)
    
    # 输出为第二级标签
    output = second_label
    
    # 特殊处理 MDD 任务（使用不同的输出格式）
    if task_name == "MDD":
        output = f"The traffic category is likely to be recognized as {second_label}."
    
    # 构建数据集
    dataset = []
    for data in traffic_data:
        dataset.append({
            "instruction": f"{instruction}\n<{granularity}>: {data}",
            "output": output
        })
    
    logger.info(f"构建了 {len(dataset)} 个 {task_name} 任务样本")
    
    return dataset


def build_tg_text_dataset(
    traffic_data: List[str],
    traffic_label: str,
    granularity: str
) -> List[Dict[str, str]]:
    """
    流量生成任务的文本数据集构建。
    
    指令类型为"请生成 [traffic_label] 流量的 [granularity]"，
    输出为实际的流量特征数据。
    
    参数：
        traffic_data: 流量特征数据列表
        traffic_label: 流量类别名称，如 'YouTube', 'Facebook'
        granularity: 粒度 ('flow' 或 'packet')
    
    返回值：
        描述-输出数据序列，每个元素是一个字典：
        {"instruction": "请生成 ... ", "output": "流量特征数据"}
    """
    if not traffic_data:
        logger.warning("流量数据为空，返回空数据集")
        return []
    
    instruction = f"Please generate a {granularity} of {traffic_label} traffic."

    dataset = [
        {
            "instruction": instruction,
            "output": data
        }
        for data in traffic_data
    ]
    
    logger.info(f"构建了 {len(dataset)} 个流量生成任务样本")

    return dataset


def build_tu_text_dataset(
    traffic_data: List[str],
    fields: Optional[List[str]] = None
) -> List[Dict[str, str]]:
    """
    流量理解任务的文本数据集构建。
    
    根据指定的协议字段 (IP, TCP, UDP, DNS 等) 生成描述-答案对，
    每个样本提问数据包中的某个字段含义。
    
    支持的字段：
        - IP: IP 协议字段
        - TCP: TCP 协议字段
        - UDP: UDP 协议字段
        - DNS: DNS 协议字段
        - TLS: TLS 协议字段
        - http.HTTPRequest: HTTP 请求字段
        - http.HTTPResponse: HTTP 回复字段
        - GeoIP: 地理位置信息
        - JA3: TLS 指纹信息
    
    参数：
        traffic_data: 流量特征数据列表
        fields: 协议字段列表，默认为 ["TCP"]
    
    返回值：
        问答数据序列，每个元素是一个字典：
        {"instruction": "数据包中协议字段信息查询", "output": "<scapy-FIELD>"}
    """
    if not traffic_data:
        logger.warning("流量数据为空，返回空数据集")
        return []
    
    if fields is None:
        fields = ["TCP"]

    knowledge_fields = []
    api_calls = []

    if "IP" in fields:
        knowledge_fields.extend(
            ["IP Version", "IP Header Length", "Differentiated Services Field", "Total Length",
             "Identification", "IP Flags", "Fragment Offset", "Time to Live", "Protocol", "IP Header Checksum",
             "Source Address", "Destination Address"]
        )
        api_calls.extend(
            ["scapy-IP-version", "scapy-IP-ihl", "scapy-IP-tos", "scapy-IP-len", "scapy-IP-id", "scapy-IP-flags",
             "scapy-IP-frag", "scapy-IP-ttl", "scapy-IP-proto", "scapy-IP-chksum", "scapy-IP-src", "scapy-IP-dst"]
        )

    if "TCP" in fields:
        knowledge_fields.extend(
            ["Source Port", "Destination Port", "Sequence Number", "Acknowledge Number",
             "TCP Flags", "Window", "TCP Header Checksum", "Urgent Pointer", "Destination Address"]
        )
        api_calls.extend(
            ["scapy-TCP-sport", "scapy-TCP-dport", "scapy-TCP-seq", "scapy-TCP-ack", "scapy-TCP-flags",
             "scapy-TCP-window", "scapy-TCP-chksum", "scapy-TCP-urgptr", "scapy-TCP-options"]
        )

    if "UDP" in fields:
        knowledge_fields.extend(
            ["Source Port", "Destination Port", "UDP Length", "UDP Header Checksum"]
        )
        api_calls.extend(
            ["scapy-UDP-sport", "scapy-UDP-dport", "scapy-UDP-len", "scapy-UDP-chksum"]
        )

    if "TLS" in fields:
        knowledge_fields.extend(
            ["Content Type", "Record Version", "TLS Message", "Message Type", "Handshake Version",
             "Cipher Suites", "Extensions"]
        )
        api_calls.extend(
            ["scapy-TLS-type", "scapy-TLS-version", "scapy-TLS-msg", "scapy-TLS-msg-msgtype", "scapy-TLS-msg-version",
             "scapy-TLS-msg-ciphers", "scapy-TLS-msg-ext"]
        )

    if "DNS" in fields:
        knowledge_fields.extend(
            ["Transaction ID", "Response", "Opcode", "Authoritative", "Truncated", "Recursion Desired",
             "Recursion Available", "Z", "Answer Authenticated", "Non-Authenticated", "Questions", "Answer RRs",
             "Authority RRs", "Additional RRs", "Queries", "Answers"]
        )
        api_calls.extend(
            ["scapy-DNS-id", "scapy-DNS-qr", "scapy-DNS-opcode", "scapy-DNS-aa", "scapy-DNS-tc", "scapy-DNS-rd",
             "scapy-DNS-ra", "scapy-DNS-z", "scapy-DNS-ad", "scapy-DNS-cd", "scapy-DNS-qdcount", "scapy-DNS-ancount",
             "scapy-DNS-nscount", "scapy-DNS-arcount", "scapy-DNS-qd", "scapy-DNS-an"]
        )

    if "http.HTTPRequest" in fields:
        knowledge_fields.extend(
            ["Headers", "Host", "User-Agent", "Accept", "Connection", "Method", "Path", "Http-Version", "Range",
             "Accept-Language", "Additional-Headers"]
        )
        api_calls.extend(
            ["scapy-http.HTTPRequest-Headers", "scapy-http.HTTPRequest-Host", "scapy-http.HTTPRequest-User-Agent",
             "scapy-http.HTTPRequest-Accept", "scapy-http.HTTPRequest-Connection", "scapy-http.HTTPRequest-Method",
             "scapy-http.HTTPRequest-Path", "scapy-http.HTTPRequest-Http-Version", "scapy-http.HTTPRequest-Range",
             "scapy-http.HTTPRequest-Accept-Language", "scapy-http.HTTPRequest-Additional-Headers"]
        )

    if "http.HTTPResponse" in fields:
        knowledge_fields.extend(
            ["Headers", 'Accept-Ranges', 'Server', 'Cache-Control', 'Connection', 'Date', 'Content-Length',
             'Content-Range', 'Content-Type', 'Last-Modified', 'Additional-Headers', 'Status-Line']
        )
        api_calls.extend(
            ["scapy-http.HTTPResponse-Headers", "scapy-http.HTTPResponse-Accept-Ranges",
             "scapy-http.HTTPResponse-Server", "scapy-http.HTTPResponse-Cache-Control",
             "scapy-http.HTTPResponse-Connection", "scapy-http.HTTPResponse-Date",
             "scapy-http.HTTPResponse-Content-Length", "scapy-http.HTTPResponse-Content-Range",
             "scapy-http.HTTPResponse-Content-Type", "scapy-http.HTTPResponse-Last-Modified",
             "scapy-http.HTTPResponse-Additional-Headers", "scapy-http.HTTPResponse-Status-Line"]
        )

    if "GeoIP" in fields:
        knowledge_fields.extend(
            ["source address", "destination address"]
        )
        api_calls.extend(
            ["<geoip-src>", "<geoip-dst>"]
        )

    if "JA3" in fields:
        knowledge_fields.extend(
            ["client fingerprints", "server fingerprints"]
        )
        api_calls.extend(
            ["<ja3-client>", "<ja3-server>"]
        )

    dataset = []
    
    if not knowledge_fields:
        logger.warning("没有可用的字段，返回空数据集")
        return []

    for data in traffic_data:
        index = random.randint(0, len(knowledge_fields) - 1)
        
        if "GeoIP" in fields or "JA3" in fields:
            instruction = f"Please analyze the {knowledge_fields[index]} in the packet: {data}"
            output = f"<{api_calls[index]}>"
        else:
            instruction = f"What is {knowledge_fields[index]} in the packet: {data}"
            output = f"<{api_calls[index]}>"
        
        dataset.append({
            "instruction": instruction,
            "output": output
        })
    
    logger.info(f"构建了 {len(dataset)} 个流量理解任务样本")

    return dataset
