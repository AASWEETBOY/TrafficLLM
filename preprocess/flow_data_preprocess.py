"""
流级别数据预处理模块

该模块负责从 PCAP 文件中提取流级别的特征数据。
支持多种特征提取方式：流字节、流序列、载荷字节等。
"""

from typing import List, Optional
import logging
import binascii
from pathlib import Path

try:
    from flowcontainer.extractor import extract
except ImportError as e:
    logging.error(f"无法导入 flowcontainer: {e}")
    raise

try:
    import scapy.all as scapy
except ImportError as e:
    logging.error(f"无法导入 scapy: {e}")
    raise

from config import flow_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_flow_data(
    pcap_file: str, 
    flow_feature: str = "flow bytes",
    max_packets: Optional[int] = None,
    max_length: Optional[int] = None
) -> List[str]:
    """
    从 PCAP 文件构建网络流数据特征。
    
    支持三种特征提取方式：
    1. "flow bytes": 提取每个流的原始字节数据（十六进制表示）
    2. "flow sequence": 提取每个数据包的长度序列
    3. "payload bytes": 提取 TCP/UDP 载荷的字节数据（十六进制表示）
    
    参数：
        pcap_file: PCAP 文件的路径
        flow_feature: 特征提取方式，支持 "flow bytes", "flow sequence", "payload bytes"
        max_packets: 每个流最多处理的数据包数量（None 时使用配置文件默认值）
        max_length: 每个数据包的最大长度（None 时使用配置文件默认值）
    
    返回值：
        包含提取的流特征字符串列表。
        - "flow bytes": 格式为 "<pck>十六进制数据<pck>十六进制数据..."
        - "flow sequence": 格式为 "长度1 长度2 长度3..."
        - "payload bytes": 格式为 "<pck>十六进制数据<pck>十六进制数据..."
    
    异常：
        FileNotFoundError: 当 PCAP 文件不存在时
        ValueError: 当 flow_feature 参数无效时
        Exception: 当读取或处理文件时出错
    
    示例：
        >>> data = build_flow_data("traffic.pcap", "flow bytes")
        >>> print(len(data))  # 流数量
    """
    # 验证文件存在性
    pcap_path = Path(pcap_file)
    if not pcap_path.exists():
        raise FileNotFoundError(f"PCAP 文件不存在: {pcap_file}")
    
    if not pcap_path.is_file():
        raise ValueError(f"路径不是文件: {pcap_file}")
    
    # 使用配置文件的默认值
    max_packets = max_packets or flow_config.MAX_PACKET_NUMBER
    max_length = max_length or flow_config.MAX_PACKET_LENGTH_IN_FLOW
    start_index = flow_config.HEX_PACKET_START_INDEX
    
    logger.info(f"开始处理 PCAP 文件: {pcap_file}, 特征类型: {flow_feature}")
    
    build_data = []

    try:
        if flow_feature == "flow bytes":
            # 提取流字节特征
            build_data = _extract_flow_bytes(pcap_file, max_packets, max_length, start_index)
            
        elif flow_feature == "flow sequence":
            # 提取流序列特征
            build_data = _extract_flow_sequence(pcap_file, max_packets)
            
        elif flow_feature == "payload bytes":
            # 提取载荷字节特征
            build_data = _extract_payload_bytes(pcap_file, max_packets, max_length, start_index)
            
        else:
            raise ValueError(
                f"不支持的特征类型: {flow_feature}. "
                f"支持的类型: 'flow bytes', 'flow sequence', 'payload bytes'"
            )
        
        logger.info(f"成功提取 {len(build_data)} 个流特征")
        
    except Exception as e:
        logger.error(f"处理 PCAP 文件时出错: {e}")
        raise

    return build_data


def _extract_flow_bytes(
    pcap_file: str, 
    max_packets: int, 
    max_length: int, 
    start_index: int
) -> List[str]:
    """
    提取流的原始字节数据。
    
    参数：
        pcap_file: PCAP 文件路径
        max_packets: 最大数据包数量
        max_length: 最大数据包长度
        start_index: 起始索引
    
    返回值：
        包含流字节特征的列表
    """
    try:
        packets = scapy.rdpcap(pcap_file)
    except Exception as e:
        raise IOError(f"无法读取 PCAP 文件: {e}")

    hex_stream = []
    for i, packet in enumerate(packets):
        if i >= max_packets:
            break
            
        try:
            packet_data = packet.copy()
            data = binascii.hexlify(bytes(packet_data))
            packet_string = data.decode()
            
            # 截取指定范围的数据
            truncated = packet_string[start_index:min(len(packet_string), max_length)]
            hex_stream.append(truncated)
            
        except Exception as e:
            logger.warning(f"处理第 {i} 个数据包时出错: {e}")
            continue

    flow_data = "<pck>" + "<pck>".join(hex_stream)
    return [flow_data]


def _extract_flow_sequence(pcap_file: str, max_packets: int) -> List[str]:
    """
    提取流的数据包长度序列。
    
    参数：
        pcap_file: PCAP 文件路径
        max_packets: 最大数据包数量
    
    返回值：
        包含流序列特征的列表
    """
    try:
        flows = extract(
            pcap_file,
            filter='tcp or udp',
            extension=["tcp.payload", "udp.payload"],
            split_flag=False,
            verbose=False
        )
    except Exception as e:
        raise IOError(f"无法提取流数据: {e}")

    build_data = []
    for key, flow in flows.items():
        flow_seq = []
        length_seq = flow.lengths
        
        for i, packet_length in enumerate(length_seq):
            if i >= max_packets:
                break
            flow_seq.append(str(packet_length))

        flow_data = " ".join(flow_seq)
        build_data.append(flow_data)
    
    return build_data


def _extract_payload_bytes(
    pcap_file: str, 
    max_packets: int, 
    max_length: int, 
    start_index: int
) -> List[str]:
    """
    提取 TCP/UDP 载荷的字节数据。
    
    参数：
        pcap_file: PCAP 文件路径
        max_packets: 最大数据包数量
        max_length: 最大数据包长度
        start_index: 起始索引
    
    返回值：
        包含载荷字节特征的列表
    """
    try:
        flows = extract(
            pcap_file,
            filter='tcp or udp',
            extension=["tcp.payload", "udp.payload"],
            split_flag=False,
            verbose=False
        )
    except Exception as e:
        raise IOError(f"无法提取流数据: {e}")

    build_data = []
    for key, flow in flows.items():
        if len(flow.extension.values()) == 0:
            continue
            
        packet_list = list(flow.extension.values())[0]
        hex_stream = []
        
        for i, packet in enumerate(packet_list):
            if i >= max_packets:
                break
                
            try:
                payload_data = packet[0][:min(len(packet[0]), max_length)]
                hex_stream.append(payload_data)
            except Exception as e:
                logger.warning(f"处理载荷数据时出错: {e}")
                continue
        
        flow_data = "<pck>" + "<pck>".join(hex_stream)
        build_data.append(flow_data)
    
    return build_data
