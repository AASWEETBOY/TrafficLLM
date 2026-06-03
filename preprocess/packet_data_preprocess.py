"""
数据包级别数据预处理模块

该模块负责从 PCAP 文件中提取数据包级别的特征数据。
支持多种特征提取方式：五元组、数据包字节、Scapy 格式、tshark 字段等。
使用 subprocess 替代 os.system 以提高安全性和可靠性。
"""

from typing import List, Optional, Dict
import logging
import binascii
import subprocess
import tempfile
from pathlib import Path

try:
    from flowcontainer.extractor import extract
except ImportError as e:
    logging.error(f"无法导入 flowcontainer: {e}")
    raise

try:
    import scapy.all as scapy
    from scapy.all import load_layer
except ImportError as e:
    logging.error(f"无法导入 scapy: {e}")
    raise

from config import packet_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_packet_data(
    pcap_file: str, 
    packet_feature: str = "traffic words",
    max_length: Optional[int] = None
) -> List[str]:
    """
    从 PCAP 文件中提取数据包特征，支持多种提取模式。
    
    参数：
        pcap_file: PCAP 文件的路径
        packet_feature: 数据包特征提取模式，支持以下选项：
            - "generation 5tuple": 提取五元组信息（源IP、目的IP、协议、源端口、目的端口）
            - "generation data": 提取五元组 + 十六进制数据包内容
            - "packet bytes": 仅提取十六进制数据包字节
            - "packet words": 提取 Scapy 格式的完整数据包信息
            - "traffic words" (默认): 使用 tshark 提取详细的流量字段信息
        max_length: 数据包的最大长度（None 时使用配置文件默认值）
    
    返回值：
        包含提取特征的列表，每个元素代表一个数据包的特征字符串
    
    异常：
        FileNotFoundError: 当 PCAP 文件不存在时
        ValueError: 当 packet_feature 参数无效时
        RuntimeError: 当 tshark 命令不可用或执行失败时
        Exception: 当读取或处理文件时出错
    
    注意：
        - "traffic words" 模式需要系统安装 Wireshark/tshark
        - 使用临时文件进行 tshark 输出，会自动清理
    
    示例：
        >>> # 提取五元组信息
        >>> data = build_packet_data("traffic.pcap", "generation 5tuple")
        
        >>> # 提取详细流量字段（推荐用于 LLM 分析）
        >>> data = build_packet_data("traffic.pcap", "traffic words")
    """
    # 验证文件存在性
    pcap_path = Path(pcap_file)
    if not pcap_path.exists():
        raise FileNotFoundError(f"PCAP 文件不存在: {pcap_file}")
    
    if not pcap_path.is_file():
        raise ValueError(f"路径不是文件: {pcap_file}")
    
    # 使用配置文件的默认值
    max_length = max_length or packet_config.MAX_PACKET_LENGTH
    start_index = packet_config.HEX_PACKET_START_INDEX
    
    logger.info(f"开始处理 PCAP 文件: {pcap_file}, 特征类型: {packet_feature}")
    
    build_data = []

    try:
        if packet_feature == "generation 5tuple":
            # 提取五元组信息
            build_data = _extract_5tuple(pcap_file, max_length, start_index)
            
        elif packet_feature == "generation data":
            # 提取五元组 + 数据包内容
            build_data = _extract_5tuple_with_data(pcap_file, max_length, start_index)
            
        elif packet_feature == "packet bytes":
            # 提取数据包字节
            build_data = _extract_packet_bytes(pcap_file, max_length, start_index)
            
        elif packet_feature == "packet words":
            # 提取 Scapy 格式的数据包
            build_data = _extract_packet_words(pcap_file)
            
        elif packet_feature == "traffic words":
            # 使用 tshark 提取详细字段
            build_data = _extract_traffic_words(pcap_file)
            
        else:
            raise ValueError(
                f"不支持的特征类型: {packet_feature}. "
                f"支持的类型: 'generation 5tuple', 'generation data', 'packet bytes', "
                f"'packet words', 'traffic words'"
            )
        
        logger.info(f"成功提取 {len(build_data)} 个数据包特征")
        
    except Exception as e:
        logger.error(f"处理 PCAP 文件时出错: {e}")
        raise

    return build_data


def _extract_5tuple(pcap_file: str, max_length: int, start_index: int) -> List[str]:
    """提取五元组信息。"""
    try:
        packets = scapy.rdpcap(pcap_file)
    except Exception as e:
        raise IOError(f"无法读取 PCAP 文件: {e}")

    build_data = []
    for packet in packets:
        try:
            tuple_dict = _get_5tuple_from_packet(packet)
            if tuple_dict:
                packet_string = str(tuple_dict)
                truncated = packet_string[start_index:min(len(packet_string), max_length)]
                build_data.append(truncated)
        except Exception as e:
            logger.warning(f"提取五元组时出错: {e}")
            continue

    return build_data


def _extract_5tuple_with_data(
    pcap_file: str, 
    max_length: int, 
    start_index: int
) -> List[str]:
    """提取五元组 + 数据包内容。"""
    try:
        packets = scapy.rdpcap(pcap_file)
    except Exception as e:
        raise IOError(f"无法读取 PCAP 文件: {e}")

    build_data = []
    for packet in packets:
        try:
            tuple_dict = _get_5tuple_from_packet(packet)
            if tuple_dict:
                # 获取数据包的十六进制表示
                packet_data = packet.copy()
                data = binascii.hexlify(bytes(packet_data))
                hex_string = data.decode()
                
                # 组合五元组和数据包内容
                packet_string = f"{tuple_dict} {hex_string}"
                truncated = packet_string[start_index:min(len(packet_string), max_length)]
                build_data.append(truncated)
        except Exception as e:
            logger.warning(f"提取五元组和数据时出错: {e}")
            continue

    return build_data


def _extract_packet_bytes(
    pcap_file: str, 
    max_length: int, 
    start_index: int
) -> List[str]:
    """提取数据包字节。"""
    try:
        packets = scapy.rdpcap(pcap_file)
    except Exception as e:
        raise IOError(f"无法读取 PCAP 文件: {e}")

    build_data = []
    for packet in packets:
        try:
            packet_data = packet.copy()
            data = binascii.hexlify(bytes(packet_data))
            packet_string = data.decode()
            truncated = packet_string[start_index:min(len(packet_string), max_length)]
            build_data.append(truncated)
        except Exception as e:
            logger.warning(f"提取数据包字节时出错: {e}")
            continue

    return build_data


def _extract_packet_words(pcap_file: str) -> List[str]:
    """提取 Scapy 格式的数据包。"""
    try:
        packets = scapy.rdpcap(pcap_file)
    except Exception as e:
        raise IOError(f"无法读取 PCAP 文件: {e}")

    build_data = []
    for packet in packets:
        try:
            packet_data = str(packet.show)[29:-1].replace("\\\\", "\\")
            build_data.append(packet_data)
        except Exception as e:
            logger.warning(f"提取数据包 show 信息时出错: {e}")
            continue

    return build_data


def _extract_traffic_words(pcap_file: str) -> List[str]:
    """使用 tshark 提取详细的流量字段信息。"""
    # 检查 tshark 是否可用
    if not _check_tshark_available():
        raise RuntimeError(
            "tshark 命令不可用。请安装 Wireshark 或确保 tshark 在系统 PATH 中。"
        )
    
    fields = packet_config.TSHARK_FIELDS
    
    # 使用临时文件
    with tempfile.NamedTemporaryFile(
        mode='w+', 
        delete=False, 
        suffix='.txt',
        encoding='utf-8'
    ) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # 构建 tshark 命令
        extract_str = " -e " + " -e ".join(fields)
        cmd = [
            "tshark",
            "-r", pcap_file,
            *extract_str.split(),
            "-T", "fields",
            "-Y", "tcp or udp"
        ]
        
        logger.debug(f"执行 tshark 命令: {' '.join(cmd)}")
        
        # 执行 tshark 命令
        with open(tmp_path, 'w', encoding='utf-8') as output_file:
            result = subprocess.run(
                cmd,
                stdout=output_file,
                stderr=subprocess.PIPE,
                timeout=packet_config.TSHARK_TIMEOUT,
                check=True,
                text=True
            )
        
        # 读取结果
        with open(tmp_path, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
        
        build_data = []
        for line in lines:
            try:
                packet_data = _parse_tshark_line(line, fields)
                if packet_data:
                    build_data.append(packet_data)
            except Exception as e:
                logger.warning(f"解析 tshark 输出行时出错: {e}")
                continue
        
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"tshark 命令超时（超过 {packet_config.TSHARK_TIMEOUT} 秒）")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"tshark 命令执行失败: {e.stderr}")
    except Exception as e:
        raise RuntimeError(f"使用 tshark 提取数据时出错: {e}")
    finally:
        # 清理临时文件
        try:
            Path(tmp_path).unlink()
        except Exception as e:
            logger.warning(f"清理临时文件失败: {e}")
    
    return build_data


def _get_5tuple_from_packet(packet) -> Optional[Dict[str, any]]:
    """从数据包中提取五元组信息。"""
    tuple_dict = {}
    
    try:
        if packet.haslayer("IP"):
            tuple_dict["src"] = packet["IP"].src
            tuple_dict["dst"] = packet["IP"].dst
            tuple_dict["proto"] = packet["IP"].proto
        else:
            return None
        
        if packet.haslayer("TCP"):
            tuple_dict["sport"] = packet["TCP"].sport
            tuple_dict["dport"] = packet["TCP"].dport
        elif packet.haslayer("UDP"):
            tuple_dict["sport"] = packet["UDP"].sport
            tuple_dict["dport"] = packet["UDP"].dport
        else:
            return None
            
        return tuple_dict
        
    except Exception as e:
        logger.warning(f"提取五元组时出错: {e}")
        return None


def _parse_tshark_line(line: str, fields: List[str]) -> Optional[str]:
    """解析 tshark 输出的一行数据。"""
    values = line.strip().split("\t")
    
    # 修改：容忍字段数量不匹配，使用min来避免索引越界
    if not values:
        return None
    
    # 调整：如果字段数不匹配，记录警告但仍然尝试解析
    if len(values) != len(fields):
        # 只在字段数量相差很大时才跳过（超过10%差异）
        if abs(len(values) - len(fields)) > len(fields) * 0.1:
            logger.debug(f"字段数量差异过大: 期望{len(fields)}个，实际{len(values)}个")
            return None
    
    # 使用实际可用的字段数量
    available_count = min(len(values), len(fields))
    
    # 如果第一个字段为空，跳过此行
    if not values[0]:
        return None
    
    packet_data = f"{fields[0]}: {values[0]}"
    
    # 使用zip自动处理长度不一致的情况
    for field, value in zip(fields[1:available_count], values[1:available_count]):
        if not value:  # 跳过空值
            continue
            
        # 特殊处理某些字段
        if field == "tcp.flags.str":
            try:
                value = value.encode("unicode_escape").decode("unicode_escape")
            except Exception:
                pass
        elif field == "tcp.payload":
            # 限制载荷长度
            value = value[:1000] if len(value) > 1000 else value
        
        packet_data += f", {field}: {value}"
    
    return packet_data


def _check_tshark_available() -> bool:
    """检查 tshark 命令是否可用。"""
    try:
        result = subprocess.run(
            ["tshark", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


