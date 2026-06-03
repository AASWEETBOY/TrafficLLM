"""
预处理模块的配置文件

此文件包含所有预处理相关的常量和配置项，便于集中管理和修改。
"""

from dataclasses import dataclass
from typing import List


@dataclass
class FlowConfig:
    """流级别数据处理配置"""
    MAX_PACKET_NUMBER: int = 10  # 每个流最多处理的数据包数量
    MAX_PACKET_LENGTH_IN_FLOW: int = 256  # 每个数据包的最大长度（字节）
    HEX_PACKET_START_INDEX: int = 0  # 十六进制数据包的起始索引


@dataclass
class PacketConfig:
    """数据包级别数据处理配置"""
    MAX_PACKET_LENGTH: int = 1024  # 单个数据包的最大长度（字节）
    HEX_PACKET_START_INDEX: int = 0  # 十六进制数据包的起始索引
    TSHARK_TIMEOUT: int = 300  # tshark 命令超时时间（秒）
    
    # tshark 提取的字段列表
    TSHARK_FIELDS: List[str] = None
    
    def __post_init__(self):
        """初始化 tshark 字段列表"""
        if self.TSHARK_FIELDS is None:
            self.TSHARK_FIELDS = [
                # FRAME 层字段
                "frame.encap_type", "frame.time", "frame.offset_shift", "frame.time_epoch", "frame.time_delta", "frame.time_relative", "frame.number", "frame.len", "frame.marked", "frame.protocols",

                # ETH 层字段
                "eth.dst", "eth.dst_resolved", "eth.src", "eth.src_resolved", "eth.type",

                # IP 层字段
                "ip.version", "ip.hdr_len", "ip.dsfield", "ip.dsfield.dscp", "ip.dsfield.ecn", "ip.len", "ip.id", "ip.flags", "ip.flags.rb", "ip.flags.df", "ip.flags.mf", "ip.frag_offset", "ip.ttl", "ip.proto", "ip.checksum", "ip.checksum.status", "ip.src", "ip.dst",

                # TCP 层字段
                "tcp.srcport", "tcp.dstport", "tcp.stream", "tcp.len", "tcp.seq", "tcp.nxtseq", "tcp.ack", "tcp.hdr_len", "tcp.flags", "tcp.flags.res", "tcp.flags.cwr", "tcp.flags.urg", "tcp.flags.ack", "tcp.flags.push", "tcp.flags.reset", "tcp.flags.syn", "tcp.flags.fin", "tcp.flags.str", "tcp.window_size", "tcp.window_size_scalefactor", "tcp.checksum", "tcp.checksum.status", "tcp.urgent_pointer", "tcp.time_relative", "tcp.time_delta", "tcp.analysis.bytes_in_flight", "tcp.analysis.push_bytes_sent", "tcp.segment", "tcp.segment.count", "tcp.reassembled.length", "tcp.payload",

                # UDP 层字段
                "udp.srcport", "udp.dstport", "udp.length", "udp.checksum", "udp.checksum.status", "udp.stream",

                # DATA 层字段
                "data.len"
]


@dataclass
class DatasetConfig:
    """数据集处理配置"""
    MAX_SAMPLING_NUMBER: int = 1000  # 每个类别的最大采样数量
    TRAINING_SAMPLE_RATIO: float = 0.90  # 训练集占比（90%训练，10%测试）
    RANDOM_SEED: int = 42  # 随机种子，确保可重复性
    
    # 流量理解任务的最大样本数
    TU_MAX_TRAIN_SAMPLES: int = 20000
    TU_MAX_TEST_SAMPLES: int = 200


@dataclass
class TaskConfig:
    """任务相关配置"""
    
    # 支持的任务类型
    TRAFFIC_TASKS = ["detection", "generation", "understanding"]
    
    # 支持的粒度
    GRANULARITIES = ["flow", "packet"]
    
    # 检测任务类型映射
    DETECTION_TASKS = {
        "ustc-tfc-2016": "EMD",   # Encrypted Malware Detection
        "iscx-botnet": "BND",     # Botnet Detection
        "iscx-vpn-2016": "EVD",   # Encrypted VPN Detection
        "lfett-2021": "EVD",      # Encrypted VPN Detection
        "dohbrw-2020": "MDD",     # Malicious DoH Detection
        "iscx-tor-2016": "TBD",   # Tor Behavior Detection
        "dapt-2020": "APT",       # APT Detection
    }
    
    # 任务指令模板
    TASK_INSTRUCTIONS = {
        "EMD": (
            "Given the following traffic data <{granularity}> that contains protocol fields, "
            "traffic features, and payloads. Please conduct the ENCRYPTED MALWARE DETECTION TASK to determine "
            "which application category the encrypted benign or malicious traffic belongs to. The categories "
            "include 'BitTorrent, FTP, Facetime, Gmail, MySQL, Outlook, SMB, Skype, Weibo, WorldOfWarcraft, "
            "Cridex, Geodo, Htbot, Miuref, Neris, Nsis-ay, Shifu, Tinba, Virut, Zeus'."
        ),
        "EAC": (
            "Given the following traffic data <{granularity}> that contains protocol fields, "
            "traffic features, and payloads. Please conduct the ENCRYPTED APP CLASSIFICATION TASK to determine "
            "which APP category the encrypted traffic belongs to."
        ),
        "BND": (
            "Given the following traffic data <{granularity}> that contains protocol fields, "
            "traffic features, and payloads. Please conduct the BOTNET DETECTION TASK to determine "
            "which type of network the traffic belongs to. The categories "
            "include 'IRC, Neris, RBot, Virut, normal'."
        ),
        "EVD": (
            "Given the following traffic data <{granularity}> that contains protocol fields, "
            "traffic features, and payloads. Please conduct the ENCRYPTED VPN DETECTION TASK to determine "
            "which behavior or application category the VPN encrypted traffic belongs to. The categories "
            "include 'aim, bittorrent, email, facebook, ftps, hangout, icq, netflix, sftp, skype, spotify, "
            "vimeo, voipbuster, youtube'."
        ),
        "MDD": (
            "Below is a traffic {granularity}. Please conduct the malicious DoH detection task: "
        ),
        "TBD": (
            "Given the following traffic data <{granularity}> that contains protocol fields, "
            "traffic features, and payloads. Please conduct the TOR BEHAVIOR DETECTION TASK to determine "
            "which behavior or application category the traffic belongs to under the Tor network. "
            "The categories include 'audio, browsing, chat, file, mail, p2p, video, voip'."
        ),
        "APT": (
            "Given the following traffic data <{granularity}> that contains protocol fields, "
            "traffic features, and payloads. Please conduct the APT DETECTION TASK to determine "
            "which behavior or application category the traffic belongs to under the APT attacks. "
            "The categories include 'APT and normal'."
        ),
    }


# 创建全局配置实例
flow_config = FlowConfig()
packet_config = PacketConfig()
dataset_config = DatasetConfig()
task_config = TaskConfig()
