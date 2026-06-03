"""
特定数据集的专用预处理工具

该模块包含针对特定数据集结构的预处理函数，例如 USTC-TFC2016 数据集。
"""

from typing import Tuple, List, Dict, Any
import logging
from pathlib import Path
from tqdm import tqdm

from preprocess_utils import build_td_text_dataset, write_labels, build_dataset, save_dataset

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ustc_tfc2016_preprocess(args, detection_task: str = "EMD") -> None:
    """
    USTC-TFC2016 数据集的专用预处理函数。
    
    针对 USTC-TFC2016 数据集的特殊结构处理，分别处理 Benign 和 Malware
    两个大类，每个类别下包含多个应用类型的子文件夹。
    
    参数：
        args: 命令行参数对象，包含：
            - input: 输入数据集根目录
            - output_path: 输出目录
            - output_name: 输出文件名前缀
            - granularity: 处理粒度 ('flow' 或 'packet')
        detection_task: 检测任务类型，默认为 "EMD" (Encrypted Malware Detection)
    
    返回值：
        无，直接将处理后的数据集保存到指定输出路径
    
    异常：
        FileNotFoundError: 当输入目录不存在时
        ValueError: 当目录结构不符合预期时
    
    输出文件：
        - {output_name}_{detection_task}_{granularity}_train.json
        - {output_name}_{detection_task}_{granularity}_test.json
        - {output_name}_label.json (包含所有类别标签的映射)
    
    数据集结构：
        input/
            Benign/
                AppType1/
                    *.pcap
                AppType2/
                    *.pcap
            Malware/
                MalwareType1/
                    *.pcap
                MalwareType2/
                    *.pcap
    """
    logger.info(f"开始处理 USTC-TFC2016 数据集，任务类型: {detection_task}")
    
    # 验证输入路径
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {args.input}")
    
    benign_class_path = input_path / "Benign"
    malware_class_path = input_path / "Malware"
    
    if not benign_class_path.exists():
        raise FileNotFoundError(f"Benign 目录不存在: {benign_class_path}")
    
    if not malware_class_path.exists():
        raise FileNotFoundError(f"Malware 目录不存在: {malware_class_path}")

    train_dataset = []
    test_dataset = []
    labels = []

    # 处理 Benign 类别
    logger.info("处理 Benign 类别...")
    benign_files = [f.name for f in benign_class_path.iterdir() if f.is_dir()]
    labels.extend(benign_files)

    for file in tqdm(benign_files, desc="Benign"):
        try:
            class_train_data, class_test_data = build_dataset(args, str(benign_class_path), file)

            train_text_data = build_td_text_dataset(
                class_train_data, 
                first_label="benign", 
                second_label=file, 
                task_name=detection_task, 
                granularity=args.granularity
            )
            test_text_data = build_td_text_dataset(
                class_test_data, 
                first_label="benign", 
                second_label=file, 
                task_name=detection_task, 
                granularity=args.granularity
            )

            train_dataset.extend(train_text_data)
            test_dataset.extend(test_text_data)
            
        except Exception as e:
            logger.warning(f"处理 Benign/{file} 时出错: {e}")
            continue

    # 处理 Malware 类别
    logger.info("处理 Malware 类别...")
    malware_files = [f.name for f in malware_class_path.iterdir() if f.is_dir()]
    labels.extend(malware_files)

    for file in tqdm(malware_files, desc="Malware"):
        try:
            class_train_data, class_test_data = build_dataset(args, str(malware_class_path), file)

            train_text_data = build_td_text_dataset(
                class_train_data, 
                first_label="malware", 
                second_label=file, 
                task_name=detection_task, 
                granularity=args.granularity
            )
            test_text_data = build_td_text_dataset(
                class_test_data, 
                first_label="malware", 
                second_label=file, 
                task_name=detection_task, 
                granularity=args.granularity
            )

            train_dataset.extend(train_text_data)
            test_dataset.extend(test_text_data)
            
        except Exception as e:
            logger.warning(f"处理 Malware/{file} 时出错: {e}")
            continue

    # 保存数据集和标签
    save_dataset(args, train_dataset, test_dataset)
    
    label_path = Path(args.output_path) / f"{args.output_name}_label.json"
    write_labels(labels, str(label_path))
    
    logger.info(f"USTC-TFC2016 数据集处理完成")
    logger.info(f"训练样本: {len(train_dataset)}, 测试样本: {len(test_dataset)}, 类别数: {len(labels)}")
