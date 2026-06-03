"""
数据集预处理第一阶段的主程序

该模块提供流量数据集的预处理入口，支持流量检测、生成、理解等多种任务。
支持多种数据集：USTC-TFC2016、ISCX-Botnet、ISCX-VPN-2016 等。

使用方法：
    python preprocess_stage1.py --input <数据集路径> --dataset_name <数据集名称> \\
        --traffic_task <任务类型> --granularity <粒度> \\
        --output_path <输出路径> --output_name <输出名称>
"""

import logging
import argparse
import random
import sys
from pathlib import Path

from tqdm import tqdm

from specfic_dataset_utils import ustc_tfc2016_preprocess
from preprocess_utils import (
    build_td_text_dataset,
    build_tg_text_dataset,
    build_tu_text_dataset,
    write_labels,
    build_dataset,
    save_dataset
)
from config import dataset_config, task_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_args():
    """
    解析并验证命令行参数。
    
    必需参数：
        --input: 原始数据集路径
        --dataset_name: 数据集名称
        --traffic_task: 流量任务类型 (detection, generation, understanding)
        --granularity: 处理粒度 (flow, packet)
        --output_path: 输出数据集路径
        --output_name: 输出数据集名称
    
    返回值：
        包含所有命令行参数的对象
    
    示例：
        python preprocess_stage1.py --input ./raw_data --dataset_name ustc-tfc-2016 \\
            --traffic_task detection --granularity flow --output_path ./output \\
            --output_name ustc_tfc
    """
    parser = argparse.ArgumentParser(
        description="流量数据集预处理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python preprocess_stage1.py \\
      --input ../datasets/ustc-tfc-2016 \\
      --dataset_name ustc-tfc-2016 \\
      --traffic_task detection \\
      --granularity packet \\
      --output_path ./datasets \\
      --output_name ustc_tfc
        """
    )
    
    parser.add_argument(
        "--input", 
        type=str, 
        required=True,
        help="原始数据集路径"
    )
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        required=True,
        help="数据集名称（如: ustc-tfc-2016, iscx-botnet 等）"
    )
    parser.add_argument(
        "--traffic_task", 
        type=str, 
        required=True,
        choices=task_config.TRAFFIC_TASKS,
        help=f"流量任务类型，可选: {', '.join(task_config.TRAFFIC_TASKS)}"
    )
    parser.add_argument(
        "--granularity", 
        type=str, 
        required=True,
        choices=task_config.GRANULARITIES,
        help=f"处理粒度，可选: {', '.join(task_config.GRANULARITIES)}"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True,
        help="输出数据集路径"
    )
    parser.add_argument(
        "--output_name", 
        type=str, 
        required=True,
        help="输出数据集名称"
    )

    args = parser.parse_args()
    
    # 验证输入路径
    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"输入路径不存在: {args.input}")
    
    # 创建输出目录
    output_path = Path(args.output_path)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"输出目录: {output_path}")
    except Exception as e:
        parser.error(f"无法创建输出目录: {e}")
    
    return args


def traffic_detection_preprocess(args, detection_task: str) -> None:
    """
    流量检测任务的数据集预处理。
    
    按分类标签遍历输入目录，为每个类别构建训练集和测试集，
    生成带有指令和输出的文本数据集用于 LLM 训练。
    
    参数：
        args: 命令行参数对象
        detection_task: 检测任务类型
    
    返回值：
        无，直接将处理后的数据集保存到指定路径
    
    输出文件：
        - {output_name}_{traffic_task}_{granularity}_train.json
        - {output_name}_{traffic_task}_{granularity}_test.json
        - {output_name}_label.json
    """
    logger.info(f"开始流量检测任务预处理，任务类型: {detection_task}")
    
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {args.input}")
    
    train_dataset = []
    test_dataset = []
    labels = []

    # 获取所有类别文件夹
    category_dirs = [f for f in input_path.iterdir() if f.is_dir()]
    
    if not category_dirs:
        raise ValueError(f"输入目录中没有找到类别文件夹: {input_path}")
    
    labels.extend([f.name for f in category_dirs])
    logger.info(f"找到 {len(labels)} 个类别: {labels}")

    for category_dir in tqdm(category_dirs, desc="处理类别"):
        try:
            train_data, test_data = build_dataset(args, str(input_path), category_dir.name)

            train_text_data = build_td_text_dataset(
                train_data, 
                second_label=category_dir.name, 
                task_name=detection_task, 
                granularity=args.granularity
            )
            test_text_data = build_td_text_dataset(
                test_data, 
                second_label=category_dir.name, 
                task_name=detection_task, 
                granularity=args.granularity
            )

            train_dataset.extend(train_text_data)
            test_dataset.extend(test_text_data)
            
        except Exception as e:
            logger.warning(f"处理类别 {category_dir.name} 时出错: {e}")
            continue

    # 保存数据集
    save_dataset(args, train_dataset, test_dataset)

    # 保存标签
    label_path = Path(args.output_path) / f"{args.output_name}_label.json"
    write_labels(labels, str(label_path))
    
    logger.info(
        f"流量检测任务预处理完成: "
        f"训练样本 {len(train_dataset)}, 测试样本 {len(test_dataset)}"
    )


def traffic_generation_preprocess(args) -> None:
    """
    流量生成任务的数据集预处理。
    
    按分类标签遍历输入目录，为每个类别构建训练集和测试集，
    生成用于生成特定类型流量的文本数据集。
    
    参数：
        args: 命令行参数对象
    
    返回值：
        无，直接将处理后的数据集保存到指定路径
    
    输出文件：
        - {output_name}_generation_{granularity}_train.json
        - {output_name}_generation_{granularity}_test.json
    """
    logger.info("开始流量生成任务预处理")
    
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {args.input}")

    train_dataset = []
    test_dataset = []

    # 获取所有类别文件夹
    category_dirs = [f for f in input_path.iterdir() if f.is_dir()]
    
    if not category_dirs:
        raise ValueError(f"输入目录中没有找到类别文件夹: {input_path}")
    
    labels = [f.name for f in category_dirs]
    logger.info(f"找到 {len(labels)} 个类别: {labels}")

    for category_dir in tqdm(category_dirs, desc="处理类别"):
        try:
            train_data, test_data = build_dataset(args, str(input_path), category_dir.name)

            train_text_data = build_tg_text_dataset(
                train_data, 
                traffic_label=category_dir.name, 
                granularity=args.granularity
            )
            test_text_data = build_tg_text_dataset(
                test_data, 
                traffic_label=category_dir.name, 
                granularity=args.granularity
            )

            train_dataset.extend(train_text_data)
            test_dataset.extend(test_text_data)
            
        except Exception as e:
            logger.warning(f"处理类别 {category_dir.name} 时出错: {e}")
            continue

    # 保存数据集
    save_dataset(args, train_dataset, test_dataset)
    
    logger.info(
        f"流量生成任务预处理完成: "
        f"训练样本 {len(train_dataset)}, 测试样本 {len(test_dataset)}"
    )


def traffic_understanding_preprocess(args) -> None:
    """
    流量理解任务的数据集预处理。
    
    解析 TCP 流量数据包，基于不同的协议字段生成问答对，
    用于教导 LLM 理解流量数据包结构和字段含义。
    注：此函数强制使用 packet 粒度处理。
    
    参数：
        args: 命令行参数对象
    
    返回值：
        无，直接将处理后的数据集保存到指定路径
    
    输出文件：
        - {output_name}_understanding_packet_train.json (最多 20000 样本)
        - {output_name}_understanding_packet_test.json (最多 200 样本)
    """
    logger.info("开始流量理解任务预处理")
    
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {args.input}")
    
    train_dataset = []
    test_dataset = []

    # 强制使用 packet 粒度
    original_granularity = args.granularity
    args.granularity = "packet"
    logger.info("流量理解任务强制使用 packet 粒度")

    # 获取所有类别文件夹
    category_dirs = [f for f in input_path.iterdir() if f.is_dir()]
    
    if not category_dirs:
        raise ValueError(f"输入目录中没有找到类别文件夹: {input_path}")
    
    labels = [f.name for f in category_dirs]
    logger.info(f"找到 {len(labels)} 个类别: {labels}")

    for category_dir in tqdm(category_dirs, desc="处理类别"):
        try:
            train_data, test_data = build_dataset(args, str(input_path), category_dir.name)

            train_text_data = build_tu_text_dataset(train_data, fields=["TCP"])
            test_text_data = build_tu_text_dataset(test_data, fields=["TCP"])

            train_dataset.extend(train_text_data)
            test_dataset.extend(test_text_data)
            
        except Exception as e:
            logger.warning(f"处理类别 {category_dir.name} 时出错: {e}")
            continue

    # 打乱并限制样本数量
    random.shuffle(train_dataset)
    random.shuffle(test_dataset)
    
    max_train = dataset_config.TU_MAX_TRAIN_SAMPLES
    max_test = dataset_config.TU_MAX_TEST_SAMPLES
    
    train_dataset = train_dataset[:max_train]
    test_dataset = test_dataset[:max_test]
    
    logger.info(
        f"样本限制: 训练集 {len(train_dataset)}/{max_train}, "
        f"测试集 {len(test_dataset)}/{max_test}"
    )

    # 保存数据集
    save_dataset(args, train_dataset, test_dataset)
    
    # 恢复原始 granularity
    args.granularity = original_granularity
    
    logger.info(
        f"流量理解任务预处理完成: "
        f"训练样本 {len(train_dataset)}, 测试样本 {len(test_dataset)}"
    )


def main():
    """
    主入口函数，根据任务类型调用相应的预处理函数。
    
    根据 traffic_task 参数和 dataset_name 选择对应的预处理流程：
    - detection: 流量检测任务
    - generation: 流量生成任务
    - understanding: 流量理解任务
    
    支持的数据集：
        detection: ustc-tfc-2016, iscx-botnet, iscx-vpn-2016, lfett-2021, 
                  dohbrw-2020, iscx-tor-2016, dapt-2020 等
        generation/understanding: 任何结构化的流量数据集
    
    返回值：
        无，处理结果保存到指定的输出路径
    """
    try:
        args = get_args()
        
        logger.info("="*60)
        logger.info("流量数据集预处理程序")
        logger.info("="*60)
        logger.info(f"数据集名称: {args.dataset_name}")
        logger.info(f"任务类型: {args.traffic_task}")
        logger.info(f"粒度: {args.granularity}")
        logger.info(f"输入路径: {args.input}")
        logger.info(f"输出路径: {args.output_path}")
        logger.info(f"输出名称: {args.output_name}")
        logger.info("="*60)
        
        traffic_task = args.traffic_task

        if traffic_task == "detection":
            # 从配置获取检测任务类型
            detection_task = task_config.DETECTION_TASKS.get(
                args.dataset_name, 
                "EAC"  # 默认为加密应用分类
            )
            
            logger.info(f"检测任务类型: {detection_task}")
            
            # 特殊处理 USTC-TFC2016 数据集
            if args.dataset_name == "ustc-tfc-2016":
                ustc_tfc2016_preprocess(args, detection_task=detection_task)
            else:
                traffic_detection_preprocess(args, detection_task=detection_task)

        elif traffic_task == "generation":
            traffic_generation_preprocess(args)

        elif traffic_task == "understanding":
            traffic_understanding_preprocess(args)
        
        else:
            raise ValueError(f"不支持的任务类型: {traffic_task}")
        
        logger.info("="*60)
        logger.info("预处理完成！")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"预处理过程中发生错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
