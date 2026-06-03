# -*- coding: utf-8 -*-
"""将 train/test 的 jsonl 样本转换为 GLM4 的 messages 格式。"""

import json
import os
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from preprocess_utils import write_labels


PACKET_SPLITTER = "\n<packet>:"
LINE_SPLITTER = "\n"
PAYLOAD_SPLITTER = ", tcp.payload:"
PACKET_LEVEL_SYSTEM_PROMPT = (
	"Given the following traffic data <packet>. Please conduct the CLASSIFICATION TASK "
	"to determine which category the traffic belongs to. The categories include "
	"'Malware Traffic Detection, Botnet Detection, Web Attack Detection, "
	"APT Attack Detection, Encrypted VPN Detection'."
)
PACKET_LEVEL_SAMPLE_COUNT = 4000
PACKET_LEVEL_TEST_SAMPLE_COUNT = 500


def write_jsonl(output_file, records):
	"""以 UTF-8 编码将记录写入 jsonl 文件。"""
	with open(output_file, "w", encoding="utf-8") as file_obj:
		for record in records:
			file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_unlabeled_messages(records):
	"""返回 records 的副本，并将 assistant 标签替换为 '-'。"""
	unlabeled = []
	for item in records:
		# 对每个 message 字典做浅拷贝，避免修改带标签原始数据。
		messages = [dict(msg) for msg in item["messages"]]
		messages[2]["content"] = "-"
		unlabeled.append({"messages": messages})
	return unlabeled


def parse_instruction(raw_instruction):
	"""从原始 instruction 文本中提取 system 与 user 内容。"""
	if PACKET_SPLITTER not in raw_instruction and LINE_SPLITTER not in raw_instruction:
		raise ValueError("instruction format is invalid")

	# 期望的源文本格式：
	# <instruction>\n<packet>:<packet_fields>, tcp.payload:<payload>
	if PACKET_SPLITTER in raw_instruction:
		instruction_part, packet_part = raw_instruction.split(PACKET_SPLITTER, 1)
	elif LINE_SPLITTER in raw_instruction:
		# 回退策略：仅按首个换行切分，兼容未包含 <packet>: 前缀的数据。
		instruction_part, packet_part = raw_instruction.split(LINE_SPLITTER, 1)
		if packet_part.startswith("<packet>:"):
			packet_part = packet_part[len("<packet>:") :]
	else:
		raise ValueError("instruction format is invalid")

	# tcp.payload 在部分数据集中可能缺失，此时保留完整 packet 字段。
	if PAYLOAD_SPLITTER in packet_part:
		packet_content = packet_part.split(PAYLOAD_SPLITTER, 1)[0]
	else:
		packet_content = packet_part
	user_content = f"<packet>:{packet_content}."
	return instruction_part, user_content


def transfer_file(root_dir, file_name):
	# 使用 pathlib 拼接路径，避免 Windows 下字符串拼接路径问题。
	input_path = Path(root_dir) / file_name
	data_df = pd.read_json(input_path, lines=True)

	records = []
	skipped_rows = 0
	for _, row in data_df.iterrows():
		try:
			instruction, input_value = parse_instruction(str(row["instruction"]))
		except ValueError:
			# 跳过格式不符合预期的行，但不中断整文件处理。
			skipped_rows += 1
			continue
		output_value = str(row["output"])

		messages = [
			{"role": "system", "content": instruction},
			{"role": "user", "content": input_value},
			{"role": "assistant", "content": output_value},
		]
		records.append({"messages": messages})

	output_name = file_name.replace(".json", ".jsonl")
	output_path = Path(root_dir) / f"GLM4_{output_name}"

	if "test" in output_name:
		# 测试集同时保留带标签文件（评估）和去标签文件（推理）。
		labeled_path = output_path.with_name(output_path.stem + "_with_label.jsonl")
		write_jsonl(labeled_path, records)

		unlabeled_path = Path(str(labeled_path).replace("_with_label", ""))
		write_jsonl(unlabeled_path, build_unlabeled_messages(records))
	else:
		write_jsonl(output_path, records)

	if skipped_rows:
		print(f"[WARN] {file_name}: skipped {skipped_rows} malformed rows")


def read_jsonl_records(file_path):
	"""读取 jsonl 文件并返回字典列表。"""
	records = []
	with open(file_path, "r", encoding="utf-8") as file_obj:
		for line in file_obj:
			line = line.strip()
			if not line:
				continue
			records.append(json.loads(line))
	return records


def build_packet_level_instruction_record(record, task_name):
	"""基于原有 messages 构造统一包级分类任务样本。"""
	messages = [dict(msg) for msg in record.get("messages", [])]
	if len(messages) < 3:
		raise ValueError("invalid messages format: expected at least 3 roles")

	messages[0]["content"] = PACKET_LEVEL_SYSTEM_PROMPT
	messages[2]["content"] = task_name
	return {"messages": messages}


def generate_packet_level_instructions_dataset():
	"""从五个流量数据集构造统一包级 instructions 训练集。"""
	project_root = Path(__file__).resolve().parent.parent
	datasets_dir = project_root / "datasets"
	instructions_dir = datasets_dir / "instructions"
	instructions_dir.mkdir(parents=True, exist_ok=True)

	config_path = project_root / "config.json"
	with open(config_path, "r", encoding="utf-8") as file_obj:
		config_data = json.load(file_obj)

	task_map = config_data.get("tasks", {})
	peft_map = config_data.get("peft_set", {})

	# 从 peft_set 中推导任务缩写与数据集目录的映射关系。
	code_to_dataset = {}
	for code, checkpoint_path in peft_map.items():
		if not checkpoint_path or "_detection_packet" not in checkpoint_path:
			continue
		dataset_name = checkpoint_path.split("_detection_packet", 1)[0]
		code_to_dataset[code] = dataset_name

	task_order = []
	sampled_train_records_by_task = {}
	sampled_test_records_by_task = {}
	for task_name, task_code in task_map.items():
		dataset_name = code_to_dataset.get(task_code)
		if not dataset_name:
			raise ValueError(f"missing dataset mapping for task code: {task_code}")

		train_file = datasets_dir / dataset_name / f"GLM4_{dataset_name}_detection_packet_train.jsonl"
		if not train_file.exists():
			raise FileNotFoundError(f"train file not found: {train_file}")

		test_with_label_file = (
			datasets_dir / dataset_name / f"GLM4_{dataset_name}_detection_packet_test_with_label.jsonl"
		)
		if not test_with_label_file.exists():
			raise FileNotFoundError(f"test file not found: {test_with_label_file}")

		source_records = read_jsonl_records(train_file)
		if len(source_records) < PACKET_LEVEL_SAMPLE_COUNT:
			raise ValueError(
				f"{train_file} has only {len(source_records)} records, "
				f"but {PACKET_LEVEL_SAMPLE_COUNT} are required"
			)

		test_records = read_jsonl_records(test_with_label_file)
		if len(test_records) < PACKET_LEVEL_TEST_SAMPLE_COUNT:
			raise ValueError(
				f"{test_with_label_file} has only {len(test_records)} records, "
				f"but {PACKET_LEVEL_TEST_SAMPLE_COUNT} are required"
			)

		sampled_train_records_by_task[task_name] = random.sample(source_records, PACKET_LEVEL_SAMPLE_COUNT)
		sampled_test_records_by_task[task_name] = random.sample(test_records, PACKET_LEVEL_TEST_SAMPLE_COUNT)
		task_order.append(task_name)

	combined_train_records = []
	for sample_index in range(PACKET_LEVEL_SAMPLE_COUNT):
		for task_name in task_order:
			record = sampled_train_records_by_task[task_name][sample_index]
			combined_train_records.append(build_packet_level_instruction_record(record, task_name))

	combined_test_labeled_records = []
	for sample_index in range(PACKET_LEVEL_TEST_SAMPLE_COUNT):
		for task_name in task_order:
			record = sampled_test_records_by_task[task_name][sample_index]
			combined_test_labeled_records.append(build_packet_level_instruction_record(record, task_name))

	# 在测试集写盘前再随机打乱一次顺序。
	random.shuffle(combined_test_labeled_records)

	train_output_path = instructions_dir / "GLM4_instructions_train.jsonl"
	write_jsonl(train_output_path, combined_train_records)

	test_labeled_output_path = instructions_dir / "GLM4_instructions_test_with_label.jsonl"
	write_jsonl(test_labeled_output_path, combined_test_labeled_records)

	test_output_path = instructions_dir / "GLM4_instructions_test.jsonl"
	write_jsonl(test_output_path, build_unlabeled_messages(combined_test_labeled_records))

	label_path = instructions_dir / "instructions_label.json"
	write_labels(task_order, str(label_path))
	print(
		f"[INFO] generated {len(combined_train_records)} packet-level train samples at {train_output_path}"
	)
	print(
		f"[INFO] generated {len(combined_test_labeled_records)} packet-level test samples at {test_labeled_output_path} and {test_output_path}"
	)


def main():
	# 使用脚本所在目录作为基准，避免受当前工作目录影响。
	src_dataset = Path(__file__).resolve().parent / "../datasets"
	for root, _, files in os.walk(src_dataset):
		for file_name in tqdm(files):
			# 仅处理原始 train/test json，跳过生成文件与其他元数据文件。
			is_target_split = "train" in file_name or "test" in file_name
			is_source_json = file_name.endswith(".json") and not file_name.startswith("GLM4_")
			if is_target_split and is_source_json:
				transfer_file(root, file_name)

	generate_packet_level_instructions_dataset()


if __name__ == "__main__":
	main()