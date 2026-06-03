import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent


def load_jsonl(path: Path) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_json_lines(path: Path) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def extract_dataset_name(peft_relative_path: str) -> str:
    # e.g. "ustc-tfc-2016_detection_packet/checkpoint-11000/" -> "ustc-tfc-2016"
    first_part = peft_relative_path.strip("/").split("/")[0]
    suffix = "_detection_packet"
    if not first_part.endswith(suffix):
        raise ValueError(f"Unexpected peft_set path format: {peft_relative_path}")
    return first_part[: -len(suffix)]


def build_task_dataset_paths(config: dict, with_label: bool = True) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for task_name, task_key in config["tasks"].items():
        peft_path = config["peft_set"].get(task_key)
        if not peft_path:
            raise KeyError(f"Task key {task_key} missing in peft_set")

        dataset_name = extract_dataset_name(peft_path)
        if with_label:
            file_name = f"GLM4_{dataset_name}_detection_packet_test_with_label.jsonl"
        else:
            file_name = f"GLM4_{dataset_name}_detection_packet_test.jsonl"

        dataset_path = ROOT / "datasets" / dataset_name / file_name
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        result[task_key] = dataset_path
    return result


def build_task_label_paths(config: dict) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for task_name, task_key in config["tasks"].items():
        peft_path = config["peft_set"].get(task_key)
        if not peft_path:
            raise KeyError(f"Task key {task_key} missing in peft_set")

        dataset_name = extract_dataset_name(peft_path)
        label_file = f"{dataset_name}_label.json"
        label_path = ROOT / "datasets" / dataset_name / label_file
        if not label_path.exists():
            raise FileNotFoundError(f"Label file not found: {label_path}")

        result[task_key] = label_path
    return result


def sanitize_task_name(task_name: str) -> str:
    return task_name.lower().replace(" ", "_").replace("/", "_")


def sample_for_task(
    rng: random.Random,
    task_name: str,
    task_key: str,
    traffic_records: List[dict],
    count: int,
) -> List[dict]:
    if not traffic_records:
        raise ValueError(f"No traffic records found for task: {task_name}")

    samples = []
    for i in range(count):
        traffic_item = rng.choice(traffic_records)

        user_content = ""
        expected_answer = ""
        for message in traffic_item.get("messages", []):
            role = message.get("role")
            content = message.get("content", "")
            if role == "user":
                user_content = content
            elif role == "assistant":
                expected_answer = content

        if not user_content:
            raise ValueError("Invalid traffic record: missing user content")

        samples.append(
            {
                "sample_id": f"{task_key}_{i + 1:04d}",
                "task_name": task_name,
                "task_key": task_key,
                "user_input": {
                    "traffic_data": user_content,
                },
                "expected_output": {
                    "stage1_task_name": task_name,
                    "stage1_task_key": task_key,
                    "stage2_answer": expected_answer,
                },
            }
        )
    return samples


def sample_for_task_by_labels(
    rng: random.Random,
    task_name: str,
    task_key: str,
    traffic_records: List[dict],
    label_names: List[str],
    samples_per_label: int,
) -> List[dict]:
    if not traffic_records:
        raise ValueError(f"No traffic records found for task: {task_name}")

    records_by_label: Dict[str, List[dict]] = {label: [] for label in label_names}
    for record in traffic_records:
        answer = ""
        for message in record.get("messages", []):
            if message.get("role") == "assistant":
                answer = message.get("content", "")
                break
        if answer in records_by_label:
            records_by_label[answer].append(record)

    samples: List[dict] = []
    idx = 1
    for label_name in label_names:
        candidates = records_by_label.get(label_name, [])
        if not candidates:
            raise ValueError(
                f"No traffic records found for label '{label_name}' in task '{task_name}'"
            )

        if len(candidates) >= samples_per_label:
            selected = rng.sample(candidates, samples_per_label)
        else:
            # If records are fewer than required, sample with replacement to fulfill quota.
            selected = [rng.choice(candidates) for _ in range(samples_per_label)]

        for traffic_item in selected:
            user_content = ""
            expected_answer = ""
            for message in traffic_item.get("messages", []):
                role = message.get("role")
                content = message.get("content", "")
                if role == "user":
                    user_content = content
                elif role == "assistant":
                    expected_answer = content

            if not user_content:
                raise ValueError("Invalid traffic record: missing user content")

            samples.append(
                {
                    "sample_id": f"{task_key}_{idx:04d}",
                    "task_name": task_name,
                    "task_key": task_key,
                    "user_input": {
                        "traffic_data": user_content,
                    },
                    "expected_output": {
                        "stage1_task_name": task_name,
                        "stage1_task_key": task_key,
                        "stage2_answer": expected_answer,
                    },
                }
            )
            idx += 1

    return samples


def generate_samples_by_task(config: dict, samples_per_label: int, seed: int, with_label: bool) -> Dict[str, List[dict]]:
    rng = random.Random(seed)

    task_dataset_paths = build_task_dataset_paths(config, with_label=with_label)
    task_label_paths = build_task_label_paths(config)

    task_samples_map: Dict[str, List[dict]] = {}
    for task_name, task_key in config["tasks"].items():
        dataset_records = load_jsonl(task_dataset_paths[task_key])
        with task_label_paths[task_key].open("r", encoding="utf-8") as f:
            label_map = json.load(f)
        label_names = list(label_map.keys())

        task_samples = sample_for_task_by_labels(
            rng=rng,
            task_name=task_name,
            task_key=task_key,
            traffic_records=dataset_records,
            label_names=label_names,
            samples_per_label=samples_per_label,
        )
        task_samples_map[task_key] = task_samples

    return task_samples_map


def save_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate random TrafficLLM two-stage test samples per task from configured tasks and datasets."
    )
    parser.add_argument(
        "--samples-per-label",
        type=int,
        default=3,
        help="Random sample count for each label in each configured task.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated_test_data",
        help="Output directory path (relative to project root), one file per task.",
    )
    parser.add_argument(
        "--no-label",
        action="store_true",
        help="Use *_test.jsonl instead of *_test_with_label.jsonl (expected stage2 answers may be placeholders).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.samples_per_label <= 0:
        raise ValueError("--samples-per-label must be > 0")

    config_path = ROOT / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    task_samples_map = generate_samples_by_task(
        config=config,
        samples_per_label=args.samples_per_label,
        seed=args.seed,
        with_label=(not args.no_label),
    )

    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files: List[Path] = []
    for task_name, task_key in config["tasks"].items():
        task_samples = task_samples_map[task_key]
        file_name = f"{task_key}_{sanitize_task_name(task_name)}_test_data.json"
        output_path = output_dir / file_name

        payload = {
            "meta": {
                "seed": args.seed,
                "samples_per_label": args.samples_per_label,
                "task_name": task_name,
                "task_key": task_key,
                "total_samples": len(task_samples),
                "traffic_source": "*_test_with_label.jsonl" if not args.no_label else "*_test.jsonl",
            },
            "samples": task_samples,
        }
        save_json(output_path, payload)
        generated_files.append(output_path)

    print(f"Generated {len(generated_files)} task files in {output_dir}")
    for path in generated_files:
        print(path)


if __name__ == "__main__":
    main()