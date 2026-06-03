import json
import os
import re
import sys

import fire
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

try:
    from peft import PeftModelForCausalLM
except ImportError:
    PeftModelForCausalLM = None

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def test_set_to_prompt(test_set):
    test_messages = []
    target_responses = []

    for idx, test_data in enumerate(test_set):
        row = json.loads(test_data)

        # Preferred format for current datasets: {"messages": [...]}.
        if "messages" in row:
            messages = row["messages"]
            prompt_messages = []
            target = None

            for message in messages:
                role = message.get("role", "")
                if role == "assistant":
                    target = str(message.get("content", "")).strip()
                    break
                prompt_messages.append({
                    "role": role,
                    "content": message.get("content", ""),
                })

            if target is None:
                raise ValueError(f"No assistant label found in test sample index={idx}.")
            if not prompt_messages:
                raise ValueError(f"No prompt messages found in test sample index={idx}.")

            test_messages.append(prompt_messages)
            target_responses.append(target)
            continue

        # Backward compatibility with older format: {"instruction": ..., "output": ...}
        if "instruction" in row and "output" in row:
            test_messages.append([{"role": "user", "content": row["instruction"]}])
            target_responses.append(row["output"])
            continue

        raise ValueError(f"Unsupported sample format at index={idx}: keys={list(row.keys())}")

    return test_messages, target_responses


def _normalize_label_text(text):
    if text is None:
        return ""
    text = str(text).strip()
    if text.endswith("。"):
        text = text[:-1]
    return text


def _extract_label(text, label_dict):
    text = _normalize_label_text(text)
    if text in label_dict:
        return text

    if " " in text:
        candidate = text.split(" ")[-1].rstrip("。")
        if candidate in label_dict:
            return candidate

    # Sampling can produce extra explanation text; try to recover a valid label from the full response.
    # Prefer longer labels first to avoid partial matches when one label is a substring of another.
    for label in sorted(label_dict.keys(), key=len, reverse=True):
        if label and label in text:
            return label

    return text


def td_evaluation(predict_responses, target_responses, label_file):
    with open(label_file, "r", encoding="utf-8") as fin:
        label_dict = json.load(fin)

    preds = []
    labels = []
    unknown_label_id = len(label_dict)

    for predict_response, target_response in zip(predict_responses, target_responses):
        pred_text = _extract_label(predict_response, label_dict)
        target_text = _extract_label(target_response, label_dict)

        if target_text not in label_dict:
            raise ValueError(f"Target label not found in label file: {target_text}")

        labels.append(label_dict[target_text])
        if pred_text in label_dict:
            preds.append(label_dict[pred_text])
        else:
            preds.append(unknown_label_id)

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(labels, preds, average="weighted", zero_division=0),
        "f1": f1_score(labels, preds, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "classification_report": classification_report(labels, preds, zero_division=0),
    }
    return metrics


def tg_evaluation(predict_responses, target_responses, test_messages, write_path="generation.json"):
    del target_responses
    dataset = {}
    for predict_response, messages in zip(predict_responses, test_messages):
        user_content = ""
        for message in messages:
            if message.get("role") == "user":
                user_content = str(message.get("content", ""))
                break
        label = user_content.split(" ")[-2] if len(user_content.split(" ")) >= 2 else "unknown"
        if label not in dataset:
            dataset[label] = []
        dataset[label].append(predict_response)

    _ensure_parent_dir(write_path)
    with open(write_path, "w", encoding="utf-8") as fout:
        json.dump(dataset, fout, indent=4, ensure_ascii=False)


def _checkpoint_sort_key(path):
    name = os.path.basename(path)
    match = re.search(r"checkpoint-(\d+)", name)
    if match:
        return int(match.group(1))
    return float("inf")


def _ensure_parent_dir(file_path):
    parent_dir = os.path.dirname(os.path.abspath(file_path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def discover_checkpoints(checkpoint_root, checkpoint_pattern="checkpoint-"):
    if checkpoint_root is None:
        return []
    if not os.path.isdir(checkpoint_root):
        raise ValueError(f"checkpoint_root is not a directory: {checkpoint_root}")

    checkpoints = []
    for entry in os.listdir(checkpoint_root):
        full_path = os.path.join(checkpoint_root, entry)
        if not os.path.isdir(full_path):
            continue
        if checkpoint_pattern and checkpoint_pattern not in entry:
            continue
        if entry.startswith("checkpoint-"):
            checkpoints.append(full_path)

    checkpoints.sort(key=_checkpoint_sort_key)
    return checkpoints


def load_model_and_tokenizer(model_name, checkpoint_path=None, ptuning_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if checkpoint_path is not None and os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
        if PeftModelForCausalLM is None:
            raise ImportError("peft is required for LoRA adapter checkpoints. Please install peft first.")

        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()
        model = PeftModelForCausalLM.from_pretrained(model, checkpoint_path, trust_remote_code=True)
        model = model.eval()
        return model, tokenizer

    if ptuning_path is not None:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, pre_seq_len=128)
        model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
        prefix_state_dict = torch.load(os.path.join(ptuning_path, "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
        model = model.half().cuda()
        model.transformer.prefix_encoder.float()
        model = model.eval()
        return model, tokenizer

    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()
    model = model.eval()
    return model, tokenizer


def infer_one_checkpoint(
    model,
    tokenizer,
    test_messages,
    traffic_task,
    max_new_tokens=1024,
    repetition_penalty=1.2,
    print_response=False,
    detection_top_p=0.8,
    detection_temperature=0.8,
    generation_top_p=0.8,
    generation_temperature=0.8,
):
    predict_responses = []

    for messages in tqdm(test_messages):
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        if traffic_task == "detection":
            gen_ids = model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=detection_top_p,
                temperature=detection_temperature,
                repetition_penalty=repetition_penalty,
                eos_token_id=model.config.eos_token_id,
            )
        elif traffic_task == "generation":
            gen_ids = model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=generation_top_p,
                temperature=generation_temperature,
                repetition_penalty=repetition_penalty,
                eos_token_id=model.config.eos_token_id,
            )
        else:
            raise ValueError(f"Unsupported traffic_task: {traffic_task}")

        gen_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, gen_ids)
        ]
        response = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

        if print_response:
            print("response:", response)
        predict_responses.append(response)

    return predict_responses


def _metrics_pass_constraints(metrics, min_accuracy, min_precision, min_recall):
    return (
        metrics["accuracy"] >= min_accuracy
        and metrics["precision"] >= min_precision
        and metrics["recall"] >= min_recall
    )


def main(
    model_name,
    test_file: str = None,
    label_file: str = None,
    traffic_task: str = None,
    ptuning_path: str = None,
    checkpoint_root: str = None,
    checkpoint_pattern: str = "checkpoint-",
    max_samples: int = 1000,
    max_new_tokens: int = 1024,
    repetition_penalty: float = 1.2,
    detection_top_p: float = 0.8,
    detection_temperature: float = 0.8,
    generation_top_p: float = 0.8,
    generation_temperature: float = 0.8,
    random_seed: int = 42,
    min_accuracy: float = 0.0,
    min_precision: float = 0.0,
    min_recall: float = 0.0,
    output_result_file: str = "batch_eval_results.json",
    print_response: bool = False,
    **kwargs,
):
    del kwargs

    if test_file is None:
        print("No Test file provided. Exiting.")
        sys.exit(1)
    if traffic_task is None:
        print("No traffic_task provided. Exiting.")
        sys.exit(1)

    assert os.path.exists(test_file), f"Provided Test file does not exist {test_file}"

    with open(test_file, "r", encoding="utf-8") as fin:
        test_set = fin.readlines()

    test_messages, target_responses = test_set_to_prompt(test_set)
    test_messages = test_messages[:max_samples]
    target_responses = target_responses[:max_samples]

    if random_seed is not None:
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

    if checkpoint_root:
        checkpoints = discover_checkpoints(checkpoint_root, checkpoint_pattern=checkpoint_pattern)
        if not checkpoints:
            print(f"No checkpoints found in {checkpoint_root}, fallback to single model evaluation.")
            checkpoints = [None]
    else:
        checkpoints = [None]

    all_results = []
    best_result = None
    best_passed = None

    for checkpoint in checkpoints:
        display_name = checkpoint if checkpoint is not None else model_name
        print(f"\n===== Evaluating: {display_name} =====")

        model, tokenizer = load_model_and_tokenizer(model_name, checkpoint_path=checkpoint, ptuning_path=ptuning_path)
        predict_responses = infer_one_checkpoint(
            model=model,
            tokenizer=tokenizer,
            test_messages=test_messages,
            traffic_task=traffic_task,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            print_response=print_response,
            detection_top_p=detection_top_p,
            detection_temperature=detection_temperature,
            generation_top_p=generation_top_p,
            generation_temperature=generation_temperature,
        )

        if traffic_task == "detection":
            if not label_file:
                raise ValueError("label_file is required for detection task")
            metrics = td_evaluation(predict_responses, target_responses, label_file)
            passed = _metrics_pass_constraints(metrics, min_accuracy, min_precision, min_recall)
            result = {
                "checkpoint": display_name,
                "passed_constraints": passed,
                **metrics,
            }
            all_results.append(result)

            if best_result is None or result["f1"] > best_result["f1"]:
                best_result = result
            if passed and (best_passed is None or result["f1"] > best_passed["f1"]):
                best_passed = result

            print(
                "metrics -> "
                f"acc: {result['accuracy']:.6f}, "
                f"precision: {result['precision']:.6f}, "
                f"recall: {result['recall']:.6f}, "
                f"f1: {result['f1']:.6f}, "
                f"passed: {result['passed_constraints']}"
            )
        else:
            write_path = "generation.json" if checkpoint is None else f"generation_{os.path.basename(checkpoint)}.json"
            tg_evaluation(predict_responses, target_responses, test_messages, write_path=write_path)
            print(f"generation result saved to: {write_path}")

        del model
        torch.cuda.empty_cache()

    if traffic_task == "detection":
        selected_best = best_passed if best_passed is not None else best_result
        if selected_best is None:
            print("No detection evaluation result generated.")
            return

        if best_passed is None:
            print("\nNo checkpoint satisfies all constraints. Fallback to highest F1 checkpoint.")

        print("\n===== Best Checkpoint =====")
        print("checkpoint:", selected_best["checkpoint"])
        print("acc:", selected_best["accuracy"])
        print("precision:", selected_best["precision"])
        print("recall:", selected_best["recall"])
        print("f1:", selected_best["f1"])
        print("passed_constraints:", selected_best["passed_constraints"])
        print("confusion matrix:\n", selected_best["confusion_matrix"])
        print("classification report:\n", selected_best["classification_report"])

        _ensure_parent_dir(output_result_file)
        with open(output_result_file, "w", encoding="utf-8") as fout:
            json.dump(
                {
                    "constraints": {
                        "min_accuracy": min_accuracy,
                        "min_precision": min_precision,
                        "min_recall": min_recall,
                    },
                    "best_checkpoint": selected_best,
                    "all_results": all_results,
                },
                fout,
                indent=2,
                ensure_ascii=False,
            )
        print(f"All evaluation results saved to: {output_result_file}")


if __name__ == "__main__":
    fire.Fire(main)
