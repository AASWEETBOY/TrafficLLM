import subprocess
from pathlib import Path
from typing import List
import fire


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FT_DIR = PROJECT_ROOT / "FT"
DATASETS_ROOT = PROJECT_ROOT / "datasets"
BASE_MODEL_DIR = PROJECT_ROOT / "models" / "glm-4-9b-chat"
STAGE1_OUTPUT_DIR = PROJECT_ROOT / "models" / "glm-4-9b-chat-lora" / "instructions"


def _resolve_project_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _run(cmd, cwd: Path) -> None:
    print("[RUN]", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _list_available_stage2_datasets(datasets_root: Path) -> List[str]:
    if not datasets_root.is_dir():
        return []
    names = []
    for item in datasets_root.iterdir():
        if item.is_dir() and item.name != "instructions":
            names.append(item.name)
    return sorted(names)


def _stage1_already_done(output_dir: Path) -> bool:
    if not output_dir.exists():
        return False
    if (output_dir / "adapter_config.json").exists():
        return True
    if (output_dir / "trainer_state.json").exists():
        return True
    if any(output_dir.glob("checkpoint-*")):
        return True
    return False


def stage1_tuning(
    model_dir: str = str(BASE_MODEL_DIR),
    datasets_dir: str = str(DATASETS_ROOT / "instructions"),
    config_file: str = "configs/lora_stage1.yaml",
    force: bool = False,
) -> None:
    """Run stage1 instruction tuning once by default."""
    if _stage1_already_done(STAGE1_OUTPUT_DIR) and not force:
        print(f"[SKIP] stage1 already done: {STAGE1_OUTPUT_DIR}")
        return

    cmd = [
        "bash",
        "train_stage1.sh",
        datasets_dir,
        model_dir,
        config_file,
    ]
    _run(cmd, cwd=FT_DIR)


def stage2_tuning(
    dataset: str,
    model_dir: str = str(BASE_MODEL_DIR),
    datasets_root: str = str(DATASETS_ROOT),
    config_file: str = "configs/lora_stage2.yaml",
) -> None:
    """Run stage2 tuning for a specific traffic dataset."""
    ds_root = Path(datasets_root)
    available = _list_available_stage2_datasets(ds_root)
    if dataset not in available:
        raise ValueError(
            f"Invalid dataset '{dataset}'. Available datasets: {', '.join(available)}"
        )

    cmd = [
        "bash",
        "train_stage2.sh",
        dataset,
        str(ds_root),
        model_dir,
        config_file,
    ]
    _run(cmd, cwd=FT_DIR)


def main(
    model_name: str = str(BASE_MODEL_DIR),
    tuning_data: str = str(DATASETS_ROOT),
    adaptation_task: str = "update",
    task_name: str = None,
    dataset: str = None,
    datasets: str = None,
    run_stage1: bool = True,
    force_stage1: bool = False,
    list_datasets: bool = False,
    stage1_config: str = "configs/lora_stage1.yaml",
    stage2_config: str = "configs/lora_stage2.yaml",
):
    """
    Examples:
      python APEFT/apeft.py --list_datasets=True
      python APEFT/apeft.py --dataset=csic-2010
      python APEFT/apeft.py --dataset=dapt-2020 --run_stage1=False
      python APEFT/apeft.py --task_name=iscx-vpn-2016
    """
    datasets_root = _resolve_project_path(tuning_data)
    model_path = _resolve_project_path(model_name)
    stage1_cfg = _resolve_project_path(stage1_config)
    stage2_cfg = _resolve_project_path(stage2_config)

    available = _list_available_stage2_datasets(datasets_root)
    if list_datasets:
        print("Available stage2 datasets:")
        for name in available:
            print(f"- {name}")
        return

    # Keep backward compatibility: dataset can come from dataset/datasets/task_name.
    selected_dataset = dataset or datasets or task_name

    if run_stage1:
        stage1_tuning(
            model_dir=str(model_path),
            datasets_dir=str(datasets_root / "instructions"),
            config_file=str(stage1_cfg),
            force=force_stage1,
        )

    if selected_dataset:
        stage2_tuning(
            dataset=selected_dataset,
            model_dir=str(model_path),
            datasets_root=str(datasets_root),
            config_file=str(stage2_cfg),
        )
        return

    if adaptation_task in ("update", "register") and task_name:
        stage2_tuning(
            dataset=task_name,
            model_dir=str(model_path),
            datasets_root=str(datasets_root),
            config_file=str(stage2_cfg),
        )
        return

    print("[INFO] stage1 finished. No stage2 dataset specified.")
    print("[INFO] Use --dataset=<name> to run stage2.")
    print(f"[INFO] Available datasets: {', '.join(available)}")


if __name__ == "__main__":
    fire.Fire(main)
