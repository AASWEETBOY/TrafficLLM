#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

NUM_GPUS=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

DATASET_NAME="${1:-}"
DATASETS_ROOT="${2:-../datasets}"
MODEL_DIR="${3:-../models/glm-4-9b-chat}"
CONFIG_TEMPLATE="${4:-configs/lora_stage2.yaml}"

if [[ -z "${DATASET_NAME}" ]]; then
	echo "[ERROR] dataset name is required."
	echo "Usage: bash train_stage2.sh <dataset_name> [datasets_root] [model_dir] [config_template]"
	exit 1
fi

if [[ "${DATASET_NAME}" == "instructions" ]]; then
	echo "[ERROR] stage2 does not support dataset 'instructions'."
	exit 1
fi

DATASET_DIR="${DATASETS_ROOT}/${DATASET_NAME}"
if [[ ! -d "${DATASET_DIR}" ]]; then
	echo "[ERROR] dataset directory not found: ${DATASET_DIR}"
	exit 1
fi

TRAIN_FILE="GLM4_${DATASET_NAME}_detection_packet_train.jsonl"
VAL_FILE="GLM4_${DATASET_NAME}_detection_packet_test_with_label.jsonl"
TEST_FILE="GLM4_${DATASET_NAME}_detection_packet_test_with_label.jsonl"
OUTPUT_DIR="../models/glm-4-9b-chat-lora/${DATASET_NAME}_detection_packet"

for f in "${TRAIN_FILE}" "${VAL_FILE}" "${TEST_FILE}"; do
	if [[ ! -f "${DATASET_DIR}/${f}" ]]; then
		echo "[ERROR] expected file not found: ${DATASET_DIR}/${f}"
		exit 1
	fi
done

RUNTIME_CONFIG="configs/lora_stage2.${DATASET_NAME}.runtime.yaml"

python3 - "${CONFIG_TEMPLATE}" "${RUNTIME_CONFIG}" "${TRAIN_FILE}" "${VAL_FILE}" "${TEST_FILE}" "${OUTPUT_DIR}" <<'PY'
from pathlib import Path
import sys
import ruamel.yaml as yaml

template, runtime, train_f, val_f, test_f, output_dir = sys.argv[1:]
parser = yaml.YAML(typ='safe', pure=True)
data = parser.load(Path(template))

data['data_config']['train_file'] = train_f
data['data_config']['val_file'] = val_f
data['data_config']['test_file'] = test_f
data['training_args']['output_dir'] = output_dir

writer = yaml.YAML()
writer.default_flow_style = False
writer.indent(mapping=2, sequence=4, offset=2)
with Path(runtime).open('w', encoding='utf-8') as fout:
    writer.dump(data, fout)
PY

PHASE="GLM4-TASK-tuning_stage2-${DATASET_NAME}"
LOG_PATH="Logs/${PHASE}/"
LOG_TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_PATH}/train_${LOG_TS}.log"
LATEST_LINK="${LOG_PATH}/train_latest.log"

mkdir -p "$LOG_PATH"
touch "${LOG_FILE}"
ln -sfn "$(basename "${LOG_FILE}")" "${LATEST_LINK}"

echo "[INFO] log file: ${LOG_FILE}"
echo "[INFO] dataset: ${DATASET_NAME}"
echo "[INFO] runtime config: ${RUNTIME_CONFIG}"

if [[ "${NUM_GPUS}" -eq 1 ]]; then
	echo "[INFO] single GPU detected, run without torchrun to avoid elastic SIGSEGV"
	python finetune.py \
	"${DATASET_DIR}" \
	"${MODEL_DIR}" \
	"${RUNTIME_CONFIG}" \
	2>&1 | tee -a "${LOG_FILE}"
else
	torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS finetune.py \
	"${DATASET_DIR}" \
	"${MODEL_DIR}" \
	"${RUNTIME_CONFIG}" \
	2>&1 | tee -a "${LOG_FILE}"
fi
