#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

NUM_GPUS=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1

DATA_DIR="${1:-../datasets/instructions}"
MODEL_DIR="${2:-../models/glm-4-9b-chat}"
CONFIG_FILE="${3:-configs/lora_stage1.yaml}"

PHASE="GLM4-TASK-tuning_stage1"
LOG_PATH="Logs/${PHASE}/"
LOG_TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_PATH}/train_${LOG_TS}.log"
LATEST_LINK="${LOG_PATH}/train_latest.log"

mkdir -p "$LOG_PATH"
touch "${LOG_FILE}"
ln -sfn "$(basename "${LOG_FILE}")" "${LATEST_LINK}"

echo "[INFO] log file: ${LOG_FILE}"

if [[ "${NUM_GPUS}" -eq 1 ]]; then
	echo "[INFO] single GPU detected, run without torchrun to avoid elastic SIGSEGV"
	python finetune.py \
	"${DATA_DIR}" \
	"${MODEL_DIR}" \
	"${CONFIG_FILE}" \
	2>&1 | tee -a "${LOG_FILE}"
else
	torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS finetune.py \
	"${DATA_DIR}" \
	"${MODEL_DIR}" \
	"${CONFIG_FILE}" \
	2>&1 | tee -a "${LOG_FILE}"
fi
