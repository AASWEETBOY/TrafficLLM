#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=1

MODEL_DIR="${1:-../output/checkpoint-3000}"
TEST_FILE="${2:-../datas/test.jsonl}"
TARGET_PATH="${3:-../datas/test_with_label.jsonl}"
LABEL_FILE="${4:-../datas/label.json}"
EVALUATION_RESULT_FILE="${5:-../output/evaluation_metric_result}"

PHASE="GLM4-inference"
LOG_PATH="Logs/${PHASE}/"
LOG_TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_PATH}/inference_${LOG_TS}.log"
LATEST_LINK="${LOG_PATH}/inference_latest.log"

mkdir -p "${LOG_PATH}"
touch "${LOG_FILE}"
ln -sfn "$(basename "${LOG_FILE}")" "${LATEST_LINK}"

echo "[INFO] log file: ${LOG_FILE}"
echo "[INFO] model_dir: ${MODEL_DIR}"
echo "[INFO] test_file: ${TEST_FILE}"
echo "[INFO] target_path: ${TARGET_PATH}"
echo "[INFO] label_file: ${LABEL_FILE}"
echo "[INFO] evaluation_result_file: ${EVALUATION_RESULT_FILE}"

python inference.py \
	"${MODEL_DIR}" \
	"${TEST_FILE}" \
	"${TARGET_PATH}" \
	"${LABEL_FILE}" \
	"${EVALUATION_RESULT_FILE}" \
	2>&1 | tee -a "${LOG_FILE}"
