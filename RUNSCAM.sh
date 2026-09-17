#!/usr/bin/env bash

# Train RGAM+SCAM from the official trained TEA baseline checkpoint with all
# parameters unfrozen, run inference with the last checkpoint, and evaluate
# the three requested AP metrics. The training output, including each epoch's
# loss summary and RGAM/SCAM alpha values, is appended to the evaluation log.
# Run this script from any directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ARCH="${ARCH:-TEAresnet_50}"
DATASET="${DATASET:-IODVideo}"
SPLIT="${SPLIT:-1}"
K="${K:-8}"
GPUS="${GPUS:-0,1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
TRAIN_MASTER_BATCH_SIZE="${TRAIN_MASTER_BATCH_SIZE:-8}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-20}"
INFER_MASTER_BATCH_SIZE="${INFER_MASTER_BATCH_SIZE:-10}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TRAIN_EXP_ID="TEA_STA_RGAM_SCAM_ALLFT_K8S1"
ACT_MODEL_NAME="TEA_STA_RGAM_SCAM_ALLFT_E8_S1"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-/home/yangjiao/llrx_workplace/Projects/IOD-Video/experiment/result_model/TEA_STA_K8S1/model_last.pth}"

MODEL_DIR="/home/yangjiao/llrx_workplace/Projects/IOD-Video/experiment/result_model/TEA_STA_RGAM_SCAM_ALLFT_K8S1"
INFERENCE_DIR="/home/yangjiao/llrx_workplace/Projects/IOD-Video/result/inference_TEA_STA_RGAM_SCAM_ALLFT_E8_S1"
METRICS_NAME="TEA_STA_RGAM_SCAM_ALLFT_E8_S1_eval.txt"
METRICS_LOG="/home/yangjiao/llrx_workplace/Projects/IOD-Video/result/${METRICS_NAME}"

mkdir -p "/home/yangjiao/llrx_workplace/Projects/IOD-Video/result" \
         "/home/yangjiao/llrx_workplace/Projects/IOD-Video/experiment/result_model"

# Remove only this run's output directory so stale frame detections cannot be
# mixed with the new checkpoint's results.
if [[ -d "$INFERENCE_DIR" ]]; then
    rm -rf -- "$INFERENCE_DIR"
fi

: > "$METRICS_LOG"
{
    echo "RGAM+SCAM run: $TRAIN_EXP_ID"
    echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Initialization: official TEA baseline checkpoint"
    echo "Baseline checkpoint: $BASELINE_CHECKPOINT"
    echo "Training: RGAM+SCAM, all parameters unfrozen, 8 epochs"
    echo "Model directory: $MODEL_DIR"
    echo "Inference directory: $INFERENCE_DIR"
    echo
} >> "$METRICS_LOG"

run_logged() {
    echo "[RUN] $*" | tee -a "$METRICS_LOG"
    "$@" 2>&1 | tee -a "$METRICS_LOG"
    local command_status=${PIPESTATUS[0]}
    if [[ $command_status -ne 0 ]]; then
        echo "[FAILED] status=$command_status" | tee -a "$METRICS_LOG"
        exit "$command_status"
    fi
}

echo "===== Train RGAM+SCAM from official weights for 8 epochs =====" | tee -a "$METRICS_LOG"
if [[ ! -f "$BASELINE_CHECKPOINT" ]]; then
    echo "Official baseline checkpoint not found: $BASELINE_CHECKPOINT" | tee -a "$METRICS_LOG" >&2
    exit 1
fi

run_logged "$PYTHON_BIN" -u train.py \
    --task train \
    --exp_id "$TRAIN_EXP_ID" \
    --K "$K" \
    --gpus "$GPUS" \
    --batch_size "$TRAIN_BATCH_SIZE" \
    --master_batch_size "$TRAIN_MASTER_BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --lr 1e-4 \
    --num_epochs 8 \
    --lr_step 5,7 \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --arch "$ARCH" \
    --pretrain_model none \
    --load_model "$BASELINE_CHECKPOINT" \
    --load_model_weights_only \
    --rgb_model "$MODEL_DIR" \
    --use_rgam \
    --use_scam \
    --scam_reduction 16 \
    --scam_spatial_kernel 4 \
    --scam_channel_group 4 \
    --scam_residual_scale 0.1

LAST_CHECKPOINT="$MODEL_DIR/model_last.pth"
if [[ ! -f "$LAST_CHECKPOINT" ]]; then
    echo "Last checkpoint not found after training: $LAST_CHECKPOINT" | tee -a "$METRICS_LOG" >&2
    exit 1
fi

echo "===== Inference with the last checkpoint =====" | tee -a "$METRICS_LOG"
run_logged "$PYTHON_BIN" det.py \
    --task normal \
    --exp_id "${ACT_MODEL_NAME}_det" \
    --K "$K" \
    --gpus "$GPUS" \
    --batch_size "$INFER_BATCH_SIZE" \
    --master_batch_size "$INFER_MASTER_BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --rgb_model "$LAST_CHECKPOINT" \
    --inference_dir "$INFERENCE_DIR" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --arch "$ARCH" \
    --use_rgam \
    --use_scam \
    --scam_reduction 16 \
    --scam_spatial_kernel 4 \
    --scam_channel_group 4 \
    --scam_residual_scale 0.1

run_act() {
    local task="$1"
    local threshold="${2:-}"
    local -a args=(
        --pkl_ACT 1
        --task "$task"
        --K "$K"
        --inference_dir "$INFERENCE_DIR"
        --dataset "$DATASET"
        --split "$SPLIT"
        --model_name "$ACT_MODEL_NAME"
        --exp_id "$METRICS_NAME"
    )
    if [[ -n "$threshold" ]]; then
        args+=(--th "$threshold")
    fi
    echo "[RUN] ACT.py ${args[*]}" | tee -a "$METRICS_LOG"
    "$PYTHON_BIN" ACT.py "${args[@]}"
}

echo "===== Evaluate mAP@0.5 =====" | tee -a "$METRICS_LOG"
run_act frameAP 0.5

echo "===== Evaluate mAP@0.75 =====" | tee -a "$METRICS_LOG"
run_act frameAP 0.75

echo "===== Evaluate mAP@0.5:0.95 =====" | tee -a "$METRICS_LOG"
run_act frameAP_all

echo "===== Finished =====" | tee -a "$METRICS_LOG"
echo "Metrics log: $METRICS_LOG" | tee -a "$METRICS_LOG"
