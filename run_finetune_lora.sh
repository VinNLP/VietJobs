#!/bin/bash

BASE_OUTPUT_DIR="/workspace/checkpoints_other_data"
DS_CONFIG_FILE="ds_config.json"

declare -a MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
)

declare -a TRAIN_FILES=(
    "data/job_classification/train.json"
    "data/salary_estimation/train.json"

)

declare -a DEV_FILES=(
    "data/job_classification/dev.json"
    "data/salary_estimation/dev.json"
)

LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.2
TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"

LEARNING_RATE=5e-5
NUM_EPOCHS=2
MICRO_BATCH_SIZE=4
EFFECTIVE_BATCH_SIZE=64
MAX_SEQ_LENGTH=512
EVAL_SAVE_STEPS=200

mkdir -p "$BASE_OUTPUT_DIR"
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
[ "$GPU_COUNT" -lt 1 ] && echo "No GPUs detected." && exit 1

for MODEL_NAME in "${MODELS[@]}"; do
    MODEL_SHORT_NAME=$(basename "$MODEL_NAME")

    for IDX in "${!TRAIN_FILES[@]}"; do
        TRAIN_FILE="${TRAIN_FILES[$IDX]}"
        DEV_FILE="${DEV_FILES[$IDX]}"
        TRAIN_BASE=$(basename "$TRAIN_FILE" .json)
        PARENT_DIR=$(dirname "$TRAIN_FILE")
        OUT_DIR="${BASE_OUTPUT_DIR}/${PARENT_DIR}/${MODEL_SHORT_NAME}_${TRAIN_BASE}"

        [ -d "$OUT_DIR" ] && continue
        mkdir -p "$OUT_DIR"

        GRAD_ACC=$(( EFFECTIVE_BATCH_SIZE / (MICRO_BATCH_SIZE * GPU_COUNT) ))

        deepspeed --num_gpus="$GPU_COUNT" lora_finetune.py \
            --model_name "$MODEL_NAME" \
            --output_dir "$OUT_DIR" \
            --train_file "$TRAIN_FILE" \
            --dev_file "$DEV_FILE" \
            --lora_r "$LORA_R" \
            --lora_alpha "$LORA_ALPHA" \
            --lora_dropout "$LORA_DROPOUT" \
            --target_modules "$TARGET_MODULES" \
            --micro_batch_size "$MICRO_BATCH_SIZE" \
            --gradient_accumulation_steps "$GRAD_ACC" \
            --learning_rate "$LEARNING_RATE" \
            --num_epochs "$NUM_EPOCHS" \
            --max_seq_length "$MAX_SEQ_LENGTH" \
            --eval_save_steps "$EVAL_SAVE_STEPS" \
            --deepspeed "$DS_CONFIG_FILE"

        [ $? -eq 0 ] && echo "Done: $MODEL_SHORT_NAME | $TRAIN_BASE" || echo "Failed: $MODEL_SHORT_NAME | $TRAIN_BASE"
    done
done