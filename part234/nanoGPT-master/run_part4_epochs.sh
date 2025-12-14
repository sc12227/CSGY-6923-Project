#!/bin/bash
set -e

BASE_OUT="part4_results"
TRAIN_OUT="${BASE_OUT}/train_run"
CONFIG="config/train_abc_char_best_1epoch.py"

START_EPOCH=4
END_EPOCH=10

ITERS_PER_EPOCH=15000   

mkdir -p "${BASE_OUT}"
mkdir -p "${TRAIN_OUT}"

for ((EPOCH=${START_EPOCH}; EPOCH<=${END_EPOCH}; EPOCH++)); do
    EPOCH_DIR=$(printf "%s/epoch_%02d" "${BASE_OUT}" "${EPOCH}")
    mkdir -p "${EPOCH_DIR}"

    TARGET_ITERS=$((EPOCH * ITERS_PER_EPOCH))

    echo "======================================"
    echo " Running Epoch ${EPOCH}"
    echo " Training dir : ${TRAIN_OUT}"
    echo " Snapshot dir : ${EPOCH_DIR}"
    echo " Target iters : ${TARGET_ITERS}"
    echo "======================================"

    if [ "${EPOCH}" -eq 1 ]; then
        INIT_FROM="scratch"
    else
        INIT_FROM="resume"
    fi

    python train.py "${CONFIG}" \
        --out_dir="${TRAIN_OUT}" \
        --init_from="${INIT_FROM}" \
        --max_iters="${TARGET_ITERS}" \
        --lr_decay_iters="${TARGET_ITERS}" \
        2>&1 | tee "${TRAIN_OUT}/log.txt"

    cp "${TRAIN_OUT}/ckpt.pt" "${EPOCH_DIR}/ckpt.pt"
    cp "${TRAIN_OUT}/log.txt" "${EPOCH_DIR}/log.txt"

    mkdir -p "${EPOCH_DIR}/samples"
    python sample.py \
        --out_dir="${EPOCH_DIR}" \
        --num_samples=5 \
        --max_new_tokens=1500 \
        --temperature=0.9 \
        --top_k=50 \
        > "${EPOCH_DIR}/samples/sample_log.txt"

    echo "[OK] Epoch ${EPOCH} saved to ${EPOCH_DIR}"
done

echo "===== ALL EPOCHS FINISHED ====="
