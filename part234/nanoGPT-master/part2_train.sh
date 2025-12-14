#!/bin/bash
set -e  # 任意一步失败就退出

LOG_DIR="scaling_logs"
mkdir -p $LOG_DIR

echo "===== START SCALING RUN ====="
date

# Tiny
echo ">>> Running Tiny model"
python train.py config/train_abc_char_tiny.py \
  | tee $LOG_DIR/tiny.log

# Small
echo ">>> Running Small model"
python train.py config/train_abc_char_small.py \
  | tee $LOG_DIR/small.log

# Medium
echo ">>> Running Medium model"
python train.py config/train_abc_char_medium.py \
  | tee $LOG_DIR/medium.log

# Large
echo ">>> Running Large model"
python train.py config/train_abc_char_large.py \
  | tee $LOG_DIR/large.log

# XL
echo ">>> Running XL model"
python train.py config/train_abc_char_xl.py \
  | tee $LOG_DIR/xl.log

echo "===== ALL MODELS FINISHED ====="
date
