#!/bin/bash
set -e

echo "===== START RNN SCALING RUN ====="
date

PYTHON=python
SCRIPT=train_rnn.py

DATA_DIR=data/abc_char
BATCH_SIZE=256
BLOCK_SIZE=256
LR=3e-4
DEVICE=cuda

echo ">>> Running RNN-S (small)"
$PYTHON $SCRIPT \
  --data_dir $DATA_DIR \
  --out_dir out-rnn-small \
  --hidden_size 256 \
  --num_layers 2 \
  --batch_size $BATCH_SIZE \
  --block_size $BLOCK_SIZE \
  --learning_rate $LR \
  --device $DEVICE \
  > out-rnn-small.log 2>&1

echo ">>> Running RNN-M (medium)"
$PYTHON $SCRIPT \
  --data_dir $DATA_DIR \
  --out_dir out-rnn-medium \
  --hidden_size 512 \
  --num_layers 3 \
  --batch_size $BATCH_SIZE \
  --block_size $BLOCK_SIZE \
  --learning_rate $LR \
  --device $DEVICE \
  > out-rnn-medium.log 2>&1

echo ">>> Running RNN-L (large)"
$PYTHON $SCRIPT \
  --data_dir $DATA_DIR \
  --out_dir out-rnn-large \
  --hidden_size 768 \
  --num_layers 4 \
  --batch_size $BATCH_SIZE \
  --block_size $BLOCK_SIZE \
  --learning_rate $LR \
  --device $DEVICE \
  > out-rnn-large.log 2>&1

echo ">>> Running RNN-XL (xl)"
$PYTHON $SCRIPT \
  --data_dir $DATA_DIR \
  --out_dir out-rnn-xl \
  --hidden_size 1024 \
  --num_layers 5 \
  --batch_size $BATCH_SIZE \
  --block_size $BLOCK_SIZE \
  --learning_rate $LR \
  --device $DEVICE \
  > out-rnn-xl.log 2>&1

echo "===== ALL RNN MODELS FINISHED ====="
date
