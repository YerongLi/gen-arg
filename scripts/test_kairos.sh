#!/usr/bin/env bash
set -e 
set -x 
CKPT_NAME=gen-KAIROS
MODEL=constrained-gen

rm -rf checkpoints/${CKPT_NAME}-pred 
PYTHONPATH=. /scratch/yerong/sha/bin/python train.py --model=$MODEL \
    --ckpt_name=${CKPT_NAME}-pred \
    --load_ckpt=/scratch/yerong/data/sha/WikiEvents/epoch=2-v0.ckpt \
    --dataset=KAIROS \
    --eval_only \
    --train_file=data/wikievents/train.jsonl \
    --val_file=data/wikievents/dev.jsonl \
    --test_file=data/wikievents/test.jsonl \
    --train_batch_size=2 \
    --eval_batch_size=2 \
    --learning_rate=3e-5 \
    --accumulate_grad_batches=2 \
    --num_train_epochs=1

python src/genie/scorer.py --gen-file=checkpoints/$CKPT_NAME-pred/predictions.jsonl \
--test-file=data/wikievents/test.jsonl \
--dataset=KAIROS \
--coref-file=data/wikievents/coref/test.jsonlines \
--head-only \
--coref 
