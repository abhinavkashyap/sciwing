#!/usr/bin/env bash

EXPERIMENT_PREFIX="conll_seq_crf"
SCRIPT_FILE="conll_seq_crf.py"

python ${SCRIPT_FILE} \
--exp_name ${EXPERIMENT_PREFIX} \
--exp_dir_path  "./output" \
--model_save_dir "./output/checkpoints" \
--device cpu \
--dropout 0.5 \
--reg 0 \
--bs 128 \
--emb_type "glove_6B_300" \
--hidden_dim 300 \
--num_layers=1 \
--bidirectional \
--lr 1e-3 \
--combine_strategy concat \
--epochs 1 \
--save_every 20 \
--log_train_metrics_every 3 \
--sample_proportion 0.1