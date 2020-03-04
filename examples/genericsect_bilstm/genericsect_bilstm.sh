#!/usr/bin/env bash

SCRIPT_FILE="genericsect_bilstm.py"

python ${SCRIPT_FILE} \
--exp_name "gensect_bi_lstm_lc" \
--exp_dir_path  "./output" \
--model_save_dir "./output/checkpoints" \
--device cpu \
--bs 32 \
--emb_type "glove_6B_50" \
--hidden_dim 512 \
--lr 1e-3 \
--bidirectional \
--combine_strategy concat \
--epochs 1 \
--save_every 1 \
--log_train_metrics_every 50 \
--sample_proportion 1