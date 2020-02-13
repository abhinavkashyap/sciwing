#!/usr/bin/env bash

SCRIPT_FILE="sectlabel_elmo_bilstm.py"

python ${SCRIPT_FILE} \
--exp_name "parsect_elmo_bi_lstm_lc" \
--exp_dir_path  "./output" \
--model_save_dir "./output/checkpoints" \
--device cpu \
--bs 10 \
--emb_type "glove_6B_50" \
--hidden_dim 512 \
--lr 1e-3 \
--bidirectional \
--combine_strategy concat \
--epochs 1 \
--save_every 5 \
--log_train_metrics_every 5 \
--sample_proportion 0.01



