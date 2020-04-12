#!/usr/bin/env bash

SCRIPT_FILE="sectlabel_elmo_bilstm.py"

python ${SCRIPT_FILE} \
--exp_name "parsect_elmo_bi_lstm_lc" \
--exp_dir_path  "./output" \
--model_save_dir "./output/checkpoints" \
--device cuda:1 \
--bs 64 \
--emb_type "glove_6B_300" \
--hidden_dim 256 \
--lr 1e-4 \
--bidirectional \
--combine_strategy concat \
--epochs 50 \
--save_every 5 \
--log_train_metrics_every 25 \
--sample_proportion 1



