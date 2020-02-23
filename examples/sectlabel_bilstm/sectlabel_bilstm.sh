#!/usr/bin/env bash


SCRIPT_FILE="sectlabel_bilstm.py"


python ${SCRIPT_FILE} \
--exp_name "sectlabel_bilstm" \
--exp_dir_path  "./output" \
--model_save_dir "./output/checkpoints" \
--emb_type "glove_6B_50" \
--device cuda:0 \
--bs 32 \
--hidden_dim 512 \
--lr 1e-3 \
--bidirectional \
--combine_strategy concat \
--epochs 1 \
--save_every 5 \
--log_train_metrics_every 50 \
--sample_proportion 0.01
