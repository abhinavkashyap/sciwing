#!/usr/bin/env bash

SCRIPT_FILE="genericsect_bow.py"

python ${SCRIPT_FILE} \
--exp_name "genericsect_bow_random" \
--exp_dir_path  "./output" \
--model_save_dir "./output/checkpoints" \
--bs 32 \
--emb_type "glove_6B_50" \
--lr 1e-4 \
--epochs 1 \
--save_every 1 \
--log_train_metrics_every 50 \
--sample_proportion 0.01