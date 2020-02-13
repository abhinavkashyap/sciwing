#!/usr/bin/env bash


SCRIPT_FILE="sectlabel_bow_bert.py"

python ${SCRIPT_FILE} \
--exp_name "sectlabel_bow_bert" \
--exp_dir_path  "./output" \
--model_save_dir "./output/checkpoints" \
--device cpu \
--bert_type "bert-base-uncased" \
--bs 32 \
--lr 1e-2 \
--epochs 1 \
--save_every 1 \
--log_train_metrics_every 50 \
--sample_proportion 1.0