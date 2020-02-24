#!/usr/bin/env bash

SCRIPT_FILE="sectlabel_bow_elmo.py"

python ${SCRIPT_FILE} \
--exp_name "sectlabel_bow_elmo" \
--exp_dir_path  "./output" \
--model_save_dir "./output/checkpoints" \
--vocab_store_location "./output/vocab.json" \
--device cuda:0 \
--layer_aggregation last \
--word_aggregation sum \
--bs 10 \
--lr 1e-4 \
--epochs 1 \
--save_every 5 \
--log_train_metrics_every 10 \
--sample_proportion 0.1
