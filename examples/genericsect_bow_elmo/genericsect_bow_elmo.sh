#!/usr/bin/env bash

SCRIPT_FILE="genericsect_bow_elmo.py"


python ${SCRIPT_FILE} \
--exp_name "genericsect_bow_elmo" \
--exp_dir_path  "./output" \
--model_save_dir "./output/checkpoints" \
--device cpu \
--layer_aggregation last \
--word_aggregation sum \
--bs 10 \
--lr 1e-4 \
--epochs 1 \
--save_every 5 \
--sample_proportion 0.1 \
--log_train_metrics_every 10

