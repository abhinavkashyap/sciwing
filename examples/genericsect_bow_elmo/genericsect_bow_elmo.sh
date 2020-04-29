#!/usr/bin/env bash

SCRIPT_FILE="genericsect_bow_elmo.py"


python ${SCRIPT_FILE} \
--exp_name "genericsect_bow_elmo" \
--exp_dir_path  "./genericsect_bow_elmo" \
--model_save_dir "./genericsect_bow_elmo" \
--device cuda:7 \
--layer_aggregation last \
--word_aggregation sum \
--bs 10 \
--lr 1e-4 \
--epochs 50 \
--save_every 5 \
--sample_proportion 1 \
--log_train_metrics_every 10

