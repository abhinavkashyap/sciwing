#!/usr/bin/env bash

cd ../clients

EXPERIMENT_PREFIX="lstm_crf_parscit"
SCRIPT_FILE="lstm_crf_client.py"

python ${SCRIPT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_debug" \
--device cpu \
--max_num_words 1000 \
--max_len 10 \
--debug \
--debug_dataset_proportion 0.02 \
--bs 10 \
--emb_type random \
--emb_dim 50 \
--hidden_dim 1024 \
--lr 1e-2 \
--bidirectional \
--combine_strategy concat \
--epochs 10 \
--save_every 50 \
--log_train_metrics_every 5

