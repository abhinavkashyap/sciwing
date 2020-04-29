#!/usr/bin/env bash

EXPERIMENT_PREFIX="lstm_crf_parscit"
SCRIPT_FILE="i2b2.py"

python ${SCRIPT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_i2b2" \
--model_save_dir "./i2b2" \
--device "cuda:2" \
--bs 64 \
--emb_type "glove_6B_100" \
--hidden_dim 256 \
--bidirectional \
--lr 1e-3 \
--char_emb_dim 25 \
--char_encoder_hidden_dim 50 \
--combine_strategy concat \
--epochs 100 \
--save_every 10 \
--log_train_metrics_every 10 \
--sample_proportion 1
