#!/usr/bin/env bash

EXPERIMENT_PREFIX="conll_elmo_small"
SCRIPT_FILE="conll_small_elmo.py"

python ${SCRIPT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_no_projection_layer" \
--exp_dir_path  "./_no_projection_layer" \
--model_save_dir "./_no_projection_layer/checkpoints" \
--device cuda:6 \
--dropout 0.5 \
--reg 0 \
--bs 64 \
--emb_type "lample_conll" \
--hidden_dim 100 \
--num_layers=1 \
--bidirectional \
--char_emb_dim 10 \
--char_encoder_hidden_dim 25 \
--lr 1e-3 \
--combine_strategy concat \
--epochs 75 \
--save_every 20 \
--log_train_metrics_every 5 \
--sample_proportion 1