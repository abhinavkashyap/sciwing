#!/usr/bin/env bash

EXPERIMENT_PREFIX="conll_small_elmo"
SCRIPT_FILE="conll_small_elmo.py"

python ${SCRIPT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_conll_small_elmo" \
--exp_dir_path  "./_conll_small_elmo" \
--model_save_dir "./_conll_small_elmo/checkpoints" \
--device cuda:3 \
--dropout 0.5 \
--reg 0 \
--bs 64 \
--emb_type "lample_conll" \
--hidden_dim 100 \
--add_projection_layer \
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