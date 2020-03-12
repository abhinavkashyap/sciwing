#!/usr/bin/env bash

EXPERIMENT_PREFIX="conll_seq_crf"
SCRIPT_FILE="conll_seq_crf.py"

python ${SCRIPT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"two_layer_rnn" \
--exp_dir_path  "./two_layer_rnn" \
--model_save_dir "./two_layer_rnn/checkpoints" \
--device cuda:2 \
--dropout 0.5 \
--reg 0 \
--bs 64 \
--emb_type "glove_6B_100" \
--hidden_dim 200 \
--num_layers=2 \
--bidirectional \
--add_projection_layer \
--char_emb_dim 10 \
--char_encoder_hidden_dim 25 \
--lr 1e-3 \
--combine_strategy concat \
--epochs 75 \
--save_every 20 \
--log_train_metrics_every 5 \
--sample_proportion 1