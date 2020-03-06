#!/usr/bin/env bash

EXPERIMENT_PREFIX="conll_seq_crf"
SCRIPT_FILE="conll_seq_crf.py"

python ${SCRIPT_FILE} \
--exp_name ${EXPERIMENT_PREFIX} \
--exp_dir_path  "./output" \
--model_save_dir "./output/checkpoints" \
--device cuda:2 \
--dropout 0.0 \
--reg 0 \
--bs 128 \
--emb_type "glove_6B_100" \
--char_emb_dim 20 \
--char_encoder_hidden_dim 25 \
--hidden_dim 350 \
--bidirectional \
--num_layers=2 \
--lr 1e-3 \
--combine_strategy concat \
--epochs 25 \
--save_every 20 \
--log_train_metrics_every 3 \
--sample_proportion 0.1