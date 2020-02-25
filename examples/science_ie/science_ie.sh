#!/usr/bin/env bash


EXPERIMENT_PREFIX="lstm_crf_scienceie"
SCRIPT_FILE="science_ie.py"

python ${SCRIPT_FILE} \
--exp_name ${EXPERIMENT_PREFIX} \
--exp_dir_path  "./output" \
--model_save_dir "./output/checkpoints" \
--device cuda:2 \
--dropout 0.5 \
--reg 0 \
--bs 64 \
--emb_type "glove_6B_50" \
--char_emb_dim 25 \
--char_encoder_hidden_dim 25 \
--hidden_dim 128 \
--lr 1e-2 \
--bidirectional \
--combine_strategy concat \
--epochs 50 \
--save_every 50 \
--log_train_metrics_every 5 \
--sample_proportion 1