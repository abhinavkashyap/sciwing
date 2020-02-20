#!/usr/bin/env bash


EXPERIMENT_PREFIX="lstm_crf_scienceie"
SCRIPT_FILE="science_ie.py"

python ${SCRIPT_FILE} \
--exp_name ${EXPERIMENT_PREFIX} \
--exp_dir_path  "./output" \
--model_save_dir "./output/checkpoints" \
--device cpu \
--dropout 0.4 \
--reg 0 \
--bs 32 \
--emb_type "glove_6B_50" \
--char_emb_dim 5 \
--char_encoder_hidden_dim 10 \
--hidden_dim 10 \
--lr 1e-2 \
--combine_strategy concat \
--epochs 1 \
--save_every 50 \
--log_train_metrics_every 5 \
--sample_proportion 1