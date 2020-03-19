#!/usr/bin/env bash

EXPERIMENT_PREFIX="conll_seq_crf"
SCRIPT_FILE="conll_seq_crf_flair.py"

python ${SCRIPT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_glove_flair_sgd" \
--exp_dir_path  "./_glove_flair_sgd" \
--model_save_dir "./_glove_flair_sgd/checkpoints" \
--device cuda:0 \
--dropout 0.5 \
--reg 0 \
--bs 20 \
--emb_type "glove_6B_300" \
--flair_type "news" \
--hidden_dim 256 \
--num_layers=1 \
--bidirectional \
--char_emb_dim 10 \
--char_encoder_hidden_dim 25 \
--lr 1e-2 \
--combine_strategy concat \
--epochs 75 \
--save_every 20 \
--log_train_metrics_every 20  \
--sample_proportion 1