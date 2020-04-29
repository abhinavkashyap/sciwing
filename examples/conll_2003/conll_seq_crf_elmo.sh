#!/usr/bin/env bash

EXPERIMENT_PREFIX="conll_seq_crf"
SCRIPT_FILE="conll_seq_crf_elmo.py"

python ${SCRIPT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_lample_elmo_seed5" \
--exp_dir_path  "./_lample_elmo_seed5" \
--model_save_dir "./_lample_elmo_seed5/checkpoints" \
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