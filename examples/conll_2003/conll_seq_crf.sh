#!/usr/bin/env bash

EXPERIMENT_PREFIX="conll_seq_crf"
SCRIPT_FILE="conll_seq_crf.py"

python ${SCRIPT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_co-4lr" \
--exp_dir_path  "./output_conll_elmo_1e-4lr" \
--model_save_dir "./output_conll_elmo_1e-4lr/checkpoints" \
--device cuda:7 \
--dropout 0.6 \
--reg 0 \
--bs 64 \
--emb_type "glove_6B_300" \
--hidden_dim 300 \
--num_layers=1 \
--bidirectional \
--char_emb_dim 10 \
--char_encoder_hidden_dim 25 \
--lr 1e-3 \
--combine_strategy concat \
--epochs 30 \
--save_every 20 \
--log_train_metrics_every 5 \
--sample_proportion 1