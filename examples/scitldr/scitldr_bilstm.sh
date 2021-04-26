#!/usr/bin/env bash

EXPERIMENT_PREFIX="bilstm_seq2seq"
SCRIPT_FILE="scitldr_bilstm.py"

python ${SCRIPT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_scitldr_aic" \
--model_save_dir "./scitldr_aic" \
--device "cuda:0" \
--bs 8 \
--emb_type "glove_6B_50" \
--hidden_dim 256 \
--bidirectional \
--lr 1e-3 \
--combine_strategy concat \
--epochs 100 \
--save_every 10 \
--log_train_metrics_every 10 \
--sample_proportion 1 \
--pred_max_length 50
