#!/usr/bin/env bash

EXPERIMENT_PREFIX="lstm_seq2seq"
SCRIPT_FILE="pubmed_summarization_bilstm_seq2seq.py"

python ${SCRIPT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_pubmed" \
--model_save_dir "./pubmed_seq2seq" \
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
--pred_max_length 500
