#!/usr/bin/env bash

EXPERIMENT_PREFIX="lstm_crf_parscit"
SCRIPT_FILE="parscit_client.py"


python ${SCRIPT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_char_enc_10kw_ml75_mcl15_500d_25cd_256h_50charench_1e-3lr_bidir_concat_20e" \
--device cuda:0 \
--max_num_words 10000 \
--max_len 75 \
--max_char_len 15 \
--debug_dataset_proportion 0.02 \
--bs 10 \
--emb_type parscit \
--emb_dim 500 \
--char_emb_dim 25 \
--hidden_dim 256 \
--lr 1e-3 \
--bidirectional \
--use_char_encoder \
--dropout 0.4 \
--char_encoder_hidden_dim 50 \
--combine_strategy concat \
--epochs 20 \
--save_every 10 \
--log_train_metrics_every 10

