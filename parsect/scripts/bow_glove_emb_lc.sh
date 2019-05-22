#!/usr/bin/env bash

cd ../clients

python bow_glove_emb_linear_classifier.py \
--exp_name "bow_glove_emb_lc_3kw_15ml_50d_10e_1e-3lr" \
--max_num_words 3000 \
--max_length 15 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type glove_6B_50 \
--emb_dim 50 \
--lr 1e-4 \
--epochs 10 \
--save_every 1 \
--log_train_metrics_every 50