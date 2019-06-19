#!/usr/bin/env bash

cd ../clients

python bow_random_emb_lc_parsect.py \
--exp_name "debug_bow_random" \
--max_num_words 3000 \
--max_length 15 \
--debug \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_dim 50 \
--lr 1e-4 \
--epochs 1 \
--save_every 1 \
--log_train_metrics_every 50

python bow_random_emb_lc_parsect.py \
--exp_name "bow_random_emb_lc_3kw_15ml_50d_10e_1e-3lr" \
--max_num_words 3000 \
--max_length 15 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_dim 50 \
--lr 1e-4 \
--epochs 10 \
--save_every 1 \
--log_train_metrics_every 50

python bow_random_emb_lc_parsect.py \
--exp_name "bow_random_emb_lc_3kw_15ml_50d_15e_1e-3lr" \
--max_num_words 3000 \
--max_length 15 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_dim 50 \
--lr 1e-4 \
--epochs 15 \
--save_every 1 \
--log_train_metrics_every 50

python bow_random_emb_lc_parsect.py \
--exp_name "bow_random_emb_lc_3kw_15ml_50d_20e_1e-3lr" \
--max_num_words 3000 \
--max_length 15 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_dim 50 \
--lr 1e-4 \
--epochs 20 \
--save_every 1 \
--log_train_metrics_every 50

python bow_random_emb_lc_parsect.py \
--exp_name "bow_random_emb_lc_3kw_15ml_50d_50e_1e-4lr" \
--max_num_words 3000 \
--max_length 15 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_dim 50 \
--lr 1e-4 \
--epochs 50 \
--save_every 1 \
--log_train_metrics_every 50

python bow_random_emb_lc_parsect.py \
--exp_name "bow_random_emb_lc_3kw_15ml_75d_50e_1e-4lr" \
--max_num_words 3000 \
--max_length 15 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_dim 75 \
--lr 1e-4 \
--epochs 50 \
--save_every 1 \
--log_train_metrics_every 50

python bow_random_emb_lc_parsect.py \
--exp_name "bow_random_emb_lc_4kw_15ml_50d_50e_1e-4lr" \
--max_num_words 4000 \
--max_length 15 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_dim 50 \
--lr 1e-4 \
--epochs 50 \
--save_every 1 \
--log_train_metrics_every 50

python bow_random_emb_lc_parsect.py \
--exp_name "bow_random_emb_lc_4kw_10ml_75d_50e_1e-4lr" \
--max_num_words 4000 \
--max_length 10 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_dim 75 \
--lr 1e-4 \
--epochs 50 \
--save_every 1 \
--log_train_metrics_every 50

python bow_random_emb_lc_parsect.py \
--exp_name "bow_random_emb_lc_4kw_15ml_75d_50e_1e-4lr" \
--max_num_words 4000 \
--max_length 15 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_dim 75 \
--lr 1e-4 \
--epochs 50 \
--save_every 1 \
--log_train_metrics_every 50