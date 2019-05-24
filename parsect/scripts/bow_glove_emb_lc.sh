#!/usr/bin/env bash

cd ../clients

python bow_glove_emb_linear_classifier.py \
--exp_name "bow_glove_emb_lc_3kw_10ml_50d_10e_1e-3lr" \
--max_num_words 3000 \
--max_length 10 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type glove_6B_50 \
--emb_dim 50 \
--lr 1e-3 \
--epochs 10 \
--save_every 1 \
--log_train_metrics_every 50

python bow_glove_emb_linear_classifier.py \
--exp_name "bow_glove_emb_lc_3kw_10ml_50d_15e_1e-3lr" \
--max_num_words 3000 \
--max_length 10 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type glove_6B_50 \
--emb_dim 50 \
--lr 1e-3 \
--epochs 15 \
--save_every 1 \
--log_train_metrics_every 50

python bow_glove_emb_linear_classifier.py \
--exp_name "bow_glove_emb_lc_3kw_10ml_50d_50e_1e-4lr" \
--max_num_words 3000 \
--max_length 10 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type glove_6B_50 \
--emb_dim 50 \
--lr 1e-4 \
--epochs 50 \
--save_every 1 \
--log_train_metrics_every 50

python bow_glove_emb_linear_classifier.py \
--exp_name "bow_glove_emb_lc_4kw_10ml_50d_50e_1e-4lr" \
--max_num_words 4000 \
--max_length 10 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type glove_6B_50 \
--emb_dim 50 \
--lr 1e-4 \
--epochs 50 \
--save_every 1 \
--log_train_metrics_every 50

python bow_glove_emb_linear_classifier.py \
--exp_name "bow_glove_emb_lc_4kw_10ml_50d_50e_1e-4lr" \
--max_num_words 4000 \
--max_length 10 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type glove_6B_50 \
--emb_dim 50 \
--lr 1e-4 \
--epochs 50 \
--save_every 1 \
--log_train_metrics_every 50

python bow_glove_emb_linear_classifier.py \
--exp_name "bow_glove_emb_lc_3kw_10ml_100d_50e_1e-4lr" \
--max_num_words 3000 \
--max_length 10 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type glove_6B_100 \
--emb_dim 100 \
--lr 1e-4 \
--epochs 50 \
--save_every 1 \
--log_train_metrics_every 50

python bow_glove_emb_linear_classifier.py \
--exp_name "bow_glove_emb_lc_4kw_10ml_100d_50e_1e-4lr" \
--max_num_words 4000 \
--max_length 10 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type glove_6B_100 \
--emb_dim 100 \
--lr 1e-4 \
--epochs 50 \
--save_every 1 \
--log_train_metrics_every 50

python bow_glove_emb_linear_classifier.py \
--exp_name "bow_glove_emb_lc_4kw_15ml_100d_50e_1e-4lr" \
--max_num_words 4000 \
--max_length 15 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type glove_6B_100 \
--emb_dim 100 \
--lr 1e-4 \
--epochs 50 \
--save_every 1 \
--log_train_metrics_every 50

python bow_glove_emb_linear_classifier.py \
--exp_name "bow_glove_emb_lc_5kw_10ml_100d_50e_1e-4lr" \
--max_num_words 5000 \
--max_length 10 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type glove_6B_100 \
--emb_dim 100 \
--lr 1e-4 \
--epochs 50 \
--save_every 1 \
--log_train_metrics_every 50

python bow_glove_emb_linear_classifier.py \
--exp_name "bow_glove_emb_lc_5kw_15ml_100d_50e_1e-4lr" \
--max_num_words 5000 \
--max_length 15 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type glove_6B_100 \
--emb_dim 100 \
--lr 1e-4 \
--epochs 50 \
--save_every 1 \
--log_train_metrics_every 50

python bow_glove_emb_linear_classifier.py \
--exp_name "bow_glove_emb_lc_3kw_10ml_200d_50e_1e-4lr" \
--max_num_words 3000 \
--max_length 10 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type glove_6B_200 \
--emb_dim 200 \
--lr 1e-4 \
--epochs 50 \
--save_every 1 \
--log_train_metrics_every 50

python bow_glove_emb_linear_classifier.py \
--exp_name "bow_glove_emb_lc_4kw_10ml_200d_50e_1e-4lr" \
--max_num_words 4000 \
--max_length 10 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type glove_6B_200 \
--emb_dim 200 \
--lr 1e-4 \
--epochs 50 \
--save_every 1 \
--log_train_metrics_every 50

python bow_glove_emb_linear_classifier.py \
--exp_name "bow_glove_emb_lc_4kw_15ml_200d_50e_1e-4lr" \
--max_num_words 4000 \
--max_length 15 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type glove_6B_200 \
--emb_dim 200 \
--lr 1e-4 \
--epochs 50 \
--save_every 1 \
--log_train_metrics_every 50

python bow_glove_emb_linear_classifier.py \
--exp_name "bow_glove_emb_lc_5kw_10ml_200d_50e_1e-4lr" \
--max_num_words 5000 \
--max_length 10 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type glove_6B_200 \
--emb_dim 200 \
--lr 1e-4 \
--epochs 50 \
--save_every 1 \
--log_train_metrics_every 50

python bow_glove_emb_linear_classifier.py \
--exp_name "bow_glove_emb_lc_5kw_15ml_200d_50e_1e-4lr" \
--max_num_words 5000 \
--max_length 15 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type glove_6B_200 \
--emb_dim 200 \
--lr 1e-4 \
--epochs 50 \
--save_every 1 \
--log_train_metrics_every 50

python bow_glove_emb_linear_classifier.py \
--exp_name "bow_glove_emb_lc_3kw_10ml_300d_50e_1e-4lr" \
--max_num_words 3000 \
--max_length 10 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type glove_6B_300 \
--emb_dim 300 \
--lr 1e-4 \
--epochs 50 \
--save_every 1 \
--log_train_metrics_every 50

python bow_glove_emb_linear_classifier.py \
--exp_name "bow_glove_emb_lc_4kw_10ml_300d_50e_1e-4lr" \
--max_num_words 4000 \
--max_length 10 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type glove_6B_300 \
--emb_dim 300 \
--lr 1e-4 \
--epochs 50 \
--save_every 1 \
--log_train_metrics_every 50

python bow_glove_emb_linear_classifier.py \
--exp_name "bow_glove_emb_lc_4kw_15ml_300d_50e_1e-4lr" \
--max_num_words 4000 \
--max_length 15 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type glove_6B_300 \
--emb_dim 300 \
--lr 1e-4 \
--epochs 50 \
--save_every 1 \
--log_train_metrics_every 50

python bow_glove_emb_linear_classifier.py \
--exp_name "bow_glove_emb_lc_5kw_10ml_300d_50e_1e-4lr" \
--max_num_words 5000 \
--max_length 10 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type glove_6B_300 \
--emb_dim 300 \
--lr 1e-4 \
--epochs 50 \
--save_every 1 \
--log_train_metrics_every 50

python bow_glove_emb_linear_classifier.py \
--exp_name "bow_glove_emb_lc_5kw_15ml_300d_50e_1e-4lr" \
--max_num_words 5000 \
--max_length 15 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type glove_6B_300 \
--emb_dim 300 \
--lr 1e-4 \
--epochs 50 \
--save_every 1 \
--log_train_metrics_every 50