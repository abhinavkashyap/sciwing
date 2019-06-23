#!/usr/bin/env bash

cd ../clients

EXPERIMENT_PREFIX="parsect_bow_glove_emb_lc"
CLIENT_FILE="bow_glove_emb_lc_parsect.py"

python ${CLIENT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_lc_3kw_10ml_50d_10e_1e-3lr" \
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

python ${CLIENT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_lc_3kw_10ml_50d_15e_1e-3lr" \
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

python ${CLIENT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_lc_3kw_10ml_50d_50e_1e-4lr" \
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

python ${CLIENT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_4kw_10ml_50d_50e_1e-4lr" \
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

python ${CLIENT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_4kw_10ml_50d_50e_1e-4lr" \
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

python ${CLIENT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_3kw_10ml_100d_50e_1e-4lr" \
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

python ${CLIENT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_4kw_10ml_100d_50e_1e-4lr" \
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

python ${CLIENT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_4kw_15ml_100d_50e_1e-4lr" \
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

python ${CLIENT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_5kw_10ml_100d_50e_1e-4lr" \
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

python ${CLIENT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_5kw_15ml_100d_50e_1e-4lr" \
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

python ${CLIENT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_lc_3kw_10ml_200d_50e_1e-4lr" \
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

python ${CLIENT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_lc_4kw_10ml_200d_50e_1e-4lr" \
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

python ${CLIENT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_lc_4kw_15ml_200d_50e_1e-4lr" \
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

python ${CLIENT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_5kw_10ml_200d_50e_1e-4lr" \
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

python ${CLIENT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_5kw_15ml_200d_50e_1e-4lr" \
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

python ${CLIENT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_3kw_10ml_300d_50e_1e-4lr" \
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

python ${CLIENT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_4kw_10ml_300d_50e_1e-4lr" \
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

python ${CLIENT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_4kw_15ml_300d_50e_1e-4lr" \
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

python ${CLIENT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_5kw_10ml_300d_50e_1e-4lr" \
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

python ${CLIENT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_5kw_15ml_300d_50e_1e-4lr" \
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