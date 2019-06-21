#!/usr/bin/env bash

cd ../clients

EXPERIMENT_PREFIX="gensect_bow_random_emb_lc"
python bow_random_emb_lc_generic_sect.py \
--exp_name "debug_"${EXPERIMENT_PREFIX} \
--max_num_words 400 \
--max_length 5 \
--debug \
--debug_dataset_proportion 0.1 \
--bs 32 \
--emb_dim 200 \
--lr 1e-2 \
--epochs 40 \
--save_every 1 \
--log_train_metrics_every 50

python bow_random_emb_lc_generic_sect.py \
--exp_name ${EXPERIMENT_PREFIX}"_300w_ml5_200d_40e" \
--max_num_words 300 \
--max_length 5 \
--debug_dataset_proportion 0.1 \
--bs 32 \
--emb_dim 200 \
--lr 1e-2 \
--epochs 40 \
--save_every 1 \
--log_train_metrics_every 5

python bow_random_emb_lc_generic_sect.py \
--exp_name ${EXPERIMENT_PREFIX}"_300w_ml5_200d_40e" \
--max_num_words 300 \
--max_length 5 \
--debug_dataset_proportion 0.1 \
--bs 32 \
--emb_dim 200 \
--lr 1e-2 \
--epochs 40 \
--save_every 1 \
--log_train_metrics_every 50

python bow_random_emb_lc_generic_sect.py \
--exp_name ${EXPERIMENT_PREFIX}"_400w_ml5_100d_40e" \
--max_num_words 400 \
--max_length 5 \
--debug_dataset_proportion 0.1 \
--bs 32 \
--emb_dim 100 \
--lr 1e-2 \
--epochs 40 \
--save_every 1 \
--log_train_metrics_every 50

python bow_random_emb_lc_generic_sect.py \
--exp_name ${EXPERIMENT_PREFIX}"_400w_ml5_200d_40e" \
--max_num_words 400 \
--max_length 5 \
--debug_dataset_proportion 0.1 \
--bs 32 \
--emb_dim 200 \
--lr 1e-2 \
--epochs 40 \
--save_every 1 \
--log_train_metrics_every 50

python bow_random_emb_lc_generic_sect.py \
--exp_name ${EXPERIMENT_PREFIX}"_700w_ml5_200d_40e" \
--max_num_words 700 \
--max_length 5 \
--debug_dataset_proportion 0.1 \
--bs 32 \
--emb_dim 200 \
--lr 1e-2 \
--epochs 40 \
--save_every 1 \
--log_train_metrics_every 50

python bow_random_emb_lc_generic_sect.py \
--exp_name ${EXPERIMENT_PREFIX}"_900w_ml5_200d_40e" \
--max_num_words 900 \
--max_length 5 \
--debug_dataset_proportion 0.1 \
--bs 32 \
--emb_dim 200 \
--lr 1e-2 \
--epochs 40 \
--save_every 1 \
--log_train_metrics_every 50

python bow_random_emb_lc_generic_sect.py \
--exp_name ${EXPERIMENT_PREFIX}"_1000w_ml5_200d_40e" \
--max_num_words 1000 \
--max_length 5 \
--debug_dataset_proportion 0.1 \
--bs 32 \
--emb_dim 200 \
--lr 1e-2 \
--epochs 40 \
--save_every 1 \
--log_train_metrics_every 50