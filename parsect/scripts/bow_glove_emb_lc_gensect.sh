#!/usr/bin/env bash

cd ../clients

EXPERIMENT_PREFIX="gensect_bow_glove_emb_lc"

python bow_glove_emb_lc_generic_sect.py \
--exp_name "debug_"${EXPERIMENT_PREFIX} \
--max_num_words 3000 \
--max_length 10 \
--debug \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type glove_6B_50 \
--emb_dim 50 \
--lr 1e-3 \
--epochs 40 \
--save_every 5 \
--log_train_metrics_every 50

python bow_glove_emb_lc_generic_sect.py \
--exp_name ${EXPERIMENT_PREFIX}"_300w_ml5_50d_40e" \
--max_num_words 300 \
--max_length 5 \
--debug \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type glove_6B_50 \
--emb_dim 50 \
--lr 1e-3 \
--epochs 40 \
--save_every 5 \
--log_train_metrics_every 50

python bow_glove_emb_lc_generic_sect.py \
--exp_name ${EXPERIMENT_PREFIX}"_400w_ml5_50d_40e" \
--max_num_words 400 \
--max_length 5 \
--debug \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type glove_6B_50 \
--emb_dim 50 \
--lr 1e-3 \
--epochs 40 \
--save_every 5 \
--log_train_metrics_every 50

python bow_glove_emb_lc_generic_sect.py \
--exp_name ${EXPERIMENT_PREFIX}"_600w_ml5_50d_40e" \
--max_num_words 400 \
--max_length 5 \
--debug \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type glove_6B_50 \
--emb_dim 50 \
--lr 1e-3 \
--epochs 40 \
--save_every 5 \
--log_train_metrics_every 50

python bow_glove_emb_lc_generic_sect.py \
--exp_name ${EXPERIMENT_PREFIX}"_900w_ml5_50d_40e" \
--max_num_words 900 \
--max_length 5 \
--debug \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type glove_6B_50 \
--emb_dim 50 \
--lr 1e-3 \
--epochs 40 \
--save_every 5 \
--log_train_metrics_every 50

python bow_glove_emb_lc_generic_sect.py \
--exp_name ${EXPERIMENT_PREFIX}"_300w_ml5_100d_40e" \
--max_num_words 300 \
--max_length 5 \
--debug \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type glove_6B_100 \
--emb_dim 100 \
--lr 1e-3 \
--epochs 40 \
--save_every 5 \
--log_train_metrics_every 50

python bow_glove_emb_lc_generic_sect.py \
--exp_name ${EXPERIMENT_PREFIX}"_600w_ml5_100d_40e" \
--max_num_words 600 \
--max_length 5 \
--debug \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type glove_6B_100 \
--emb_dim 100 \
--lr 1e-3 \
--epochs 40 \
--save_every 5 \
--log_train_metrics_every 50

python bow_glove_emb_lc_generic_sect.py \
--exp_name ${EXPERIMENT_PREFIX}"_600w_ml5_100d_40e" \
--max_num_words 600 \
--max_length 5 \
--debug \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type glove_6B_100 \
--emb_dim 100 \
--lr 1e-3 \
--epochs 40 \
--save_every 5 \
--log_train_metrics_every 50


python bow_glove_emb_lc_generic_sect.py \
--exp_name ${EXPERIMENT_PREFIX}"_900w_ml5_100d_40e" \
--max_num_words 900 \
--max_length 5 \
--debug \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type glove_6B_100 \
--emb_dim 100 \
--lr 1e-3 \
--epochs 40 \
--save_every 5 \
--log_train_metrics_every 50


python bow_glove_emb_lc_generic_sect.py \
--exp_name ${EXPERIMENT_PREFIX}"_900w_ml5_200d_40e" \
--max_num_words 900 \
--max_length 5 \
--debug \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type glove_6B_200 \
--emb_dim 200 \
--lr 1e-3 \
--epochs 40 \
--save_every 5 \
--log_train_metrics_every 50