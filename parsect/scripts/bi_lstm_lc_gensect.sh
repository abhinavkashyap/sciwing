#!/usr/bin/env bash

cd ../clients

EXPERIMENT_PREFIX="gensect_bi_lstm_lc"
SCRIPT_FILE="bi_lstm_encoder_random_emb_lc_gensect.py"

python ${SCRIPT_FILE} \
--exp_name "debug_"${EXPERIMENT_PREFIX} \
--device cuda:0 \
--max_num_words 1000 \
--max_length 10 \
--debug \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type random \
--emb_dim 300 \
--hidden_dim 512 \
--lr 1e-3 \
--bidirectional \
--combine_strategy concat \
--epochs 30 \
--save_every 1 \
--log_train_metrics_every 50

python ${SCRIPT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_100w_ml5_100d_100h_1e-3lr_30e" \
--device cuda:0 \
--max_num_words 100 \
--max_length 5 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type random \
--emb_dim 100 \
--hidden_dim 100h \
--lr 1e-3 \
--bidirectional \
--combine_strategy concat \
--epochs 30 \
--save_every 5 \
--log_train_metrics_every 50

python ${SCRIPT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_400w_ml5_100d_100h_1e-3lr_30e" \
--device cuda:0 \
--max_num_words 400 \
--max_length 5 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type random \
--emb_dim 100 \
--hidden_dim 100 \
--lr 1e-3 \
--bidirectional \
--combine_strategy concat \
--epochs 30 \
--save_every 5 \
--log_train_metrics_every 50

python ${SCRIPT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_900w_ml5_100d_100h_1e-3lr_30e" \
--device cuda:0 \
--max_num_words 900 \
--max_length 5 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type random \
--emb_dim 100 \
--hidden_dim 100 \
--lr 1e-3 \
--bidirectional \
--combine_strategy concat \
--epochs 30 \
--save_every 5 \
--log_train_metrics_every 50

python ${SCRIPT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_900w_ml5_150d_100h_1e-3lr_30e" \
--device cuda:0 \
--max_num_words 900 \
--max_length 5 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type random \
--emb_dim 150 \
--hidden_dim 100 \
--lr 1e-3 \
--bidirectional \
--combine_strategy concat \
--epochs 30 \
--save_every 5 \
--log_train_metrics_every 50