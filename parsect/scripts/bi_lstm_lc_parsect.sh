#!/usr/bin/env bash

cd ../clients

EXPERIMENT_PREFIX="parsect_bi_lstm_lc"
SCRIPT_FILE="bi_lstm_encoder_random_emb_lc_parsect.py"


python ${SCRIPT_FILE} \
--exp_name "debug_"${EXPERIMENT_PREFIX} \
--device cpu \
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
--epochs 1 \
--save_every 5 \
--log_train_metrics_every 50


python ${SCRIPT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_1kw_ml10_300drandom_512h_1e-3lr_bidir_concat_40e" \
--device cuda:0 \
--max_num_words 1000 \
--max_length 32 \
--debug_dataset_proportion 0.01 \
--bs 10 \
--emb_type random \
--emb_dim 300 \
--hidden_dim 512 \
--lr 1e-3 \
--bidirectional \
--combine_strategy concat \
--epochs 40 \
--save_every 5 \
--log_train_metrics_every 50

python ${SCRIPT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_2kw_ml10_300drandom_512h_1e-3lr_bidir_concat_40e" \
--device cuda:0 \
--max_num_words 2000 \
--max_length 10 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type random \
--emb_dim 300 \
--hidden_dim 512 \
--lr 1e-3 \
--bidirectional \
--combine_strategy concat \
--epochs 40 \
--save_every 5 \
--log_train_metrics_every 50


python ${SCRIPT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_5kw_ml10_300drandom_512h_1e-3lr_bidir_concat_40e" \
--device cuda:0 \
--max_num_words 5000 \
--max_length 10 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type random \
--emb_dim 300 \
--hidden_dim 512 \
--lr 1e-3 \
--bidirectional \
--combine_strategy concat \
--epochs 40 \
--save_every 5 \
--log_train_metrics_every 50

python ${SCRIPT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_8kw_ml10_300drandom_512h_1e-3lr_bidir_concat_40e" \
--device cuda:0 \
--max_num_words 8000 \
--max_length 10 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type random \
--emb_dim 300 \
--hidden_dim 512 \
--lr 1e-3 \
--bidirectional \
--combine_strategy concat \
--epochs 40 \
--save_every 5 \
--log_train_metrics_every 50

python bi_lstm_encoder_random_emb_lc_parsect.py \
--exp_name ${EXPERIMENT_PREFIX}"_5kw_ml15_300drandom_512h_1e-3lr_bidir_concat_40e" \
--device cuda:0 \
--max_num_words 5000 \
--max_length 15 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type random \
--emb_dim 300 \
--hidden_dim 512 \
--lr 1e-3 \
--bidirectional \
--combine_strategy concat \
--epochs 40 \
--save_every 1 \
--log_train_metrics_every 50

python ${SCRIPT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_5kw_ml20_300drandom_512h_1e-3lr_bidir_concat_40e" \
--device cuda:0 \
--max_num_words 5000 \
--max_length 20 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type random \
--emb_dim 300 \
--hidden_dim 512 \
--lr 1e-3 \
--bidirectional \
--combine_strategy concat \
--epochs 40 \
--save_every 5 \
--log_train_metrics_every 50


python ${SCRIPT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_5kw_ml20_150drandom_512h_1e-3lr_bidir_concat_40e" \
--device cuda:0 \
--max_num_words 5000 \
--max_length 20 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type random \
--emb_dim 150 \
--hidden_dim 512 \
--lr 1e-3 \
--bidirectional \
--combine_strategy concat \
--epochs 40 \
--save_every 5 \
--log_train_metrics_every 50