#!/usr/bin/env bash

cd ../clients

python bi_lstm_encoder_random_emb_linear_classifier.py \
--exp_name "debug_bi_lstm_lc" \
--max_num_words 1000 \
--max_length 10 \
--debug \
--debug_dataset_proportion 0.01 \
--bs 10 \
--emb_type random \
--emb_dim 300 \
--hidden_dim 512 \
--lr 1e-3 \
--bidirectional \
--combine_strategy concat \
--epochs 40 \
--save_every 1 \
--log_train_metrics_every 5
