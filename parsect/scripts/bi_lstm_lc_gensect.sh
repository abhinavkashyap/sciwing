#!/usr/bin/env bash

cd ../clients

python bi_lstm_encoder_random_emb_lc_gensect.py \
--exp_name "debug_bi_lstm_lc_gensect" \
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
--epochs 30 \
--save_every 1 \
--log_train_metrics_every 50