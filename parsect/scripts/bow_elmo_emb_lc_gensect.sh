#!/usr/bin/env bash

cd ../clients

python bow_elmo_emb_lc_gensect.py \
--exp_name "debug_bow_elmo_gensect" \
--max_length 15 \
--max_num_words 15000 \
--device cpu \
--debug \
--layer_aggregation last \
--word_aggregation sum \
--debug_dataset_proportion 0.01 \
--bs 10 \
--emb_dim 1024 \
--lr 1e-4 \
--epochs 1 \
--save_every 1 \
--log_train_metrics_every 10

python bow_elmo_emb_lc_gensect.py \
--exp_name "bow_elmo_gensect_ml5_layeragg_last_wordagg_sum_20e_1e-4lr" \
--max_length 5 \
--max_num_words 15000 \
--device cuda:0 \
--layer_aggregation last \
--word_aggregation sum \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_dim 1024 \
--lr 1e-4 \
--epochs 20 \
--save_every 1 \
--log_train_metrics_every 10

python bow_elmo_emb_lc_gensect.py \
--exp_name "bow_elmo_gensect_ml5_layeragg_first_word_agg_average_20e_1e-4lr" \
--max_length 5 \
--max_num_words 15000 \
--device cuda:0 \
--layer_aggregation first \
--word_aggregation average \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_dim 1024 \
--lr 1e-4 \
--epochs 20 \
--save_every 1 \
--log_train_metrics_every 10
