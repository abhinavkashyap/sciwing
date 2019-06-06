#!/usr/bin/env bash

cd ../clients

python bow_elmo_emb_linear_classifier.py \
--exp_name "debug_bow_elmo_emb_lc_70e_1e-4lr" \
--max_length 15 \
--device cuda:0 \
--debug \
--layer_aggregation last \
--word_aggregation sum \
--debug_dataset_proportion 0.01 \
--bs 10 \
--emb_dim 1024 \
--lr 1e-4 \
--epochs 70 \
--save_every 1 \
--log_train_metrics_every 10

python bow_elmo_emb_linear_classifier.py \
--exp_name "bow_elmo_emb_lc_layeragg_last_wordagg_sum_10e_1e-3lr" \
--max_length 15 \
--device cuda:0 \
--layer_aggregation last \
--word_aggregation sum \
--debug_dataset_proportion 0.01 \
--bs 10 \
--emb_dim 1024 \
--lr 1e-4 \
--epochs 10 \
--save_every 1 \
--log_train_metrics_every 10

python bow_elmo_emb_linear_classifier.py \
--exp_name "bow_elmo_emb_lc_layeragg_last_wordagg_sum_40e_1e-4lr" \
--max_length 15 \
--device cuda:0 \
--layer_aggregation last \
--word_aggregation sum \
--debug_dataset_proportion 0.01 \
--bs 10 \
--emb_dim 1024 \
--lr 1e-4 \
--epochs 40 \
--save_every 1 \
--log_train_metrics_every 10


python bow_elmo_emb_linear_classifier.py \
--exp_name "bow_elmo_emb_lc_layeragg_first_wordagg_sum_20e_1e-4lr" \
--max_length 15 \
--device cuda:0 \
--layer_aggregation first \
--word_aggregation sum \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_dim 1024 \
--lr 1e-4 \
--epochs 20 \
--save_every 1 \
--log_train_metrics_every 50

python bow_elmo_emb_linear_classifier.py \
--exp_name "bow_elmo_emb_lc_layeragg_first_wordagg_sum_40e_1e-4lr" \
--max_length 15 \
--device cuda:0 \
--layer_aggregation first \
--word_aggregation sum \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_dim 1024 \
--lr 1e-4 \
--epochs 40 \
--save_every 1 \
--log_train_metrics_every 50


python bow_elmo_emb_linear_classifier.py \
--exp_name "bow_elmo_emb_lc_layeragg_first_wordagg_average_40e_1e-4lr" \
--max_length 15 \
--device cuda:0 \
--layer_aggregation first \
--word_aggregation average \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_dim 1024 \
--lr 1e-4 \
--epochs 40 \
--save_every 1 \
--log_train_metrics_every 50


python bow_elmo_emb_linear_classifier.py \
--exp_name "bow_elmo_emb_lc_layeragg_average_wordagg_sum_40e_1e-4lr" \
--max_length 15 \
--device cuda:0 \
--layer_aggregation average \
--word_aggregation sum \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_dim 1024 \
--lr 1e-4 \
--epochs 40 \
--save_every 1 \
--log_train_metrics_every 50
