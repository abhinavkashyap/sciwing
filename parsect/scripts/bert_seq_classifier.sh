#!/usr/bin/env bash

cd ../clients

python bert_seq_classifier_client.py \
--exp_name "debug_bert_seq_classifier_gpu" \
--bert_type "bert-base-uncased" \
--device cuda:0 \
--max_num_words 15000 \
--max_len 25 \
--debug \
--debug_dataset_proportion 0.02 \
--bs 32 \
--emb_dim 768 \
--lr 1e-4 \
--epochs 20 \
--save_every 15 \
--log_train_metrics_every 1

python bert_seq_classifier_client.py \
--exp_name "bert_seq_classifier_base_uncased_emb_lc_20e_1e-4lr" \
--bert_type "bert-base-uncased" \
--device cuda:0 \
--max_num_words 15000 \
--max_len 25 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_dim 768 \
--lr 1e-4 \
--epochs 20 \
--save_every 1 \
--log_train_metrics_every 50