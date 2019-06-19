#!/usr/bin/env bash

cd ../clients

python bert_seq_classifier_client.py \
--exp_name "debug_bert_seq_classifier_gpu" \
--bert_type "bert-large-cased" \
--device cuda \
--max_num_words 15000 \
--max_len 20 \
--debug \
--debug_dataset_proportion 0.3 \
--bs 32 \
--emb_dim 1024 \
--lr 1e-2 \
--epochs 1 \
--save_every 1 \
--log_train_metrics_every 50

#python bert_seq_classifier_client.py \
#--exp_name "bert_seq_classifier_base_cased_emb_lc_10e_1e-2lr" \
#--bert_type "bert-base-cased" \
#--device cuda:0 \
#--max_num_words 15000 \
#--max_len 20 \
#--debug_dataset_proportion 0.01 \
#--bs 32 \
#--emb_dim 768 \
#--lr 1e-2 \
#--epochs 1 \
#--save_every 1 \
#--log_train_metrics_every 50