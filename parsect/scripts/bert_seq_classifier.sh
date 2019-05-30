#!/usr/bin/env bash

cd ../clients

python bert_seq_classifier_client.py \
--exp_name "debug_bert_seq_classifier_base_cased_emb_lc_10e_1e-2lr" \
--bert_type "bert-large-cased" \
--debug \
--debug_dataset_proportion 0.01 \
--return_instances \
--bs 32 \
--emb_dim 1024 \
--lr 1e-2 \
--epochs 10 \
--save_every 1 \
--log_train_metrics_every 50