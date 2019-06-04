#!/usr/bin/env bash

cd ../clients

python bow_bert_emb_linear_classifier.py \
--exp_name "bow_bert_base_cased_emb_lc_10e_1e-2lr" \
--device cuda:0 \
--bert_type "bert-large-cased" \
--debug_dataset_proportion 0.01 \
--return_instances \
--bs 32 \
--emb_dim 1024 \
--lr 1e-2 \
--epochs 10 \
--save_every 1 \
--log_train_metrics_every 50