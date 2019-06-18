#!/usr/bin/env bash

cd ../clients

python bow_bert_emb_lc_gensect.py \
--exp_name "debug_bow_bert_emb_gensect" \
--device cpu \
--bert_type "bert-large-cased" \
--max_length 20 \
--max_num_words 15000 \
--debug \
--debug_dataset_proportion 0.01 \
--return_instances \
--bs 32 \
--emb_dim 1024 \
--lr 1e-2 \
--epochs 1 \
--save_every 1 \
--log_train_metrics_every 50