#!/usr/bin/env bash

cd ../clients

EXPERIMENT_PREFIX="parsect_bow_bert"

python bow_bert_emb_lc_parsect.py \
--exp_name "debug_"${EXPERIMENT_PREFIX} \
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

python bow_bert_emb_lc_parsect.py \
--exp_name ${EXPERIMENT_PREFIX}"_base_cased_emb_lc_10e_1e-2lr" \
--device cuda:0 \
--max_length 20 \
--max_num_words 15000 \
--bert_type "bert-large-cased" \
--debug_dataset_proportion 0.01 \
--return_instances \
--bs 32 \
--emb_dim 1024 \
--lr 1e-2 \
--epochs 10 \
--save_every 1 \
--log_train_metrics_every 50

python bow_bert_emb_lc_parsect.py \
--exp_name ${EXPERIMENT_PREFIX}"_large_cased_emb_lc_20e_1e-2lr" \
--device cuda:0 \
--max_length 20 \
--max_num_words 15000 \
--bert_type "bert-large-cased" \
--debug_dataset_proportion 0.01 \
--return_instances \
--bs 32 \
--emb_dim 1024 \
--lr 1e-2 \
--epochs 20 \
--save_every 1 \
--log_train_metrics_every 50

python bow_bert_emb_lc_parsect.py \
--exp_name ${EXPERIMENT_PREFIX}"_large_cased_emb_lc_40e_1e-3lr" \
--device cuda:0 \
--max_length 20 \
--max_num_words 15000 \
--bert_type "bert-large-cased" \
--debug_dataset_proportion 0.01 \
--return_instances \
--bs 32 \
--emb_dim 1024 \
--lr 1e-3 \
--epochs 40 \
--save_every 1 \
--log_train_metrics_every 50

# scibert experiments
python bow_bert_emb_lc_parsect.py \
--exp_name ${EXPERIMENT_PREFIX}"_sci_cased_emb_lc_20e_1e-2lr" \
--device cuda:0 \
--max_length 20 \
--max_num_words 15000 \
--bert_type "scibert-sci-cased" \
--debug_dataset_proportion 0.01 \
--return_instances \
--bs 32 \
--emb_dim 768 \
--lr 1e-2 \
--epochs 20 \
--save_every 1 \
--log_train_metrics_every 50

python bow_bert_emb_lc_parsect.py \
--exp_name ${EXPERIMENT_PREFIX}"_sci_cased_emb_lc_40e_1e-3lr" \
--device cuda:0 \
--max_length 20 \
--max_num_words 15000 \
--bert_type "scibert-sci-cased" \
--debug_dataset_proportion 0.01 \
--return_instances \
--bs 32 \
--emb_dim 768 \
--lr 1e-3 \
--epochs 40 \
--save_every 1 \
--log_train_metrics_every 50