#!/usr/bin/env bash

cd ../clients

#python bow_bert_emb_linear_classifier.py \
#--exp_name "bow_bert_base_cased_emb_lc_10e_1e-2lr" \
#--device cuda:0 \
#--bert_type "bert-large-cased" \
#--debug_dataset_proportion 0.01 \
#--return_instances \
#--bs 32 \
#--emb_dim 1024 \
#--lr 1e-2 \
#--epochs 10 \
#--save_every 1 \
#--log_train_metrics_every 50

python bow_bert_emb_linear_classifier.py \
--exp_name "bow_bert_large_cased_emb_lc_20e_1e-2lr" \
--device cuda:0 \
--bert_type "bert-large-cased" \
--debug_dataset_proportion 0.01 \
--return_instances \
--bs 32 \
--emb_dim 1024 \
--lr 1e-2 \
--epochs 20 \
--save_every 1 \
--log_train_metrics_every 50

python bow_bert_emb_linear_classifier.py \
--exp_name "bow_bert_large_cased_emb_lc_40e_1e-3lr" \
--device cuda:0 \
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
python bow_bert_emb_linear_classifier.py \
--exp_name "bow_scibert_sci_cased_emb_lc_20e_1e-2lr" \
--device cuda:0 \
--bert_type "scibert-sci-cased" \
--debug_dataset_proportion 0.01 \
--return_instances \
--bs 32 \
--emb_dim 1024 \
--lr 1e-2 \
--epochs 20 \
--save_every 1 \
--log_train_metrics_every 50

python bow_bert_emb_linear_classifier.py \
--exp_name "bow_scibert_sci_cased_emb_lc_40e_1e-3lr" \
--device cuda:0 \
--bert_type "scibert-sci-cased" \
--debug_dataset_proportion 0.01 \
--return_instances \
--bs 32 \
--emb_dim 1024 \
--lr 1e-3 \
--epochs 40 \
--save_every 1 \
--log_train_metrics_every 50