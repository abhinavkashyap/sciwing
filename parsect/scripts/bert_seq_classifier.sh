#!/usr/bin/env bash

cd ../clients

EXPERIMENT_PREFIX="parsect_bert_seq"
SCRIPT_FILE="bert_seq_classifier_client.py"

#python ${SCRIPT_FILE} \
#--exp_name "debug_"${EXPERIMENT_PREFIX} \
#--bert_type "bert-base-uncased" \
#--device cuda:0 \
#--max_num_words 15000 \
#--max_len 25 \
#--debug \
#--debug_dataset_proportion 0.02 \
#--bs 32 \
#--emb_dim 768 \
#--lr 1e-4 \
#--epochs 1 \
#--save_every 15 \
#--log_train_metrics_every 1

python ${SCRIPT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_large_cased_emb_40e_1e-4lr" \
--bert_type "bert-large-cased" \
--device cuda:0 \
--max_num_words 15000 \
--max_len 25 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_dim 1024 \
--lr 1e-4 \
--epochs 40 \
--save_every 1 \
--log_train_metrics_every 50

#python ${SCRIPT_FILE} \
#--exp_name ${EXPERIMENT_PREFIX}"_sci_uncased_emb_lc_10e_1e-4lr" \
#--bert_type "scibert-sci-uncased" \
#--device cuda:0 \
#--max_num_words 15000 \
#--max_len 25 \
#--debug_dataset_proportion 0.01 \
#--bs 32 \
#--emb_dim 768 \
#--lr 1e-4 \
#--epochs 10 \
#--save_every 5 \
#--log_train_metrics_every 50