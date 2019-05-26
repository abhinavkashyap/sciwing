#!/usr/bin/env bash

cd ../clients

python bow_elmo_emb_linear_classifier.py \
--exp_name "bow_elmo_emb_lc_10e_1e-3lr" \
--debug_dataset_proportion 0.01 \
--return_instances \
--bs 32 \
--emb_dim 1024 \
--lr 1e-4 \
--epochs 10 \
--save_every 1 \
--log_train_metrics_every 50

python bow_elmo_emb_linear_classifier.py \
--exp_name "bow_elmo_emb_lc_10e_1e-4lr" \
--debug_dataset_proportion 0.01 \
--return_instances \
--bs 32 \
--emb_dim 1024 \
--lr 1e-4 \
--epochs 10 \
--save_every 1 \
--log_train_metrics_every 50
