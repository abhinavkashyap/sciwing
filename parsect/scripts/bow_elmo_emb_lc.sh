#!/usr/bin/env bash

cd ../clients

python bow_elmo_emb_linear_classifier.py \
--exp_name "bow_elmo_emb_lc_10e_1e-3lr" \
--device cuda:0 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_dim 1024 \
--lr 1e-3 \
--epochs 10 \
--save_every 1 \
--log_train_metrics_every 50


#python bow_elmo_emb_linear_classifier.py \
#--exp_name "bow_elmo_emb_lc_20e_1e-3lr_1" \
#--device cpu \
#--debug_dataset_proportion 0.01 \
#--bs 32 \
#--emb_dim 1024 \
#--lr 1e-3 \
#--epochs 20 \
#--save_every 1 \
#--log_train_metrics_every 50


#python bow_elmo_emb_linear_classifier.py \
#--exp_name "bow_elmo_emb_lc_40e_1e-4lr" \
#--device cpu \
#--debug_dataset_proportion 0.01 \
#--bs 32 \
#--emb_dim 1024 \
#--lr 1e-4 \
#--epochs 40 \
#--save_every 1 \
#--log_train_metrics_every 50
#
#python bow_elmo_emb_linear_classifier.py \
#--exp_name "bow_elmo_emb_lc_50e_1e-4lr" \
#--device cpu \
#--debug_dataset_proportion 0.01 \
#--bs 32 \
#--emb_dim 1024 \
#--lr 1e-4 \
#--epochs 50 \
#--save_every 1 \
#--log_train_metrics_every 50
