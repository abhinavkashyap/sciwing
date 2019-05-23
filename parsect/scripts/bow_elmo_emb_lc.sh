#!/usr/bin/env bash

cd ../clients

python bow_elmo_emb_linear_classifier.py \
--exp_name "bow_elmo_emb_lc_10e_1e-3lr" \
--debug \
--debug_dataset_proportion 0.01 \
--bs 32 \
--lr 1e-4 \
--epochs 10 \
--save_every 1 \
--log_train_metrics_every 50
