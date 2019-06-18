#!/usr/bin/env bash

cd ../clients

python bow_random_emb_lc_generic_sect.py \
--exp_name "debug_bow_random_generic_sect" \
--max_num_words 400 \
--max_length 5 \
--debug \
--debug_dataset_proportion 0.1 \
--bs 32 \
--emb_dim 200 \
--lr 1e-2 \
--epochs 40 \
--save_every 1 \
--log_train_metrics_every 50