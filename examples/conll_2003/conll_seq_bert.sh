#!/usr/bin/env bash

EXPERIMENT_PREFIX="conll_seq_crf"
SCRIPT_FILE="conll_seq_bert.py"

python ${SCRIPT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_lample_bert_larger_lr" \
--exp_dir_path  "./_lample_bert_larger_lr" \
--model_save_dir "./_lample_bert_larger_lr/checkpoints" \
--device cuda:2 \
--dropout 0.5 \
--reg 0 \
--bs 64 \
--emb_type "lample_conll" \
--bert_type "bert-base-uncased" \
--hidden_dim 200 \
--add_projection_layer \
--num_layers=1 \
--bidirectional \
--char_emb_dim 10 \
--char_encoder_hidden_dim 25 \
--lr 1e-2 \
--combine_strategy concat \
--epochs 75 \
--save_every 20 \
--log_train_metrics_every 5 \
--sample_proportion 1