#!/usr/bin/env bash

EXPERIMENT_PREFIX="conll_seq_crf_gazetteer_attention"
SCRIPT_FILE="conll_seq_gazeteer_attention.py"

python ${SCRIPT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_gazetteer_attention_trainable_glove_300" \
--exp_dir_path  "./_gazetteer_attention_trainable_glove_300" \
--model_save_dir "./_gazetteer_attention_trainable_glove_300/checkpoints" \
--device cuda:6 \
--dropout 0.5 \
--reg 0 \
--bs 64 \
--emb_type "lample_conll" \
--hidden_dim 150 \
--num_layers=1 \
--bidirectional \
--char_emb_dim 10 \
--char_encoder_hidden_dim 25 \
--lr 1e-3 \
--combine_strategy concat \
--epochs 75 \
--save_every 20 \
--log_train_metrics_every 5 \
--sample_proportion 1