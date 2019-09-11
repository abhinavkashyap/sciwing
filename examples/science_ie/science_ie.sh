#!/usr/bin/env bash


EXPERIMENT_PREFIX="lstm_crf_scienceie"
SCRIPT_FILE="science_ie.py"

python ${SCRIPT_FILE} \
--exp_name ${EXPERIMENT_PREFIX} \
--exp_dir_path  "./output" \
--model_save_dir "./output/checkpoints" \
--vocab_store_location "./output/vocab.json" \
--char_vocab_store_location "./output/char_vocab.json" \
--device cpu \
--max_num_words 1000 \
--max_len 10 \
--max_char_len 25 \
--dropout 0.4 \
--reg 0 \
--debug \
--debug_dataset_proportion 0.01 \
--bs 10 \
--emb_type random \
--emb_dim 50 \
--char_emb_dim 25 \
--hidden_dim 1024 \
--lr 1e-2 \
--bidirectional \
--use_char_encoder \
--char_encoder_hidden_dim 100 \
--combine_strategy concat \
--epochs 1 \
--save_every 50 \
--log_train_metrics_every 5 \
--seq_num_layers 1