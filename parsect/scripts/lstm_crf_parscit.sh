#!/usr/bin/env bash

cd ../clients

EXPERIMENT_PREFIX="lstm_crf_parscit"
SCRIPT_FILE="lstm_crf_client.py"

#python ${SCRIPT_FILE} \
#--exp_name ${EXPERIMENT_PREFIX}"_debug" \
#--device cpu \
#--max_num_words 1000 \
#--max_len 10 \
#--max_char_len 25 \
#--debug \
#--debug_dataset_proportion 0.02 \
#--bs 10 \
#--emb_type random \
#--emb_dim 50 \
#--char_emb_dim 25 \
#--hidden_dim 1024 \
#--lr 1e-2 \
#--bidirectional \
#--use_char_encoder \
#--char_encoder_hidden_dim 100 \
#--combine_strategy concat \
#--epochs 10 \
#--save_every 50 \
#--log_train_metrics_every 5

python ${SCRIPT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}_"no_char_enc_10kw_ml20_100d_512h_1e-2lr_bidir_concat_25e" \
--device cuda:0 \
--max_num_words 10000 \
--max_len 20 \
--max_char_len 100 \
--debug_dataset_proportion 0.02 \
--bs 10 \
--emb_type random \
--emb_dim 100 \
--char_emb_dim 25 \
--hidden_dim 512 \
--lr 1e-2 \
--bidirectional \
--char_encoder_hidden_dim 100 \
--combine_strategy concat \
--epochs 25 \
--save_every 10 \
--log_train_metrics_every 10


python ${SCRIPT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_char_enc_10kw_ml20_mcl100_100d_25cd_512h_100charench_1e-2lr_bidir_concat_25e" \
--device cuda:0 \
--max_num_words 10000 \
--max_len 20 \
--max_char_len 100 \
--debug_dataset_proportion 0.02 \
--bs 10 \
--emb_type random \
--emb_dim 100 \
--char_emb_dim 25 \
--hidden_dim 512 \
--lr 1e-2 \
--bidirectional \
--use_char_encoder \
--char_encoder_hidden_dim 100 \
--combine_strategy concat \
--epochs 25 \
--save_every 10 \
--log_train_metrics_every 10
