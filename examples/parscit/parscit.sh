#!/usr/bin/env bash

EXPERIMENT_PREFIX="lstm_crf_parscit"
SCRIPT_FILE="parscit.py"


python ${SCRIPT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_char_enc_10kw_ml75_mcl15_500d_25cd_256h_50charench_1e-3lr_bidir_concat_20e" \
--model_save_dir "./output/checkpoints" \
--device cpu \
--bs 64 \
--emb_type "glove_6B_100" \
--char_emb_dim 25 \
--hidden_dim 1024 \
--bidirectional \
--lr 1e-3 \
--char_encoder_hidden_dim 100 \
--combine_strategy concat \
--epochs 20 \
--save_every 10 \
--log_train_metrics_every 10 \
--sample_proportion 0.1

