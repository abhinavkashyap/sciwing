#!/usr/bin/env bash

EXPERIMENT_PREFIX="lstm_crf_parscit"
SCRIPT_FILE="parscit.py"

#python ${SCRIPT_FILE} \
#--exp_name ${EXPERIMENT_PREFIX}"_glove100_128h_15e" \
#--model_save_dir "./output/checkpoints" \
#--device cuda:1 \
#--bs 32 \
#--emb_type "glove_6B_100" \
#--char_emb_dim 25 \
#--hidden_dim 128 \
#--lr 1e-3 \
#--char_encoder_hidden_dim 100 \
#--combine_strategy concat \
#--epochs 25 \
#--save_every 10 \
#--log_train_metrics_every 20 \
#--sample_proportion 1
#
#python ${SCRIPT_FILE} \
#--exp_name ${EXPERIMENT_PREFIX}"_glove200_128h_bidir_15e" \
#--model_save_dir "./output/checkpoints" \
#--device cuda:1 \
#--bs 32 \
#--emb_type "glove_6B_200" \
#--char_emb_dim 25 \
#--hidden_dim 128 \
#--bidirectional \
#--lr 1e-2 \
#--char_encoder_hidden_dim 100 \
#--combine_strategy concat \
#--epochs 15 \
#--save_every 10 \
#--log_train_metrics_every 20 \
#--sample_proportion 1
#
#python ${SCRIPT_FILE} \
#--exp_name ${EXPERIMENT_PREFIX}"_glove200_256h_bidir_15e" \
#--model_save_dir "./output/checkpoints" \
#--device cuda:2 \
#--bs 32 \
#--emb_type "glove_6B_200" \
#--char_emb_dim 25 \
#--hidden_dim 256 \
#--bidirectional \
#--lr 1e-2 \
#--char_encoder_hidden_dim 100 \
#--combine_strategy concat \
#--epochs 15 \
#--save_every 10 \
#--log_train_metrics_every 20 \
#--sample_proportion 1

#python ${SCRIPT_FILE} \
#--exp_name ${EXPERIMENT_PREFIX}"_glove200_64h_bidir_25e_1e-3lr" \
#--model_save_dir "./output/checkpoints" \
#--device cuda:1 \
#--bs 64 \
#--emb_type "glove_6B_200" \
#--char_emb_dim 25 \
#--hidden_dim 64 \
#--bidirectional \
#--lr 1e-3 \
#--char_encoder_hidden_dim 100 \
#--combine_strategy concat \
#--epochs 25 \
#--save_every 10 \
#--log_train_metrics_every 10 \
#--sample_proportion 1

#python ${SCRIPT_FILE} \
#--exp_name ${EXPERIMENT_PREFIX}"_glove200_64h_bidir_100e_1e-3lr" \
#--model_save_dir "./output/checkpoints" \
#--device cuda:1 \
#--bs 64 \
#--emb_type "glove_6B_200" \
#--char_emb_dim 25 \
#--hidden_dim 64 \
#--bidirectional \
#--lr 1e-3 \
#--char_encoder_hidden_dim 100 \
#--combine_strategy concat \
#--epochs 100 \
#--save_every 10 \
#--log_train_metrics_every 10 \
#--sample_proportion 1
#
#python ${SCRIPT_FILE} \
#--exp_name ${EXPERIMENT_PREFIX}"_glove300_64h_bidir_100e_1e-3lr" \
#--model_save_dir "./output/checkpoints" \
#--device cuda:1 \
#--bs 64 \
#--emb_type "glove_6B_300" \
#--char_emb_dim 25 \
#--hidden_dim 64 \
#--bidirectional \
#--lr 1e-3 \
#--char_encoder_hidden_dim 100 \
#--combine_strategy concat \
#--epochs 100 \
#--save_every 10 \
#--log_train_metrics_every 10 \
#--sample_proportion 1
#
#python ${SCRIPT_FILE} \
#--exp_name ${EXPERIMENT_PREFIX}"_parscit_64h_bidir_100e_1e-3lr" \
#--model_save_dir "./output/checkpoints" \
#--device cuda:1 \
#--bs 64 \
#--emb_type "parscit" \
#--char_emb_dim 25 \
#--hidden_dim 64 \
#--bidirectional \
#--lr 1e-3 \
#--char_encoder_hidden_dim 100 \
#--combine_strategy concat \
#--epochs 100 \
#--save_every 10 \
#--log_train_metrics_every 10 \
#--sample_proportion 1

#python ${SCRIPT_FILE} \
#--exp_name ${EXPERIMENT_PREFIX}"_parscit_128h_bidir_100e_1e-3lr" \
#--model_save_dir "./output/checkpoints" \
#--device cuda:1 \
#--bs 64 \
#--emb_type "parscit" \
#--char_emb_dim 25 \
#--hidden_dim 128 \
#--bidirectional \
#--lr 1e-3 \
#--char_encoder_hidden_dim 100 \
#--combine_strategy concat \
#--epochs 100 \
#--save_every 10 \
#--log_train_metrics_every 10 \
#--sample_proportion 1

python ${SCRIPT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_final_model" \
--model_save_dir "./output/"${EXPERIMENT_PREFIX}"_final" \
--device "cuda:2" \
--bs 64 \
--emb_type "parscit" \
--hidden_dim 256 \
--bidirectional \
--lr 1e-3 \
--char_emb_dim 25 \
--char_encoder_hidden_dim 50 \
--combine_strategy concat \
--epochs 50 \
--save_every 10 \
--log_train_metrics_every 10 \
--sample_proportion 1