#!/usr/bin/env bash

EXPERIMENT_PREFIX="conll_seq_crf"
SCRIPT_FILE="conll_seq_crf.py"

python ${SCRIPT_FILE} \
--exp_name ${EXPERIMENT_PREFIX}"_optim_conlleval_fscore_smallerlr" \
--exp_dir_path  "./output_optim_conlleval_fscore_smallerlr" \
--model_save_dir "./output_optim_conlleval_fscore_smallerlr" \
--device cpu \
--dropout 0.5 \
--reg 0 \
--bs 128 \
--emb_type "glove_6B_300" \
--hidden_dim 300 \
--num_layers=1 \
--bidirectional \
--lr 5e-4 \
--combine_strategy concat \
--epochs 50 \
--save_every 20 \
--log_train_metrics_every 3 \
--sample_proportion 1