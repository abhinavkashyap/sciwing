#!/usr/bin/env bash


SCRIPT_FILE="citation_intent_clf.py"


python ${SCRIPT_FILE} \
--exp_name "citation_intent_clf_elmo_slower" \
--exp_dir_path  "./citation_intent_clf_elmo_slower" \
--model_save_dir "./citation_intent_clf_elmo_slower/checkpoints" \
--emb_type "glove_6B_100" \
--device cuda:7 \
--bs 32 \
--hidden_dim 50 \
--lr 1e-4 \
--bidirectional \
--combine_strategy concat \
--epochs 50 \
--save_every 10 \
--log_train_metrics_every 20 \
--sample_proportion 1
