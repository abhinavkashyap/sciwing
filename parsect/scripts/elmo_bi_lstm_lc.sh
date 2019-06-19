#!/usr/bin/env bash

cd ../clients

python elmo_bi_lstm_encoder_linear_classifier.py \
--exp_name "debug_elmo_bi_lstm_lc" \
--device cpu \
--max_num_words 1000 \
--max_length 10 \
--debug \
--debug_dataset_proportion 0.01 \
--bs 10 \
--emb_type random \
--emb_dim 300 \
--hidden_dim 512 \
--lr 1e-3 \
--bidirectional \
--combine_strategy concat \
--epochs 1 \
--save_every 1 \
--log_train_metrics_every 5

python elmo_bi_lstm_encoder_linear_classifier.py \
--exp_name "elmo_bi_lstm_lc_1kw_ml10_300d_512h_1e-3_bidir_concat_20e" \
--device cuda:0 \
--max_num_words 1000 \
--max_length 10 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type random \
--emb_dim 300 \
--hidden_dim 512 \
--lr 1e-3 \
--bidirectional \
--combine_strategy concat \
--epochs 20 \
--save_every 1 \
--log_train_metrics_every 10

python elmo_bi_lstm_encoder_linear_classifier.py \
--exp_name "elmo_bi_lstm_lc_2kw_ml10_300d_512h_1e-3_bidir_concat_20e" \
--device cuda:0 \
--max_num_words 2000 \
--max_length 10 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type random \
--emb_dim 300 \
--hidden_dim 512 \
--lr 1e-3 \
--bidirectional \
--combine_strategy concat \
--epochs 20 \
--save_every 1 \
--log_train_metrics_every 50

python elmo_bi_lstm_encoder_linear_classifier.py \
--exp_name "elmo_bi_lstm_lc_3kw_ml10_300d_512h_1e-3_bidir_concat_20e" \
--device cuda:0 \
--max_num_words 3000 \
--max_length 10 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type random \
--emb_dim 300 \
--hidden_dim 512 \
--lr 1e-3 \
--bidirectional \
--combine_strategy concat \
--epochs 20 \
--save_every 1 \
--log_train_metrics_every 50


python elmo_bi_lstm_encoder_linear_classifier.py \
--exp_name "elmo_bi_lstm_lc_5kw_ml10_300d_512h_1e-3_bidir_concat_20e" \
--device cuda:0 \
--max_num_words 5000 \
--max_length 10 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type random \
--emb_dim 300 \
--hidden_dim 512 \
--lr 1e-3 \
--bidirectional \
--combine_strategy concat \
--epochs 20 \
--save_every 1 \
--log_train_metrics_every 50


python elmo_bi_lstm_encoder_linear_classifier.py \
--exp_name "elmo_bi_lstm_lc_8kw_ml10_300d_512h_1e-3_bidir_concat_20e" \
--device cuda:0 \
--max_num_words 8000 \
--max_length 10 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type random \
--emb_dim 300 \
--hidden_dim 512 \
--lr 1e-3 \
--bidirectional \
--combine_strategy concat \
--epochs 20 \
--save_every 1 \
--log_train_metrics_every 50

python elmo_bi_lstm_encoder_linear_classifier.py \
--exp_name "elmo_bi_lstm_lc_5kw_ml15_300d_512h_1e-3_bidir_concat_20e" \
--device cuda:0 \
--max_num_words 5000 \
--max_length 15 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type random \
--emb_dim 300 \
--hidden_dim 512 \
--lr 1e-3 \
--bidirectional \
--combine_strategy concat \
--epochs 20 \
--save_every 5 \
--log_train_metrics_every 50


python elmo_bi_lstm_encoder_linear_classifier.py \
--exp_name "elmo_bi_lstm_lc_5kw_ml20_300d_512h_1e-3_bidir_concat_20e" \
--device cuda:0 \
--max_num_words 5000 \
--max_length 20 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type random \
--emb_dim 300 \
--hidden_dim 512 \
--lr 1e-3 \
--bidirectional \
--combine_strategy concat \
--epochs 20 \
--save_every 5 \
--log_train_metrics_every 50

python elmo_bi_lstm_encoder_linear_classifier.py \
--exp_name "elmo_bi_lstm_lc_5kw_ml20_300d_256h_1e-3_bidir_concat_20e" \
--device cuda:0 \
--max_num_words 5000 \
--max_length 20 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type random \
--emb_dim 300 \
--hidden_dim 256 \
--lr 1e-3 \
--bidirectional \
--combine_strategy concat \
--epochs 20 \
--save_every 5 \
--log_train_metrics_every 50

python elmo_bi_lstm_encoder_linear_classifier.py \
--exp_name "elmo_bi_lstm_lc_5kw_ml20_300d_128h_1e-3_bidir_concat_20e" \
--device cuda:0 \
--max_num_words 5000 \
--max_length 20 \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type random \
--emb_dim 300 \
--hidden_dim 128 \
--lr 1e-3 \
--bidirectional \
--combine_strategy concat \
--epochs 20 \
--save_every 5 \
--log_train_metrics_every 50



