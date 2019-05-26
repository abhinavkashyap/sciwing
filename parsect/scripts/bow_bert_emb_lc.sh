#!/usr/bin/env bash

#!/usr/bin/env bash

cd ../clients

python bow_bert_emb_linear_classifier.py \
--exp_name "debug_bow_bert_base_cased_emb_lc_40e_1e-1lr" \
--bert_type "bert-base-cased" \
--debug \
--debug_dataset_proportion 0.01 \
--return_instances \
--bs 10 \
--emb_dim 768 \
--lr 1e-1 \
--epochs 40 \
--save_every 1 \
--log_train_metrics_every 10