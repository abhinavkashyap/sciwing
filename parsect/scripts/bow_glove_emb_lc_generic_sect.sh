#!/usr/bin/env bash

#!/usr/bin/env bash

cd ../clients

python bow_glove_emb_lc_generic_sect.py \
--exp_name "debug_bow_glove_genericsect" \
--max_num_words 3000 \
--max_length 10 \
--debug \
--debug_dataset_proportion 0.01 \
--bs 32 \
--emb_type glove_6B_50 \
--emb_dim 50 \
--lr 1e-3 \
--epochs 40 \
--save_every 1 \
--log_train_metrics_every 50