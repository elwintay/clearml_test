#!/usr/bin/env bash

export MAX_LENGTH_SRC=512
export MAX_LENGTH_TGT=90
export BERT_MODEL=bert-base-uncased

export BATCH_SIZE=1
export NUM_EPOCHS=1 # original 50
export SEED=1

# export OUTPUT_DIR_NAME=muc1024_e1
# export CURRENT_DIR=${PWD}
# export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}
# mkdir -p $OUTPUT_DIR

export OUTPUT_DIR=/gtt/bert-base-uncased/checkpoints/wikievents512_e1

# Add parent directory to python path to access transformer_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

# for th in 100 200 300 400 500 600 700 800 900 1000  # 1 10 100 1000 10000 100000, 100 200 300 400 500
# for th in 10 20 30 40 50 60 70 80 90 100
# for th in 100
# for th in 10 50 60 70 80 90 100 110 120 130 140 150
for th in 80
do
echo "=========================================================================================="
echo "                                           threshold (${th})                              "
echo "=========================================================================================="
python3 main.py  \
--data_dir /gtt/data/wikievents/muc_format/ \
--model_type bert \
--model_name_or_path /gtt/models/bert-base-uncased/model \
--output_dir $OUTPUT_DIR \
--max_seq_length_src  $MAX_LENGTH_SRC \
--max_seq_length_tgt $MAX_LENGTH_TGT \
--num_train_epochs $NUM_EPOCHS \
--train_batch_size $BATCH_SIZE \
--eval_batch_size $BATCH_SIZE \
--seed $SEED \
--n_gpu 1 \
--thresh $th \
--learning_rate 5e-7 \
--do_predict
#--do_train
# --do_predict \
done