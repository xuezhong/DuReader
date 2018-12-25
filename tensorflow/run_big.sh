#!/bin/bash
set -x
set -e

# echo "running run.sh..."

PWD_DIR=`pwd`
PYDIR="python"
PYLIB="$PYDIR/lib/python2.7/site-packages/"
export CUDA_VISIBLE_DEVICES=4
export PATH="$PWD_DIR/$PYDIR/bin/:$PWD_DIR/$PYDIR/lib/:$PATH"
export PYTHONPATH="$PWD_DIR/$PYDIR/lib/:$PWD_DIR/$PYLIB:$PYTHONPATH"

python run.py --train \
        --gpu 6 \
        --learning_rate 0.001 \
	--optim adam \
        --weight_decay 0.0001 \
        --dropout_keep_prob 0.8 \
        --batch_size 32 \
        --epochs 10 \
        --algo BIDAF \
        --embed_size 300 \
        --hidden_size 150 \
        --max_p_num 5 \
        --max_p_len 500 \
        --max_q_len 60 \
        --max_a_len 200 \
        --train_files 'data/preprocessed/trainset/search.train.json' 'data/preprocessed/trainset/zhidao.train.json'\
        --dev_files 'data/preprocessed/devset/search.dev.json' 'data/preprocessed/devset/zhidao.dev.json' \
        --vocab_dir data/vocab \
        --model_dir output/models \
        --result_dir output/results \
        --summary_dir output/summary \
	--simple_net 3 \
	$@
