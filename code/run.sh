#!/bin/bash
set -x
set -e

# echo "running run.sh..."

if [ ! -d log ]; then
    mkdir log
else
    rm -r log/*
fi

if [ ! -d output ]; then
    mkdir output
else
    rm -r output/*
fi

mkdir output/models/
mkdir output/results/
mkdir output/summary/

PWD_DIR=`pwd`
PYDIR="python"
PYLIB="$PYDIR/lib/python2.7/site-packages/"

export PATH="$PWD_DIR/$PYDIR/bin/:$PWD_DIR/$PYDIR/lib/:$PATH"
export PYTHONPATH="$PWD_DIR/$PYDIR/lib/:$PWD_DIR/$PYLIB:$PYTHONPATH"
export LD_LIBRARY_PATH="/home/work/cuda-8.0/lib64:/home/work/cudnn/cudnn_v6/cuda/lib64:/home/work/cuda-8.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH"

echo `which python3` > $PWD_DIR/log/info.log
echo $PATH >> $PWD_DIR/log/info.log 
# ls /home/work/cuda-8.0/lib64 >> $PWD_DIR/log/info.log
# ls /home/work/cudnn/cudnn_v5/cuda/lib64 >> $PWD_DIR/log/info.log
# ls /home/work/cuda-8.0/extras/CUPTI/lib64 >> $PWD_DIR/log/info.log

python -c 'import sys; print(sys.path)' >> $PWD_DIR/log/info.log
echo $WORK_DIR >> $PWD_DIR/log/info.log

python -m pdb src/run.py --train \
        --gpu 3 \
        --learning_rate 0.001 \
        --weight_decay 0 \
        --dropout_keep_prob 0.8 \
        --batch_size 32 \
        --epochs 5 \
        --algo BIDAF \
        --embed_size 300 \
        --hidden_size 150 \
        --max_p_num 5 \
        --max_p_len 500 \
        --max_q_len 60 \
        --max_a_len 200 \
        --train_files 'dureader/train.json' \
        --dev_files 'dureader/dev.json' \
        --vocab_dir dureader \
        --model_dir output/models \
        --result_dir output/results \
        --summary_dir output/summary \
        --log_path log/train.log \
