export CUDA_VISIBLE_DEVICES=2

python  run.py --train --algo BIDAF --epochs 10 --batch_size 32 $@
