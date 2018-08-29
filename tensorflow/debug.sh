export CUDA_VISIBLE_DEVICES=2

python  -m pdb run.py --train --algo BIDAF --epochs 10 --batch_size 3 --embed_size 7 --max_p_len 20 --hidden_size 7 --max_q_len 20 $@
