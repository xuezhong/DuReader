export CUDA_VISIBLE_DEVICES=2

python  -m pdb run.py --train --algo BIDAF --epochs 10 --batch_size 1  --max_p_len 10 --hidden_size 3 --max_q_len 10 \
--train_files 'small/train_demo.json' \
--dev_files 'small/dev_demo.json' \
--vocab_dir 'small/' \
--optim sgd \
--simple_net 3 \
--para_init \
--learning_rate 0.0 \
--log_interval 1 \
$@
