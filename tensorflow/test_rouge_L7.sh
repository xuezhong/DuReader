export CUDA_VISIBLE_DEVICES=3

python  $@ run.py --train --algo BIDAF --epochs 2 --batch_size 8  \
--train_files 'small/train_demo.json' \
--dev_files 'small/dev_demo.json' \
--vocab_dir 'demo' \
--optim adam \
--embed_size 300 \
--hidden_size 150 \
--max_p_num 5 \
--max_p_len 500 \
--max_q_len 60 \
--max_a_len 200 \
--simple_net 3 \
--dev_interval 20 \
--log_interval 20 \
--para_init \
--learning_rate 1 
