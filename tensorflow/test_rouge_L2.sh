export CUDA_VISIBLE_DEVICES=3

python run.py --train --algo BIDAF --epochs 50 --batch_size 1  \
--train_files 'small/train_demo.json' \
--dev_files 'small/dev_demo.json.head' \
--vocab_dir 'demo' \
--optim adam \
--embed_size 300 \
--hidden_size 3 \
--max_p_num 5 \
--max_p_len 10 \
--max_q_len 6 \
--max_a_len 20 \
--simple_net 3 \
--dev_interval 1 \
--log_interval 1 \
--debug_print \
--para_init \
--learning_rate 0.0 \
$@
