export CUDA_VISIBLE_DEVICES=3

python src/run.py --train --algo BIDAF --epochs 50 --batch_size 8  --max_p_len 500 --hidden_size 150 --max_q_len 60 \
--train_files 'small/train_demo.json' \
--dev_files 'small/dev_demo.json' \
--vocab_dir 'small/' \
--optim rprop \
--embed_size 300 \
--hidden_size 150 \
--max_p_num 5 \
--max_p_len 500 \
--max_q_len 60 \
--max_a_len 200 \
--weight_decay 0 \
--ema_decay 0.0 \
--dropout_keep_prob 0.8 \
$@
