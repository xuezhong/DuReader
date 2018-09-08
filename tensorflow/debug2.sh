export CUDA_VISIBLE_DEVICES=2

python  run.py --train --algo BIDAF --epochs 10 --batch_size 1  --max_p_len 10 --hidden_size 3 --max_q_len 10 \
--train_files 'demo/trainset/search.train.json' \
--dev_files 'demo/devset/search.dev.json' \
--test_files 'demo/testset/search.test.json' \
--vocab_dir 'demo' \
--optim sgd \
--simple_net 0 \
--debug_print \
--para_init \
--learning_rate 0.0 \
$@
