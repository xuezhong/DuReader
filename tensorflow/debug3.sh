export CUDA_VISIBLE_DEVICES=2

python  run.py --train \
	--algo BIDAF \
	--epochs 10 \
	--batch_size 8  \
	--max_p_len 500 \
	--hidden_size 150 \
	--max_q_len 60 \
--train_files 'small/train_demo.json' \
--dev_files 'small/dev_demo.json' \
--vocab_dir '../../DuReaderWithPaddle/DuReader_v2/demo/' \
--optim sgd \
--simple_net 3 \
--para_init \
--learning_rate 0.0 \
--log_interval 1 \
--debug_print \
$@
