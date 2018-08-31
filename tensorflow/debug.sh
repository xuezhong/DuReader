export CUDA_VISIBLE_DEVICES=2

python  -m pdb run.py --train --algo BIDAF --epochs 10 --batch_size 3  --max_p_len 20 --hidden_size 7 --max_q_len 20 \
--train_files '../data/demo/trainset/search.train.json' \
--dev_files '../data/demo/devset/search.dev.json' \
--test_files '../data/demo/testset/search.test.json' \
--vocab_dir '../data/demo/vocab/' \
--optim rprop \
$@
