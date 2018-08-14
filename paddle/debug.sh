export CUDA_VISIBLE_DEVICES=3
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
export PYTHONPATH=/centos/my_Dureader/paddle:/centos/my_Dureader

python -m pdb ./models/test_bidaf/env/run.py --trainset ../data/demo/trainset/search.train.json --testset ../data/demo/devset/search.dev.json --vocab_file ../data/demo/vocab.search --emb_dim 300 --batch_size 2 --vocab_size 31825 --trainer_count 1 --log_period 10 --test_period 100 --use_gpu --save_dir ./models/test_bidaf/models --algo bidaf --saving_period 1000 
