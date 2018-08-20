export CUDA_VISIBLE_DEVICES=3
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
export PYTHONPATH=/centos/my_Dureader/fluid:/centos/my_Dureader

python -m pdb train.py --trainset ../data/demo/trainset/search.train.json --testset ../data/demo/devset/search.dev.json --vocab_file ../data/demo/vocab.search --embedding_dim 300 --batch_size 20 --vocab_size 31825 --use_gpu false --save_dir ./models/test_bidaf/models --max_p_len 1000 
