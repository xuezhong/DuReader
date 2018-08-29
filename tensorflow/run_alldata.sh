python run.py --train --algo BIDAF --epochs 32 --batch_size 16 --gpu 3 \
--train_files '../data/preprocessed/trainset/search.train.json' \
              '../data/preprocessed/trainset/zhidao.train.json' \
--dev_files '../data/preprocessed/devset/search.dev.json' \
            '../data/preprocessed/devset/zhidao.dev.json' \
--test_files '../data/preprocessed/testset/search.test.json' \
            '../data/preprocessed/testset/zhidao.test.json' \
--simple_decoder --hidden_size 300 
