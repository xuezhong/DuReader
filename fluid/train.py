#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import os

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.framework as framework
from paddle.fluid.executor import Executor

from args import *
import bidaf_model
import dataset
def padding(input, max_len, padding_index):
    input = input[0:max_len]
    input = input + (max_len - len(input)) * [padding_index]
    return input

def prepare_batch_input(insts, args):
    doc_num = 5
    
    new_insts = []
    for inst in insts:
        p_id = []
        start_label = []
        end_label = []
        start_num = [0]
        end_num = [0]

        q_id = inst[0]
        q_id = padding(q_id, args.max_q_len, 0)
        
        for i in range(1 + 0 * doc_num, 1 + 1 * doc_num):
            p_id = p_id + inst[i]
        p_id = padding(p_id, args.max_p_len, 0)
        
        for i in range(1 + 2 * doc_num, 1 + 3 * doc_num):
            start_label = start_label + inst[i]
        start_label = padding(start_label, args.max_p_len, [0.0])
        start_label = [x[0] for x in start_label] 
        for i in range(len(start_label)):
            if start_label[i] == 1.0:
                start_num = [i]
                break
        
        for i in range(1 + 3 * doc_num, 1 + 4 * doc_num):
            end_label = end_label + inst[i]
        for i in range(len(end_label)):
            if end_label[i] == 1.0:
                end_num = [i]
                break

        end_label = padding(end_label, args.max_p_len, [0.0])
        end_label = [x[0] for x in end_label]
        
        new_inst = [q_id, p_id, start_label, end_label]
        new_insts.append(new_inst)
    return new_insts
        

def train():
    args = parse_args()

    if args.enable_ce:
        framework.default_startup_program().random_seed = 111

    # Training process
    avg_cost, feed_order = bidaf_model.bidaf(
	args.embedding_dim,
	args.encoder_size,
	args.decoder_size,
	args.vocab_size,
	args.vocab_size,
	args.max_length,
        args)
    # clone from default main program and use it as the validation program
    main_program = fluid.default_main_program()
    inference_program = fluid.default_main_program().clone()

    optimizer = fluid.optimizer.Adam(
        learning_rate=args.learning_rate,
        regularization=fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=1e-5))

    optimizer.minimize(avg_cost)

    # Disable shuffle for Continuous Evaluation only
    train_batch_generator = paddle.batch(dataset.DuReaderQA(
		   file_names=args.trainset,
		   vocab_file=args.vocab_file,
		   vocab_size=args.vocab_size,
		   max_p_len=args.max_p_len,
		   shuffle=(False),
		   preload=(False)).create_reader(),
                   batch_size=args.batch_size,
                   drop_last=False)


    place = core.CUDAPlace(0) if args.use_gpu else core.CPUPlace()
    exe = Executor(place)
    exe.run(framework.default_startup_program())

    feed_list = [
        main_program.global_block().var(var_name) for var_name in feed_order
    ]
    feeder = fluid.DataFeeder(feed_list, place)

    def validation():
        # Use test set as validation each pass
        total_loss = 0.0
        count = 0
        val_feed_list = [
            inference_program.global_block().var(var_name)
            for var_name in feed_order
        ]
        val_feeder = fluid.DataFeeder(val_feed_list, place)

        for batch_id, data in enumerate(test_batch_generator()):
            val_fetch_outs = exe.run(inference_program,
                                     feed=val_feeder.feed(data),
                                     fetch_list=[avg_cost],
                                     return_numpy=False)

            total_loss += np.array(val_fetch_outs[0])[0]
            count += 1

        return total_loss / count

    for pass_id in range(1, args.pass_num + 1):
        pass_start_time = time.time()
        words_seen = 0
        for batch_id, data in enumerate(train_batch_generator()):
            
            input_data_dict = prepare_batch_input(data, args)
            fetch_outs = exe.run(framework.default_main_program(),
                                 feed=feeder.feed(input_data_dict),
                                 fetch_list=[avg_cost])

            avg_cost_train = np.array(fetch_outs[0])
            print('pass_id=%d, batch_id=%d, train_loss: %f' %
                  (pass_id, batch_id, avg_cost_train))
            # This is for continuous evaluation only
            if args.enable_ce and batch_id >= 100:
                break

        pass_end_time = time.time()
        test_loss = validation()
        time_consumed = pass_end_time - pass_start_time
        words_per_sec = words_seen / time_consumed
        print("pass_id=%d, test_loss: %f, words/s: %f, sec/pass: %f" %
              (pass_id, test_loss, words_per_sec, time_consumed))

        # This log is for continuous evaluation only
        if args.enable_ce:
            print("kpis\ttrain_cost\t%f" % avg_cost_train)
            print("kpis\ttest_cost\t%f" % test_loss)
            print("kpis\ttrain_duration\t%f" % time_consumed)

        if pass_id % args.save_interval == 0:
            model_path = os.path.join(args.save_dir, str(pass_id))
            if not os.path.isdir(model_path):
                os.makedirs(model_path)

            fluid.io.save_persistables(
                executor=exe,
                dirname=model_path,
                main_program=framework.default_main_program())


if __name__ == '__main__':
    train()
