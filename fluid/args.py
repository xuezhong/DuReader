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

import argparse
import distutils.util


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=512,
        help="The dimension of embedding table. (default: %(default)d)")
    parser.add_argument(
        "--encoder_size",
        type=int,
        default=512,
        help="The size of encoder bi-rnn unit. (default: %(default)d)")
    parser.add_argument(
        "--decoder_size",
        type=int,
        default=512,
        help="The size of decoder rnn unit. (default: %(default)d)")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="The sequence number of a mini-batch data. (default: %(default)d)")
    parser.add_argument(
        "--dict_size",
        type=int,
        default=31825,
        help="The dictionary capacity. Dictionaries of source sequence and "
        "target dictionary have same capacity. (default: %(default)d)")
    parser.add_argument(
        "--pass_num",
        type=int,
        default=5,
        help="The pass number to train. (default: %(default)d)")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate used to train the model. (default: %(default)f)")
    parser.add_argument(
        "--no_attention",
        action='store_true',
        help="If set, run no attention model instead of attention model.")
    parser.add_argument(
        "--beam_size",
        type=int,
        default=3,
        help="The width for beam searching. (default: %(default)d)")
    parser.add_argument(
        "--use_gpu",
        type=distutils.util.strtobool,
        default=True,
        help="Whether to use gpu. (default: %(default)d)")
    parser.add_argument(
        "--debug",
        type=distutils.util.strtobool,
        default=False,
        help="Whether to print debug info. (default: %(default)d)")
    parser.add_argument(
        "--max_length",
        type=int,
        default=50,
        help="The maximum length of sequence when doing generation. "
        "(default: %(default)d)")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="model",
        help="Specify the path to save trained models.")
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1,
        help="Save the trained model every n passes."
        "(default: %(default)d)")
    parser.add_argument(
        "--enable_ce",
        action='store_true',
        help="If set, run the task with continuous evaluation logs.")
    parser.add_argument('--trainset', nargs='+', help='train dataset')
    parser.add_argument('--testset', nargs='+', help='test dataset')
    parser.add_argument('--vocab_file', help='dict')
    parser.add_argument('--vocab_size', help='vocab size',
                        default=-1, type=int)
    parser.add_argument('--max_p_len', type=int, default=500)
    parser.add_argument('--max_q_len', type=int, default=9)
    parser.add_argument('--doc_num', type=int, default=5)
    parser.add_argument('--single_doc', action='store_true')
    parser.add_argument('--simple_decode', action='store_true')
    args = parser.parse_args()
    return args
