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

import paddle.fluid.layers as layers
import numpy as np

def seq_to_seq_net(embedding_dim, encoder_size, decoder_size, source_dict_dim,
                   target_dict_dim,  max_length, args):
    def encoder(input_name, input_shape):
        input_ids = layers.data(
            name=input_name, shape=[1], dtype='int64', lod_level=1)
        input_embedding = layers.embedding(
            input=input_ids,
            size=[source_dict_dim, embedding_dim],
            dtype='float32',
            is_sparse=True)
        fc1 = layers.fc(input=input_embedding, size=encoder_size * 4, act='tanh')
        lstm_hidden0, lstm_0 = layers.dynamic_lstm(
            input=fc1, size=encoder_size * 4, is_reverse = True)
        encoder_out = layers.sequence_last_step(input=lstm_hidden0)
        return encoder_out
    q_name = 'q_ids'
    q_shape = np.array([args.batch_size, args.max_q_len], dtype="int32")
    q_enc = encoder(q_name, q_shape)
        
    p_name = 'p_ids'
    p_shape = np.array([args.batch_size, args.max_p_len], dtype="int32")
    p_enc = encoder(p_name, p_shape)

    start_labels = layers.data(
	name="start_lables", shape=[args.max_p_len], dtype='float32', lod_level=0)
    
    end_labels = layers.data(
	name="end_lables", shape=[1], dtype='int32', lod_level=1)

    #decode
    decode_out = layers.fc(input=p_enc, size=args.max_p_len, act='tanh')

    #compute loss
    cost = layers.cross_entropy(input=decode_out, label=start_labels, soft_label=True)
    avg_cost = layers.mean(x=cost)

    feeding_list = ['q_ids', 'p_ids', "start_lables", "end_lables"]
    return avg_cost, feeding_list
