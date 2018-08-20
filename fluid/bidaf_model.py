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

def bidaf(embedding_dim, encoder_size, decoder_size, source_dict_dim,
                   target_dict_dim,  max_length, args):
    def bi_lstm_encoder(input_seq, gate_size):
        # A bi-directional lstm encoder implementation.
        # Linear transformation part for input gate, output gate, forget gate
        # and cell activation vectors need be done outside of dynamic_lstm.
        # So the output size is 4 times of gate_size.
        input_forward_proj = layers.fc(input=input_seq,
                                             size=gate_size * 4,
                                             act='tanh',
                                             bias_attr=False)
        forward, _ = layers.dynamic_lstm(
            input=input_forward_proj, size=gate_size * 4, use_peepholes=False)
        input_reversed_proj = layers.fc(input=input_seq,
                                              size=gate_size * 4,
                                              act='tanh',
                                              bias_attr=False)
        reversed, _ = layers.dynamic_lstm(
            input=input_reversed_proj,
            size=gate_size * 4,
            is_reverse=True,
            use_peepholes=False)
        encoder_out = layers.concat(input=[forward, reversed], axis = 1)
        return encoder_out

    def encoder(input_name, input_shape):
        input_ids = layers.data(
            name=input_name, shape=[1], dtype='int64', lod_level=1)
        input_embedding = layers.embedding(
            input=input_ids,
            size=[source_dict_dim, embedding_dim],
            dtype='float32',
            is_sparse=True)
        encoder_out = bi_lstm_encoder(input_seq=input_embedding, gate_size=embedding_dim)
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
	name="end_lables", shape=[1], dtype='float32', lod_level=1)

    #decode
    decode_out = layers.fc(input=p_enc, size=1, act='tanh')
    decode_out = layers.sequence_softmax(decode_out)

    #compute loss
    if args.debug == True:
        layers.Print(decode_out, message='decode_out')
        layers.Print(start_labels, message='start_labels')
    cost = layers.cross_entropy(input=decode_out, label=end_labels, soft_label=True)
    avg_cost = layers.mean(x=cost)

    feeding_list = ['q_ids', 'p_ids', "start_lables", "end_lables"]
    return avg_cost, feeding_list
