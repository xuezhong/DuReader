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

    def encoder(input_name):
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
    q_enc = encoder(q_name)
        
    p_name = 'p_ids'
    p_enc = encoder(p_name)
    
    drnn = layers.DynamicRNN()
    with drnn.block():
        h_cur = drnn.step_input(p_enc)
        u_all = drnn.static_input(q_enc)
        h_expd = layers.sequence_expand(x=h_cur, y=u_all)
        s_t_ = layers.elementwise_mul(x=u_all, y=h_expd, axis=0)
        s_t = layers.reduce_sum(input=s_t_, dim=1) 
        s_t = layers.sequence_softmax(input=s_t)
        u_expr = layers.elementwise_mul(x=u_all, y=s_t, axis=0)
        u_expr = layers.sequence_pool(input=u_expr, pool_type='sum') 
        
 
        if args.debug == True:
            layers.Print(h_expd, message='h_expd')
            layers.Print(h_cur, message='h_cur')
            layers.Print(u_all, message='u_all')
            layers.Print(s_t, message='s_t')
            layers.Print(s_t_, message='s_t_')
            layers.Print(u_expr, message='u_expr')
        drnn.output(u_expr)
        
    u_expr_ = drnn() 
    
    drnn2 = layers.DynamicRNN()
    with drnn2.block():
        h_cur = drnn2.step_input(p_enc)
        u_all = drnn2.static_input(q_enc)
        h_expd = layers.sequence_expand(x=h_cur, y=u_all)
        s_t_ = layers.elementwise_mul(x=u_all, y=h_expd, axis=0)
        s_t2 = layers.reduce_sum(input=s_t_, dim=1, keep_dim=True) 
        b_t = layers.sequence_pool(input=s_t2, pool_type='max') 
       
 
        if args.debug == True:
            layers.Print(s_t2, message='s_t2')
            layers.Print(b_t, message='b_t')
        drnn2.output(b_t)
    b_ = drnn2()
    b_all = layers.sequence_softmax(input=b_) 
    h_expr = layers.elementwise_mul(x=p_enc, y=b_all, axis=0)
    h_expr = layers.sequence_pool(input=h_expr, pool_type='sum') 
        

    H_expr = layers.sequence_expand(x=h_expr, y=p_enc)
    H_expr = layers.lod_reset(x=H_expr, y=p_enc) 
    h_u_ = layers.elementwise_mul(x=H_expr, y=u_expr_, axis=0)
    h_h_ = layers.elementwise_mul(x=H_expr, y=p_enc, axis=0) 
    
    g_ = layers.concat(input=[H_expr, u_expr_, h_u_, h_h_], axis = 1) 

    m1 = bi_lstm_encoder(input_seq=g_, gate_size=embedding_dim) 
    m2 = bi_lstm_encoder(input_seq=m1, gate_size=embedding_dim)

    
    p1 = layers.concat(input=[g_, m1], axis = 1) 
    p2 = layers.concat(input=[g_, m2], axis = 1) 

    p1 = layers.fc(input=p1, size=1, act='softmax')
    p1 = layers.sequence_softmax(p1)

    p2 = layers.fc(input=p2, size=1, act='softmax')
    p2 = layers.sequence_softmax(p2)



    #attention
    #shape(batch_size, seq_len, dim*2)
    u = layers.reshape(x=q_enc, shape = [-1, args.max_q_len, embedding_dim*2])
    h = layers.reshape(x=p_enc, shape = [-1, args.max_p_len, embedding_dim*2])
 

    #shape(max_p_len, max_q_len) 
    s = layers.matmul(x=h, y=u, transpose_y=True)    
    
    #shape(max_p_len, max_q_len) 
    a = layers.softmax(input=s)
   
    #shape(batch_size, max_p_len, dim*2) 
    u_exp = layers.matmul(x=a, y=u)

    # shape(max_p_len)
    b = layers.reduce_max(s, dim=2)
    # shape(max_p_len)
    b = layers.softmax(b) 
    #shape(batch_size, dim*2)
     
    h_ = layers.transpose(h, perm=[0,2,1])
    #shape(batch_size, 2*dim)
    h_exp_ = layers.matmul(x=h_, y = b, transpose_y=True)
    #shape(batch_size, max_p_len, 2*dim)
    h_exp_ = layers.sequence_expand(x=h_exp_, y=p_enc)
    h_exp = u_exp
  
    #shape(batch_size, max_p_len, 8*dim)
    h_u = layers.elementwise_mul(x=h, y=u_exp, axis = 0)
    h_h = layers.elementwise_mul(x=h, y=h_exp, axis = 0)
    g = layers.concat(input=[h, u_exp, h_u, h_h], axis = 2) 
  
    #m1 = bi_lstm_encoder(input_seq=g_l, gate_size=embedding_dim) 
    #m2 = bi_lstm_encoder(input_seq=m1, gate_size=embedding_dim)
    start_labels = layers.data(
	name="start_lables", shape=[1], dtype='float32', lod_level=1)
    
    end_labels = layers.data(
	name="end_lables", shape=[1], dtype='float32', lod_level=1)

    #decode
    decode_out = layers.fc(input=p_enc, size=1, act='tanh')
    decode_out = layers.sequence_softmax(decode_out)

    #compute loss
    if args.debug == True:
        layers.Print(s, message='s')
        #layers.Print(h, message='h')
        layers.Print(u, message='u')
        layers.Print(u_exp, message='u_exp')
        layers.Print(u_expr_, message='u_expr_')
        layers.Print(h_exp, message='h_exp')
        layers.Print(h_expr, message='h_expr')
        layers.Print(H_expr, message='H_expr')
        layers.Print(a, message='a')
        layers.Print(b_, message='b_')
        layers.Print(b_all, message='b_all')
        layers.Print(g, message='g')
        layers.Print(g_, message='g_')
        layers.Print(m1, message='m1')
        layers.Print(p1, message='p1')
        layers.Print(m2, message='m2')
        layers.Print(h_h, message='h_h')
        layers.Print(q_enc, message='q_enc')
        layers.Print(p_enc, message='p_enc')
        layers.Print(decode_out, message='decode_out')
        layers.Print(start_labels, message='start_labels')
    cost = layers.cross_entropy(input=decode_out, label=end_labels, soft_label=True)
    avg_cost = layers.mean(x=cost)

    feeding_list = ['q_ids', 'p_ids', "start_lables", "end_lables"]
    return avg_cost, feeding_list
