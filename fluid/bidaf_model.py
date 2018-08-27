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

    def attn_flow(q_enc, p_enc, p_ids_name):
        tag = p_ids_name + "::" 
	drnn = layers.DynamicRNN()
	with drnn.block():
	    h_cur = drnn.step_input(p_enc)
	    u_all = drnn.static_input(q_enc)
	    h_expd = layers.sequence_expand(x=h_cur, y=u_all)
	    s_t_ = layers.elementwise_mul(x=u_all, y=h_expd, axis=0)
	    s_t1 = layers.reduce_sum(input=s_t_, dim=1) 
	    s_t = layers.sequence_softmax(input=s_t1)
	    u_expr = layers.elementwise_mul(x=u_all, y=s_t, axis=0)
	    u_expr = layers.sequence_pool(input=u_expr, pool_type='sum') 
	    
     
	    if args.debug == True:
		'''
		layers.Print(h_expd, message='h_expd')
		layers.Print(h_cur, message='h_cur')
		layers.Print(u_all, message='u_all')
		layers.Print(s_t, message='s_t')
		layers.Print(s_t_, message='s_t_')
		layers.Print(u_expr, message='u_expr')
		'''
	    drnn.output(u_expr)
	    
	U_expr = drnn() 
	#'''
	drnn2 = layers.DynamicRNN()
	with drnn2.block():
	    h_cur = drnn2.step_input(p_enc)
	    u_all = drnn2.static_input(q_enc)
	    h_expd = layers.sequence_expand(x=h_cur, y=u_all)
	    s_t_ = layers.elementwise_mul(x=u_all, y=h_expd, axis=0)
	    s_t2 = layers.reduce_sum(input=s_t_, dim=1, keep_dim=True) 
	    b_t = layers.sequence_pool(input=s_t2, pool_type='max') 
	   
     
	    if args.debug == True:
		'''
		layers.Print(s_t2, message='s_t2')
		layers.Print(b_t, message='b_t')
		'''
	    drnn2.output(b_t)
	b = drnn2()
	b_norm = layers.sequence_softmax(input=b) 
	h_expr = layers.elementwise_mul(x=p_enc, y=b_norm, axis=0)
	h_expr = layers.sequence_pool(input=h_expr, pool_type='sum') 
	    

	H_expr = layers.sequence_expand(x=h_expr, y=p_enc)
	H_expr = layers.lod_reset(x=H_expr, y=p_enc) 
	h_u = layers.elementwise_mul(x=H_expr, y=U_expr, axis=0)
	h_h = layers.elementwise_mul(x=H_expr, y=p_enc, axis=0) 
	
	g = layers.concat(input=[H_expr, U_expr, h_u, h_h], axis = 1) 

        #fusion
	m = bi_lstm_encoder(input_seq=g, gate_size=embedding_dim) 
	if args.debug == True:
	    layers.Print(U_expr, message=tag + 'U_expr')
	    layers.Print(H_expr, message=tag + 'H_expr')
	    layers.Print(b, message=tag + 'b')
	    layers.Print(b_norm, message=tag + 'b_norm')
	    layers.Print(g, message=tag +'g')
	    layers.Print(m, message=tag + 'm')
	    layers.Print(h_h, message=tag + 'h_h')
	    layers.Print(q_enc, message=tag + 'q_enc')
	    layers.Print(p_enc, message=tag + 'p_enc')
       
        return m, g
    
    def lstm_step(x_t, hidden_t_prev, cell_t_prev, size):
	def linear(inputs):
	    return layers.fc(input=inputs, size=size, bias_attr=True)

	forget_gate = layers.sigmoid(x=linear([hidden_t_prev, x_t]))
	input_gate = layers.sigmoid(x=linear([hidden_t_prev, x_t]))
	output_gate = layers.sigmoid(x=linear([hidden_t_prev, x_t]))
	cell_tilde = layers.tanh(x=linear([hidden_t_prev, x_t]))

	cell_t = layers.sums(input=[
	    layers.elementwise_mul(
		x=forget_gate, y=cell_t_prev), layers.elementwise_mul(
		    x=input_gate, y=cell_tilde)
	])

	hidden_t = layers.elementwise_mul(
	    x=output_gate, y=layers.tanh(x=cell_t))

	return hidden_t, cell_t 
    
    #point network
    def point_network_decoder(p_vec, q_vec, decoder_size):
        random_attn = layers.gaussian_random(shape=[1, decoder_size])
	random_attn = layers.sequence_expand(x=random_attn, y=q_vec)
        random_attn = layers.fc(input=random_attn, size=decoder_size, act=None)
        U = layers.fc(input=q_vec,
			    size=decoder_size,
			    act=None) + random_attn
        U = layers.tanh(U)
        
        logits = layers.fc(input=U,
			    size=1,
			    act=None)
        scores = layers.sequence_softmax(input=logits)
	pooled_vec = layers.elementwise_mul(x=q_vec, y=scores, axis=0)
	pooled_vec = layers.sequence_pool(input=pooled_vec, pool_type='sum')

        init_state = layers.fc(input=pooled_vec,
			    size=decoder_size,
			    act=None)

        def custom_dynamic_rnn(p_vec, init_state, decoder_size):
            context = layers.fc(input=p_vec,
			    size=decoder_size,
			    act=None)

	    drnn = layers.DynamicRNN()
	    with drnn.block():
		H_s = drnn.step_input(p_vec)
		ctx = drnn.static_input(context)

		c_prev = drnn.memory(init=init_state, need_reorder=True)
		m_prev = drnn.memory(init=init_state, need_reorder=True)
		m_prev1 = layers.fc(input=m_prev, size=decoder_size, act=None)
		m_prev1 = layers.sequence_expand(x=m_prev1, y=ctx)

		Fk = ctx + m_prev1
		Fk = layers.fc(input=Fk, size=decoder_size, act='tanh')
		logits = layers.fc(input=Fk, size=1, act=None)

		scores = layers.sequence_softmax(input=logits)
		attn_ctx = layers.elementwise_mul(x=ctx, y=scores, axis=0)
		attn_ctx = layers.sequence_pool(input=attn_ctx, pool_type='sum')
		hidden_t, cell_t = lstm_step(attn_ctx, hidden_t_prev=m_prev1, cell_t_prev=c_prev, size=decoder_size)

		drnn.update_memory(ex_mem=m_prev, new_mem=hidden_t)
		drnn.update_memory(ex_mem=c_prev, new_mem=cell_t)
      
		drnn.output(scores)
	    beta = drnn()
            return beta

        fw_outputs = custom_dynamic_rnn(p_vec, init_state, decoder_size) 
        bw_outputs = custom_dynamic_rnn(p_vec, init_state, decoder_size)
       
        def sequence_slice(x, index):
            #offset = layers.fill_constant(shape=[1, args.batch_size], value=index, dtype='float32')
            #length = layers.fill_constant(shape=[1, args.batch_size], value=1, dtype='float32')
            #return layers.sequence_slice(x, offset, length)
            idx = layers.fill_constant(shape=[1], value=1, dtype='int32')
            idx.stop_gradient = True
            from paddle.fluid.layers.control_flow import lod_rank_table 
            from paddle.fluid.layers.control_flow import lod_tensor_to_array 
            from paddle.fluid.layers.control_flow import array_read 
            from paddle.fluid.layers.control_flow import array_to_lod_tensor 
            table = lod_rank_table(x, level=0)
            table.stop_gradient = True
            array = lod_tensor_to_array(x, table)
            slice_array = array_read(array=array, i=idx)
            return array_to_lod_tensor(slice_array, table)
        
        start_prob = layers.elementwise_mul(x=sequence_slice(fw_outputs, 0), y=sequence_slice(bw_outputs, 1), axis=0) / 2
        end_prob = layers.elementwise_mul(x=sequence_slice(fw_outputs, 1), y=sequence_slice(bw_outputs, 0), axis=0) / 2
        return start_prob, end_prob
 
 
    q_enc = encoder('q_ids')

    if args.single_doc:
        p_enc = encoder('p_ids')
        m, g = attn_flow(q_enc, p_enc, 'p_ids')
        
    else:
        p_ids_names = []
        ms = []
        gs = []
	for i in range(args.doc_num):
	    p_ids_name = "pids_%d" % i
	    p_ids_names.append(p_ids_name)
	    p_enc = encoder(p_ids_name)
	    
	    m_i, g_i = attn_flow(q_enc, p_enc, p_ids_name)
	    ms.append(m_i)
	    gs.append(g_i)
	    m = layers.sequence_concat(x=ms, axis = 0) 
	    g = layers.sequence_concat(x=gs, axis = 0) 
            
    if args.simple_decode:
        m2 = bi_lstm_encoder(input_seq=m, gate_size=embedding_dim)
        
        gm1 = layers.concat(input=[g, m], axis = 1) 
        gm2 = layers.concat(input=[g, m2], axis = 1) 
        start_prob = layers.fc(input=gm1, size=1, act='softmax')
        end_prob = layers.fc(input=gm2, size=1, act='softmax')
    else:

	p_vec = layers.sequence_concat(x=m, axis = 0) 
	q_vec = bi_lstm_encoder(input_seq=q_enc, gate_size=embedding_dim)
        start_prob, end_prob = point_network_decoder(p_vec=p_vec, q_vec=q_vec, decoder_size = decoder_size)

    start_prob = layers.sequence_softmax(start_prob)
    end_prob = layers.sequence_softmax(end_prob)

    pred = layers.concat(input=[start_prob, end_prob], axis = 0) 
    #'''
    start_labels = layers.data(
	name="start_lables", shape=[1], dtype='float32', lod_level=1)
    
    end_labels = layers.data(
	name="end_lables", shape=[1], dtype='float32', lod_level=1)
    
    label = layers.concat(input=[start_labels, end_labels], axis=0)
    label.stop_gradient = True

    #compute loss
    cost = layers.cross_entropy(input=pred, label=label, soft_label=True)
    #cost = layers.cross_entropy(input=decode_out, label=end_labels, soft_label=True)
    cost = layers.reduce_sum(cost) / args.batch_size
     
    if args.debug == True:
        layers.Print(p1, message='p1')
        layers.Print(pred, message='pred')
        layers.Print(label, message='label')
        layers.Print(start_labels, message='start_labels')
        layers.Print(cost, message='cost')
    
    if args.single_doc:
        feeding_list = ['q_ids',  "start_lables", "end_lables", 'p_ids']
    else:
        feeding_list = ['q_ids',  "start_lables", "end_lables" ] + p_ids_names
    return cost, feeding_list
