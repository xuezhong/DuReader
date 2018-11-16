# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module implements the reading comprehension models based on:
1. the BiDAF algorithm described in https://arxiv.org/abs/1611.01603
2. the Match-LSTM algorithm described in https://openreview.net/pdf?id=B1-q5Pqxl
Note that we use Pointer Network for the decoding stage of both models.
"""

import os
import time
import logging
import json
import numpy as np
import tensorflow as tf
from utils import compute_bleu_rouge
from utils import normalize
from layers.basic_rnn import rnn
from layers.match_layer import MatchLSTMLayer
from layers.match_layer import AttentionFlowMatchLayer
from layers.pointer_net import PointerNetDecoder

import tensorflow.contrib as tc
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python import debug as tf_debug

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

name_dict = {
  'word_embedding/word_embeddings':1,
  'question_encoding/bidirectional_rnn/fw':2,
  'question_encoding/bidirectional_rnn/bw':3,
  'passage_encoding/bidirectional_rnn/fw':4,
  'passage_encoding/bidirectional_rnn/bw':5,
  'passage_encoding/bidirectional_rnn/bw/multi_rnn_cell/cell_0/lstm_cell/kernel':51,
  'passage_encoding/bidirectional_rnn/bw/multi_rnn_cell/cell_0/lstm_cell/bias':52,
  'fusion/bidirectional_rnn/fw':6,
  'fusion/bidirectional_rnn/bw':7,
  'pn_decoder':8,
  'pn_decoder/fw':81,
  'pn_decoder/bw':82,
  'pn_decoder/attend_pooling':83,
  'pn_decoder/fully_connected':84,
  'pn_decoder/random_attn_vector':85,
  'pred_prob':9
}

slot_dict = {}
def init_slot():
    global slot_dict
    slot_dict = {}

def name2slot(para_name):
    res = []
    for key_name in name_dict.keys():
        if para_name.find(key_name) >= 0:
            res.append(name_dict[key_name])
    return res

def update_slot(slots, p_array):
    p_mean, p_max, p_min, p_num = p_array.mean(), p_array.max(), p_array.min(), np.prod(p_array.shape)
    for slot in slots:
        if slot in slot_dict:
            s_mean, s_max, s_min, s_num = slot_dict[slot]
            s_mean = (s_mean*s_num + p_mean*p_num) / (p_num + s_num)
            s_max = max(s_max, p_max)
            s_min = min(s_min, p_min)
            s_num = p_num + s_num
            slot_dict[slot] = [s_mean, s_max, s_min, s_num]
        else:
            slot_dict[slot] = [p_mean, p_max, p_min, p_num]

def record_slot(logger):
    for slot in slot_dict:
        logger.info("slot:" + "\t".join([str(x) for x in [slot] + slot_dict[slot]]))

class RCModel(object):
    """
    Implements the main reading comprehension model.
    """

    def __init__(self, vocab, args):

        # logging
        self.logger = logging.getLogger("brc")

        # basic config
        self.algo = args.algo
        self.hidden_size = args.hidden_size
        self.optim_type = args.optim
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.use_dropout = args.dropout_keep_prob < 1
        self.batch_size = args.batch_size * args.max_p_num
        
        self.result_dir = args.result_dir

        # length limit
        self.max_p_num = args.max_p_num
        self.max_p_len = args.max_p_len
        self.max_q_len = args.max_q_len
        self.max_a_len = args.max_a_len
        self.simple_net = args.simple_net
        self.para_init = args.para_init
        self.init1 = args.init1
        self.init2 = args.init2
        self.debug_dev = args.debug_dev
        self.dev_interval = args.dev_interval
        self.log_interval = args.log_interval
        self.lstm_direction = args.lstm_direction
        self.detail = args.detail
        self.skip_num = args.skip_num

        # the vocab
        self.vocab = vocab
        self.shuffle = args.shuffle

        # session info
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        
        self.debug_print = args.debug_print
        self.sumary = args.sumary
        
        self._build_graph()

        # save info
        self.saver = tf.train.Saver()

        # initialize the model
        self.sess.run(tf.global_variables_initializer())
        
        if args.sumary: 
            self.train_writer = tf.summary.FileWriter('train_sumary', self.sess.graph)
        
        if args.debug:
            self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess, ui_type='curses')

    def var_print(self, tag, p_array, p_name, name):
        if isinstance(p_array,np.float32):
            p_array=np.array([p_array]) 
        if not isinstance(p_array, np.ndarray):
            p_array = p_array.values
        param_num = np.prod(p_array.shape)
        p_array3 = np.multiply(np.multiply(p_array, p_array), p_array)
        self.logger.info(tag + ": {0} ({1}),  l3={2} sum={3}  max={4}  min={5} mean={6} num={7} {8}".format(p_name, name, p_array3.sum(), p_array.sum(), p_array.max(), p_array.min(), p_array.mean(), p_array.shape, param_num))
        if self.detail:
            self.logger.info(" ".join([tag + "[", p_name, '] shape [', str(p_array.shape), ']', str(p_array)]))
 
    def print_num_of_total_parameters(self, output_detail=False, output_to_logging=False):
        if not self.debug_print:
            return
        init_slot()
        total_parameters = 0
        parameters_string = ""
        
        for variable in tf.trainable_variables():
        #for variable in tf.all_variables():

            shape = variable.get_shape()
            p_array = self.sess.run(variable.name)
            slots = name2slot(variable.name)
            if slots:
                update_slot(slots, p_array)
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
            self.var_print('para', p_array, variable.name, variable.name)
        record_slot(self.logger)
        self.logger.info("Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))

    def _build_graph(self):
        """
        Builds the computation graph with Tensorflow
        """
        start_t = time.time()
        self._setup_placeholders()
        self._embed()
        self._encode()
        self._match()
        self._fuse()
        self._decode()
        self._predict()
        self._compute_loss()
        with tf.name_scope('optim'):
            self._create_train_op()
        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))
        param_num = sum([np.prod(self.sess.run(tf.shape(v))) for v in self.all_params])
        self.logger.info('There are {} parameters in the model'.format(param_num))
        self.merged = tf.summary.merge_all()

    def _setup_placeholders(self):
        """
        Placeholders
        """
        self.p = tf.placeholder(tf.int32, [None, None])
        self.q = tf.placeholder(tf.int32, [None, None])
        self.p_length = tf.placeholder(tf.int32, [None])
        self.q_length = tf.placeholder(tf.int32, [None])
        self.start_label = tf.placeholder(tf.int32, [None])
        self.end_label = tf.placeholder(tf.int32, [None])
        self.dropout_keep_prob = tf.placeholder(tf.float32)

        self.passage_mask = tf.sequence_mask(self.p_length,
                                            maxlen=tf.shape(self.p)[1],
                                            dtype=tf.float32)
        self.question_mask = tf.sequence_mask(self.q_length,
                                            maxlen=tf.shape(self.q)[1],
                                            dtype=tf.float32)

    def _embed(self):
        """
        The embedding layer, question and passage share embeddings
        """
        with tf.device('/cpu:0'), tf.variable_scope('word_embedding'):
            self.word_embeddings = tf.get_variable(
                'word_embeddings',
                shape=(self.vocab.size(), self.vocab.embed_dim),
                initializer=tf.constant_initializer(self.vocab.embeddings),
                trainable=True
            )
            self.p_emb = tf.nn.embedding_lookup(self.word_embeddings, self.p)
            self.q_emb = tf.nn.embedding_lookup(self.word_embeddings, self.q)
            variable_summaries(self.p_emb)
            variable_summaries(self.q_emb)

    def _encode(self):
        """
        Employs two Bi-LSTMs to encode passage and question separately
        """
        init = None 
        if self.para_init:
            init_w = tf.constant_initializer(self.init1)
            init_b = tf.constant_initializer(self.init1) 
        else:
            init_w = initializers.xavier_initializer()
            init_b = tf.zeros_initializer()

        if self.simple_net in [0, 1, 4]: 
            with tf.variable_scope('passage_encoding'):
                self.sep_p_encodes = tc.layers.fully_connected(self.p_emb, num_outputs=2*self.hidden_size, activation_fn=tf.nn.tanh, weights_initializer=init_w, biases_initializer=init_b)
            with tf.variable_scope('question_encoding'):
                self.sep_q_encodes = tc.layers.fully_connected(self.q_emb, num_outputs=2*self.hidden_size, activation_fn=tf.nn.tanh, weights_initializer=init_w, biases_initializer=init_b) 
        if self.simple_net in [2, 3, 5, 7, 8]:
            with tf.variable_scope('passage_encoding'):
                self.sep_p_encodes, self.seq_p_states, self.p_r = rnn('bi-lstm', self.p_emb, self.p_length, self.hidden_size, self.init1, batch_size=self.batch_size, debug=self.para_init)
            with tf.variable_scope('question_encoding'):
                self.sep_q_encodes, self.seq_q_states, _= rnn('bi-lstm', self.q_emb, self.q_length, self.hidden_size, self.init1, batch_size=self.batch_size, debug=self.para_init)
            if self.use_dropout:
                self.sep_p_encodes = tf.nn.dropout(self.sep_p_encodes, self.dropout_keep_prob)
                self.sep_q_encodes = tf.nn.dropout(self.sep_q_encodes, self.dropout_keep_prob)
            
        #self.sep_p_encodes *= tf.expand_dims(self.passage_mask, -1)
        #self.sep_q_encodes *= tf.expand_dims(self.question_mask, -1)
        variable_summaries(self.sep_p_encodes)
        variable_summaries(self.sep_q_encodes)

    def _match(self):
        """
        The core of RC model, get the question-aware passage encoding with either BIDAF or MLSTM
        """
        if self.simple_net in [0]:
            return
        if self.algo == 'MLSTM':
            match_layer = MatchLSTMLayer(self.hidden_size)
        elif self.algo == 'BIDAF':
            match_layer = AttentionFlowMatchLayer(self.hidden_size) 
        else:
            raise NotImplementedError('The algorithm {} is not implemented.'.format(self.algo))
        self.match_p_encodes, self.sim_matrix, self.context2question_attn, self.b, self.question2context_attn = match_layer.match(self.sep_p_encodes, self.sep_q_encodes,
                                                    self.p_length, self.q_length)
        if self.use_dropout:
            self.match_p_encodes = tf.nn.dropout(self.match_p_encodes, self.dropout_keep_prob)

    def _fuse(self):
        """
        Employs Bi-LSTM again to fuse the context information after match layer
        """
        if self.simple_net in [0]:
            return

        if self.para_init:
            init_w = tf.constant_initializer(self.init1)
            init_b = tf.constant_initializer(self.init1) 
        else:
            init_w = initializers.xavier_initializer()
            init_b = tf.zeros_initializer()

        with tf.variable_scope('fusion'):
            if self.simple_net in [1, 4]:
                self.fuse_p_encodes = tc.layers.fully_connected(self.match_p_encodes, num_outputs=2*self.hidden_size, activation_fn=tf.nn.tanh, weights_initializer=init_w, biases_initializer=init_b)
            if self.simple_net in [2, 3, 8]:
                self.fuse_p_encodes, _, _= rnn('bi-lstm', self.match_p_encodes, self.p_length,
                                         self.hidden_size, self.init1, batch_size=self.batch_size, layer_num=1, debug=self.para_init)

            if self.use_dropout:
                self.fuse_p_encodes = tf.nn.dropout(self.fuse_p_encodes, self.dropout_keep_prob)

    def _decode(self):
        if self.para_init:
            init_w = tf.constant_initializer(self.init1)
            init_b = tf.constant_initializer(self.init1) 
        else:
            init_w = initializers.xavier_initializer()
            init_b = tf.zeros_initializer()

        batch_size = tf.shape(self.start_label)[0]
        with tf.variable_scope('same_question_concat'):
            if self.simple_net in [1, 2, 3, 4]:
                self.fuse_p_encodes *= tf.expand_dims(self.passage_mask, -1)
            self.sep_q_encodes *= tf.expand_dims(self.question_mask, -1)

            if self.simple_net in [0, 5, 7]:
                self.ps_enc_ = tf.reshape(
                    self.sep_p_encodes,
                    [batch_size, -1, 2 * self.hidden_size]
                )
                if self.lstm_direction in [1]:
                    self.ps_enc = self.ps_enc_[0:, 0:, 0:self.hidden_size]
                if self.lstm_direction in [2]:
                    self.ps_enc = self.ps_enc_[0:, 0:, self.hidden_size:]
                if self.lstm_direction in [3]:
                    self.ps_enc = self.ps_enc_
            self.concat_passage_mask = tf.reshape(
                self.passage_mask,
                [batch_size, -1]
            )

            if self.simple_net in [1]:
                self.m2 = tc.layers.fully_connected(self.fuse_p_encodes, num_outputs=2*self.hidden_size, activation_fn=tf.nn.tanh, weights_initializer=init_w, biases_initializer=init_b)
            if self.simple_net in [2]:
                self.m2, _, _ = rnn('bi-lstm',  self.fuse_p_encodes, self.p_length,
                                         self.hidden_size, self.init1, batch_size=self.batch_size, layer_num=1, debug=self.para_init)

            if self.simple_net in [1, 2]:
                self.concat_passage_encodes = tf.reshape(
                    self.fuse_p_encodes,
                    [batch_size, -1, 2 * self.hidden_size]
                )
                g = tf.reshape(
                    self.match_p_encodes,
                    [batch_size, -1, 8 * self.hidden_size]
                )
                m2 = tf.reshape(
                    self.m2,
                    [batch_size, -1, 2 * self.hidden_size]
                )
                with tf.variable_scope('simple_decoder'):
                    self.gm1 = tf.concat([g, self.concat_passage_encodes], -1)
                    self.gm2 = tf.concat([g, m2], -1)

            
            if self.simple_net in [3, 4, 8]:
                self.concat_passage_encodes = tf.reshape(
                    self.fuse_p_encodes,
                    [batch_size, -1, 2 * self.hidden_size]
                )
                self.no_dup_question_encodes = tf.reshape(
                    self.sep_q_encodes,
                    [batch_size, -1, 2 * self.hidden_size]
                )
                self.no_dup_question_mask = tf.reshape(
                    self.question_mask,
                    [batch_size, -1]
            )


    def _predict(self):
        if self.para_init:
            init_w = tf.constant_initializer(self.init1)
            init_b = tf.constant_initializer(self.init1) 
        else:
            init_w = initializers.xavier_initializer()
            init_b = tf.zeros_initializer()

        if self.simple_net in [0, 5, 7]:
            neg_mask = -1e9*(1. - tf.expand_dims(self.concat_passage_mask, -1))
            with tf.variable_scope('pred_prob_start'):
                if self.simple_net in [0, 5]:
                    self.start_probs_p = neg_mask + tc.layers.fully_connected(self.ps_enc, num_outputs=1, activation_fn=None, weights_initializer=init_w, biases_initializer=None)
                    self.start_probs = tf.nn.softmax(tf.keras.backend.squeeze(self.start_probs_p, -1), 1)
                if  self.simple_net in [7]:
                    self.start_probs_p = tf.reduce_sum(self.ps_enc, -1, keepdims=True)
                    self.start_probs_p = neg_mask + self.start_probs_p
                    self.start_probs = tf.nn.softmax(tf.keras.backend.squeeze(self.start_probs_p, -1), 1)
            with tf.variable_scope('pred_prob_end'):
                if self.simple_net in [0, 5]:
                    self.end_probs_p = neg_mask + tc.layers.fully_connected(self.ps_enc, num_outputs=1, activation_fn=None, weights_initializer=init_w, biases_initializer=None)
                    self.end_probs = tf.nn.softmax(tf.keras.backend.squeeze(self.end_probs_p, -1), 1)
                if  self.simple_net in [7]:
                    self.end_probs_p = tf.reduce_sum(self.ps_enc, -1, keepdims=True)
                    self.end_probs_p = neg_mask + self.end_probs_p
                    self.end_probs = tf.nn.softmax(tf.keras.backend.squeeze(self.end_probs_p, -1), 1)
        if self.simple_net in [1, 2]:
            neg_mask = -1e9*(1. - tf.expand_dims(self.concat_passage_mask, -1))
            self.start_probs = tf.nn.softmax(tf.keras.backend.squeeze(neg_mask + tc.layers.fully_connected(self.gm1, num_outputs=1, activation_fn=None, weights_initializer=init_w, biases_initializer=None),-1),1)
            self.end_probs = tf.nn.softmax(tf.keras.backend.squeeze(neg_mask + tc.layers.fully_connected(self.gm2, num_outputs=1, activation_fn=None, weights_initializer=init_w, biases_initializer=None),-1),1)          
        if self.simple_net in [8]:
            neg_mask = -1e9*(1. - tf.expand_dims(self.concat_passage_mask, -1))
            self.start_probs = tf.nn.softmax(tf.keras.backend.squeeze(neg_mask + tc.layers.fully_connected(self.concat_passage_encodes, num_outputs=1, activation_fn=None, weights_initializer=init_w, biases_initializer=None),-1),1)
            self.end_probs = tf.nn.softmax(tf.keras.backend.squeeze(neg_mask + tc.layers.fully_connected(self.concat_passage_encodes, num_outputs=1, activation_fn=None, weights_initializer=init_w, biases_initializer=None),-1),1)  
        if self.simple_net in [3, 4]:
            decoder = PointerNetDecoder(self.hidden_size, self.para_init, self.init1, self.init2)
            self.start_probs, self.end_probs, self.pn_init_state, self.pn_f0, self.pn_f1, self.pn_b0, self.pn_b1= decoder.decode(self.concat_passage_encodes,
                              self.no_dup_question_encodes,
                              self.concat_passage_mask,
                              self.no_dup_question_mask)

    def _compute_loss(self):
        """
        The loss function
        """

        def sparse_nll_loss(probs, labels, epsilon=1e-9, scope=None):
            """
            negative log likelyhood loss
            """
            with tf.name_scope(scope, "log_loss"):
                labels = tf.one_hot(labels, tf.shape(probs)[1], axis=1)
                losses = - tf.reduce_sum(labels * tf.log(probs + epsilon), 1)
            return losses

        self.start_loss = sparse_nll_loss(probs=self.start_probs, labels=self.start_label)
        self.end_loss = sparse_nll_loss(probs=self.end_probs, labels=self.end_label)
        self.all_params = tf.trainable_variables()
        self.loss = tf.reduce_mean(tf.add(self.start_loss, self.end_loss))
        if self.weight_decay > 0:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
            self.loss2 = self.loss + self.weight_decay * l2_loss
        else:
            self.loss2 = self.loss

    def _create_train_op(self):
        """
        Selects the training algorithm and creates a train operation with it
        """
        if self.optim_type == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.optim_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.optim_type == 'rprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.optim_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            raise NotImplementedError('Unsupported optimizer: {}'.format(self.optim_type))
        self.train_op = self.optimizer.minimize(self.loss2)

    def _train_epoch(self, train_batches, dropout_keep_prob, batch_size, pad_id, data):
        """
        Trains the model for a single epoch.
        Args:
            train_batches: iterable batch data for training
            dropout_keep_prob: float value indicating dropout keep probability
        """
        total_num, total_loss = 0, 0
        log_every_n_batch, n_batch_loss = self.log_interval, 0
        self.print_num_of_total_parameters(True, True)
        for bitx, batch in enumerate(train_batches, 1):
            if bitx <= self.skip_num:
                continue
            feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['question_length'],
                         self.start_label: batch['start_id'],
                         self.end_label: batch['end_id'],
                         self.dropout_keep_prob: dropout_keep_prob}
            if self.debug_print:
                if self.simple_net in [0, 5, 7]:
                    para_name='passage_encoding/bidirectional_rnn/bw/multi_rnn_cell/cell_0/lstm_cell/bias:0'.split(',')
                    para=[] 
                    var_dict = {}
                    for x in self.all_params:
                        var_dict[x.name] = x
                    for name in para_name:
                        if name in var_dict:
                            para.append(var_dict[name])
                        else:
                            self.logger.error('para {0} not a variable'.format(name))
                            exit(-1)
                        
                    grad_names='self.p_emb, self.ps_enc, self.start_probs,self.start_probs_p'.split(',')
                    grad_names = grad_names + para_name
                    grad = tf.gradients(ys=self.loss2, xs=[self.p_emb, self.ps_enc,self.start_probs, self.start_probs_p] + para)
                    
                    names = 'self.train_op, self.loss, grad, self.p_emb, self.q_emb, self.sep_p_encodes,self.p_r, self.sep_q_encodes, self.p, self.p_length, self.q, self.ps_enc, self.start_probs,self.seq_p_states, self.seq_q_states'.split(',')
                    res = self.sess.run([self.train_op, self.loss, grad, self.p_emb, self.q_emb, self.sep_p_encodes, self.p_r, self.sep_q_encodes, self.p, self.p_length, self.q, self.ps_enc, self.start_probs, self.seq_p_states, self.seq_q_states], feed_dict)
                    
                if self.simple_net in [1, 2]:
                    para_name = []
                    para = [] 
                    grad_names='self.p_emb, self.start_probs'.split(',')
                    grad_names = grad_names + para_name
                    grad = tf.gradients(ys=self.loss2, xs=[self.p_emb,self.start_probs] + para)
                    res = self.sess.run([self.train_op, self.loss, grad, self.p_length, self.q_length, self.p_emb, self.q_emb, self.sep_p_encodes, self.sep_q_encodes, self.p, self.q, self.match_p_encodes, self.fuse_p_encodes, 
                                     self.m2, self.gm1, self.gm2, self.start_probs, self.sim_matrix, self.context2question_attn, self.b, self.question2context_attn], feed_dict)
                    names = 'self.train_op, self.loss, grad, self.p_length, self.q_length, self.p_emb, self.q_emb, self.sep_p_encodes, self.sep_q_encodes, self.p, self.q, self.match_p_encodes, self.fuse_p_encodes,'\
                            ' self.m2, self.gm1, self.gm2, self.start_probs, self.sim_matrix, self.context2question_attn, self.b, self.question2context_attn'.split(',')
                if self.simple_net in [3, 4]: 
                    para_name = ['pn_decoder/random_attn_vector:0', 'word_embedding/word_embeddings:0']
                    para = [] 
                    var_dict = {}
                    for x in self.all_params:
                        var_dict[x.name] = x
                    for name in para_name:
                        if name in var_dict:
                            para.append(var_dict[name])
                        else:
                            self.logger.error('para {0} not a variable'.format(name))
                            exit(-1)

                    grad_names='self.p_emb, self.start_probs'.split(',')
                    grad_names = grad_names + para_name
                    grad = tf.gradients(ys=self.loss2, xs=[self.p_emb,self.start_probs] + para)

                    res = self.sess.run([self.train_op, self.loss, grad, self.loss2, self.p_length, self.q_length, self.p_emb, self.q_emb, self.sep_p_encodes, self.sep_q_encodes, self.p, self.q, self.match_p_encodes, self.fuse_p_encodes,self.no_dup_question_encodes, 
                         self.start_probs, self.sim_matrix, self.context2question_attn, self.b, self.question2context_attn, self.pn_init_state, self.pn_f0, self.pn_f1, self.pn_b0, self.pn_b1], feed_dict)
                    names = 'self.train_op, self.loss, grad, self.loss2, self.p_length, self.q_length, self.p_emb, self.q_emb, self.sep_p_encodes, self.sep_q_encodes, self.p, self.q, self.match_p_encodes, self.fuse_p_encodes,self.no_dup_question_encodes,'\
                            'self.start_probs, self.sim_matrix, self.context2question_attn, self.b, self.question2context_attn, self.pn_init_state, self.pn_f0, self.pn_f1, self.pn_b0, self.pn_b1'.split(',')

                if self.simple_net in [8]:
                    para_name = []
                    para = [] 
                    grad_names='self.p_emb, self.start_probs'.split(',')
                    grad_names = grad_names + para_name
                    grad = tf.gradients(ys=self.loss2, xs=[self.p_emb,self.start_probs] + para)
                    res = self.sess.run([self.train_op, self.loss, grad, self.p_length, self.q_length, self.p_emb, self.q_emb, self.sep_p_encodes, self.sep_q_encodes, self.p, self.q, self.match_p_encodes, self.fuse_p_encodes, 
                                     self.start_probs, self.sim_matrix, self.context2question_attn, self.b, self.question2context_attn], feed_dict)
                    names = 'self.train_op, self.loss, grad, self.p_length, self.q_length, self.p_emb, self.q_emb, self.sep_p_encodes, self.sep_q_encodes, self.p, self.q, self.match_p_encodes, self.fuse_p_encodes,'\
                            'self.start_probs, self.sim_matrix, self.context2question_attn, self.b, self.question2context_attn'.split(',')


                loss = res[1]
                grad_res = res[2]
                for i in range(3, len(res)):
                    p_name = names[i]
                    p_array = res[i]
                    self.var_print('var', p_array, p_name, p_name)
                for i in range(0, len(grad_res)):
                    p_name = grad_names[i]
                    p_array = grad_res[i]
                    self.var_print('grad', p_array, p_name, p_name)

            elif self.sumary:
                merged, loss = self.sess.run([self.merged, self.loss2], feed_dict)
                self.train_writer.add_summary(merged, bitx)
            else:
                _, loss = self.sess.run([self.train_op, self.loss2], feed_dict)
            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])
            n_batch_loss += loss
            if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                self.print_num_of_total_parameters(True, True)
                self.logger.info('Average loss from batch {} to {} is {}'.format(
                    bitx - log_every_n_batch + 1, bitx, "%.10f"%(n_batch_loss / log_every_n_batch)))
                n_batch_loss = 0
            if (data.dev_set is not None) and self.dev_interval > 0 and (bitx % self.dev_interval == 0):
                eval_batches = data.gen_mini_batches('dev', batch_size, pad_id, shuffle=False)
                eval_loss, bleu_rouge = self.evaluate(eval_batches, result_dir=self.result_dir, result_prefix='dev.predicted', result_name='%d,%d'%(0, bitx))
                self.logger.info('Dev eval loss {}'.format(eval_loss))
                self.logger.info('Dev eval result: {}'.format(bleu_rouge))

            if self.debug_print and bitx >= 8:
                exit()
        return 1.0 * total_loss / total_num

    def train(self, data, epochs, batch_size, save_dir, save_prefix,
              dropout_keep_prob=1.0, evaluate=True):
        """
        Train the model with data
        Args:
            data: the BRCDataset class implemented in dataset.py
            epochs: number of training epochs
            batch_size:
            save_dir: the directory to save the model
            save_prefix: the prefix indicating the model type
            dropout_keep_prob: float value indicating dropout keep probability
            evaluate: whether to evaluate the model on test set after each epoch
        """
        pad_id = self.vocab.get_id(self.vocab.pad_token)
        max_bleu_4 = 0
        self.print_num_of_total_parameters(True, True)
        
        if self.debug_dev:
            eval_batches = data.gen_mini_batches('dev', batch_size, pad_id, shuffle=False)
            eval_loss, bleu_rouge = self.evaluate(eval_batches)
            self.logger.info('Dev eval loss {}'.format(eval_loss))
            self.logger.info('Dev eval result: {}'.format(bleu_rouge))

            if bleu_rouge['Bleu-4'] > max_bleu_4:
                self.save(save_dir, save_prefix)
                max_bleu_4 = bleu_rouge['Bleu-4']
            exit()

        for epoch in range(1, epochs + 1):
            self.logger.info('Training the model for epoch {}'.format(epoch))
            train_batches = data.gen_mini_batches('train', batch_size, pad_id, shuffle=self.shuffle)
            train_loss = self._train_epoch(train_batches, dropout_keep_prob, batch_size, pad_id, data)
            self.logger.info('Average train loss for epoch {} is {}'.format(epoch, train_loss))

            if evaluate:
                self.logger.info('Evaluating the model after epoch {}'.format(epoch))
                if data.dev_set is not None:
                    eval_batches = data.gen_mini_batches('dev', batch_size, pad_id, shuffle=False)
                    eval_loss, bleu_rouge = self.evaluate(eval_batches, result_dir=self.result_dir, result_prefix='dev.predicted')
                    self.logger.info('Dev eval loss {}'.format(eval_loss))
                    self.logger.info('Dev eval result: {}'.format(bleu_rouge))

                    if bleu_rouge['Bleu-4'] > max_bleu_4:
                        self.save(save_dir, save_prefix)
                        max_bleu_4 = bleu_rouge['Bleu-4']
                else:
                    self.logger.warning('No dev set is loaded for evaluation in the dataset!')
            else:
                self.save(save_dir, save_prefix + '_' + str(epoch))

    def evaluate(self, eval_batches, result_dir=None, result_prefix=None, result_name='', save_full_info=False):
        """
        Evaluates the model performance on eval_batches and results are saved if specified
        Args:
            eval_batches: iterable batch data
            result_dir: directory to save predicted answers, answers will not be saved if None
            result_prefix: prefix of the file for saving predicted answers,
                           answers will not be saved if None
            save_full_info: if True, the pred_answers will be added to raw sample and saved
        """
        pred_answers, ref_answers = [], []
        total_loss, total_num = 0, 0
        n_batch_loss = 0.0
        n_batch = 0
        for b_itx, batch in enumerate(eval_batches, 1):
            feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['question_length'],
                         self.start_label: batch['start_id'],
                         self.end_label: batch['end_id'],
                         self.dropout_keep_prob: 1.0}
            if self.debug_print:
                if self.simple_net in [0, 5]:
                    res = self.sess.run([self.loss, self.p_emb, self.q_emb, self.sep_p_encodes, self.sep_q_encodes, self.p, self.q, self.start_probs], feed_dict)
                    names = 'self.loss, self.p_emb, self.q_emb, self.sep_p_encodes, self.sep_q_encodes, self.p, self.q, self.start_probs'.split(',')
                if self.simple_net in [1, 2]: 
                    res = self.sess.run([self.loss, self.p_length, self.q_length, self.p_emb, self.q_emb, self.sep_p_encodes, self.sep_q_encodes, self.p, self.q, self.match_p_encodes, self.fuse_p_encodes, 
                                     self.gm1, self.gm2, self.start_probs, self.sim_matrix, self.context2question_attn, self.b, self.question2context_attn], feed_dict)
                    names = 'self.loss, self.p_length, self.q_length, self.p_emb, self.q_emb, self.sep_p_encodes, self.sep_q_encodes, self.p, self.q, self.match_p_encodes, self.fuse_p_encodes, \
                         self.gm1, self.gm2, self.start_probs, self.sim_matrix, self.context2question_attn, self.b, self.question2context_attn'.split(',')
                if self.simple_net in [3, 4]: 
                    res = self.sess.run([self.loss, self.start_probs, self.end_probs, self.loss2, self.start_label, self.end_label, self.p_length, self.q_length, self.p_emb, self.q_emb, self.sep_p_encodes, self.sep_q_encodes, self.p, self.q, self.match_p_encodes, self.fuse_p_encodes, 
                        self.sim_matrix, self.context2question_attn, self.b, self.question2context_attn, self.pn_init_state, self.pn_f0, self.pn_f1, self.pn_b0, self.pn_b1], feed_dict)
                    names = 'self.loss, self.start_probs, self.end_probs, self.loss2, self.start_label, self.end_label, self.p_length, self.q_length, self.p_emb, self.q_emb, self.sep_p_encodes, self.sep_q_encodes, self.p, self.q, self.match_p_encodes, self.fuse_p_encodes, \
                         self.sim_matrix, self.context2question_attn, self.b, self.question2context_attn, self.pn_init_state, self.pn_f0, self.pn_f1, self.pn_b0, self.pn_b1'.split(',')

                loss, start_probs, end_probs = res[0:3]
                for i in range(1, len(res)):
                    p_name = names[i]
                    p_array = res[i]
                    self.var_print('var', p_array, p_name, p_name)
            else:
                start_probs, end_probs, loss = self.sess.run([self.start_probs,
                                                          self.end_probs, self.loss], feed_dict)
            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])
            n_batch_loss = loss * len(batch['raw_data'])
            n_batch += len(batch['raw_data'])
            if self.log_interval > 0 and b_itx % self.log_interval == 0:
                #self.print_num_of_total_parameters(True, True)
                self.logger.info('Average dev loss from batch {} to {} is {}'.format(
                    b_itx - self.log_interval + 1, b_itx, "%.10f"%(n_batch_loss / n_batch)))
                n_batch_loss = 0.0
                n_batch = 0

            padded_p_len = len(batch['passage_token_ids'][0])
            for sample, start_prob, end_prob in zip(batch['raw_data'], start_probs, end_probs):

                best_answer, best_span = self.find_best_answer(sample, start_prob, end_prob, padded_p_len)
                if save_full_info:
                    sample['pred_answers'] = [best_answer]
                    pred_answers.append(sample)
                else:
                    pred = {'question_id': sample['question_id'],
                                         'question_type': sample['question_type'],
                                         'answers': [best_answer],
                                         'entity_answers': [[]],
                                         'yesno_answers': [best_span]}
                    pred_answers.append(pred)
                    if self.debug_print:
                        self.logger.info('pred=' + json.dumps(pred, ensure_ascii=False))
                if 'answers' in sample:
                    ref = {'question_id': sample['question_id'],
                                         'question_type': sample['question_type'],
                                         'answers': sample['answers'],
                                         'entity_answers': [[]],
                                         'yesno_answers': [best_span]}
                    ref_answers.append(ref)
                    if self.debug_print:
                        self.logger.info('ref=' + json.dumps(ref, ensure_ascii=False))

        if result_dir is not None and result_prefix is not None:
            result_file = os.path.join(result_dir, result_prefix + result_name + '.json')
            with open(result_file, 'w') as fout:
                for pred_answer in pred_answers:
                    fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')

            self.logger.info('Saving {} results to {}'.format(result_prefix, result_file))
            #exit()

        # this average loss is invalid on test set, since we don't have true start_id and end_id
        ave_loss = 1.0 * total_loss / total_num
        # compute the bleu and rouge scores if reference answers is provided
        if len(ref_answers) > 0:
            pred_dict, ref_dict = {}, {}
            for pred, ref in zip(pred_answers, ref_answers):
                question_id = ref['question_id']
                if len(ref['answers']) > 0:
                    pred_dict[question_id] = normalize(pred['answers'])
                    ref_dict[question_id] = normalize(ref['answers'])
            bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)
        else:
            bleu_rouge = None
        return ave_loss, bleu_rouge

    def find_best_answer(self, sample, start_prob, end_prob, padded_p_len):
        """
        Finds the best answer for a sample given start_prob and end_prob for each position.
        This will call find_best_answer_for_passage because there are multiple passages in a sample
        """
        best_p_idx, best_span, best_score = None, None, 0
        for p_idx, passage in enumerate(sample['passages']):
            if p_idx >= self.max_p_num:
                continue
            passage_len = min(self.max_p_len, len(passage['passage_tokens']))
            answer_span, score = self.find_best_answer_for_passage(
                start_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                end_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                passage_len)
            if score > best_score:
                best_score = score
                best_p_idx = p_idx
                best_span = answer_span
        if best_p_idx is None or best_span is None:
            best_answer = ''
        else:
            best_answer = ''.join(
                sample['passages'][best_p_idx]['passage_tokens'][best_span[0]: best_span[1] + 1])
        return best_answer, best_span

    def find_best_answer_for_passage(self, start_probs, end_probs, passage_len=None):
        """
        Finds the best answer with the maximum start_prob * end_prob from a single passage
        """
        if passage_len is None:
            passage_len = len(start_probs)
        else:
            passage_len = min(len(start_probs), passage_len)
        best_start, best_end, max_prob = -1, -1, 0
        for start_idx in range(passage_len):
            for ans_len in range(self.max_a_len):
                end_idx = start_idx + ans_len
                if end_idx >= passage_len:
                    continue
                prob = start_probs[start_idx] * end_probs[end_idx]
                if prob > max_prob:
                    best_start = start_idx
                    best_end = end_idx
                    max_prob = prob
        return (best_start, best_end), max_prob

    def save(self, model_dir, model_prefix):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model saved in {}, with prefix {}.'.format(model_dir, model_prefix))

    def restore(self, model_dir, model_prefix):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model restored from {}, with prefix {}'.format(model_dir, model_prefix))
