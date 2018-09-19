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
This module implements the joint rc model.
"""

import os
import json
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
from heapq import nlargest
from rc_model import RCModel
from utils import compute_bleu_rouge
from utils import normalize


class MTRCModel(RCModel):
    def __init__(self, vocab, args):
        self.use_boundary_loss = True
        self.use_content_loss = True
        self.use_verif_loss = True
        super(MTRCModel, self).__init__(vocab, args)

    def _build_graph(self):
        """
        Builds the computation graph with Tensorflow
        """
        self._setup_placeholders()
        self._embed()
        self.p_emb = tf.concat([self.p_emb, tf.expand_dims(self.em, -1)], -1)
        self._encode()
        self._match()
        self._fuse()

        with tf.variable_scope('boundary'):
            self._decode()
        with tf.variable_scope('content'):
            self._content()
        with tf.variable_scope('verif'):
            self._verify()

        self._compute_loss()

    def _setup_placeholders(self):
        self.p = tf.placeholder(tf.int32, [None, None])
        self.q = tf.placeholder(tf.int32, [None, None])
        self.p_length = tf.placeholder(tf.int32, [None])
        self.q_length = tf.placeholder(tf.int32, [None])
        self.start_label = tf.placeholder(tf.int32, [None])
        self.end_label = tf.placeholder(tf.int32, [None])
        self.para_label = tf.placeholder(tf.int32, [None])
        self.content_label = tf.placeholder(tf.float32, [None, None])
        self.dropout_keep_prob = tf.placeholder(tf.float32)
        self.em = tf.placeholder(tf.float32, [None, None])

        self.passage_mask = tf.sequence_mask(self.p_length,
                                            maxlen=tf.shape(self.p)[1],
                                            dtype=tf.float32)
        self.question_mask = tf.sequence_mask(self.q_length,
                                            maxlen=tf.shape(self.q)[1],
                                            dtype=tf.float32)


    def _content(self):
        self.para_content_logit = tc.layers.fully_connected(
            tc.layers.fully_connected(self.fuse_p_encodes, self.hidden_size, tf.nn.relu, scope='content_fc'),
            # self.fuse_passage_encodes,
            1, activation_fn=None, scope='content_score'
        )
        self.para_content_score = tf.nn.sigmoid(self.para_content_logit)
        self.para_content_score *= tf.expand_dims(self.passage_mask, -1)

        self.concat_content_score = tf.reshape(self.para_content_score,
                                               [tf.shape(self.start_label)[0], -1])

    def _verify(self):
        para_content_logit = self.para_content_logit -1e9*(1. - tf.expand_dims(self.passage_mask, -1))
        self.para_content_dist = tf.nn.softmax(para_content_logit, 1)

        self.passage_ans_rep = tf.reduce_sum(self.fuse_p_encodes * self.para_content_dist, 1)
        self.reshaped_passage_ans_rep = tf.reshape(self.passage_ans_rep,
                                                   [tf.shape(self.content_label)[0],
                                                    -1, self.passage_ans_rep.shape[-1].value])
        self.ans_sim_mat = tf.matmul(self.reshaped_passage_ans_rep,
                                     self.reshaped_passage_ans_rep, transpose_b=True)
        self.ans_sim_mat *= (1 - tf.expand_dims(tf.eye(tf.shape(self.ans_sim_mat)[1]), 0))
        self.ans_sim_mat = tf.nn.softmax(self.ans_sim_mat, -1)
        self.collected_ans_evid_rep = tf.matmul(self.ans_sim_mat, self.reshaped_passage_ans_rep)

        self.collected_ans_evid_rep = tf.reshape(self.collected_ans_evid_rep,
                                                 [-1, self.collected_ans_evid_rep.shape[-1].value])

        self.ans_verif_logit = tc.layers.fully_connected(tf.concat([self.passage_ans_rep, self.collected_ans_evid_rep,
                                                                    self.passage_ans_rep * self.collected_ans_evid_rep,
                                                                    # self.passage_ans_rep - self.collected_ans_evid_rep
                                                                    ], -1),
                                                         1, activation_fn=None, scope='verify_score_fc')
        self.reshaped_ans_verif_logit = tf.reshape(self.ans_verif_logit, [tf.shape(self.content_label)[0], -1])
        self.reshaped_ans_verif_score = tf.nn.softmax(self.reshaped_ans_verif_logit, 1)

    def _compute_loss(self):
        def sparse_nll_loss(probs, labels, epsilon=1e-9, scope=None):
            """negative log likelyhood loss"""
            with tf.name_scope(scope, "log_loss"):
                labels = tf.one_hot(labels, tf.shape(probs)[1], axis=1)
                losses = - tf.reduce_sum(labels * tf.log(probs + epsilon), 1)
            return losses

        def musk_ce_loss(probs, labels, musk, epsilon=1e-9, scope=None):
            """cross entropy loss"""
            with tf.name_scope(scope, "log_loss"):
                all_losses = labels * tf.log(probs + epsilon) + (1 - labels) * tf.log(1 - probs + epsilon)
                all_losses *= musk
                losses = - tf.reduce_sum(all_losses, -1) / (tf.reduce_sum(musk, -1) + epsilon)
                # losses = - tf.reduce_mean(labels * tf.log(probs + epsilon)
                #                           + (1 - labels) * tf.log(1 - probs + epsilon), -1)
            return losses

        def ce_loss(probs, labels, epsilon=1e-9, scope=None):
            """cross entropy loss"""
            with tf.name_scope(scope, "log_loss"):
                losses = - tf.reduce_mean(labels * tf.log(probs + epsilon)
                                          + (1 - labels) * tf.log(1 - probs + epsilon), -1)
            return losses

        self.start_loss = sparse_nll_loss(probs=self.start_probs, labels=self.start_label)
        self.end_loss = sparse_nll_loss(probs=self.end_probs, labels=self.end_label)
        self.boundary_loss = tf.reduce_mean(self.start_loss + self.end_loss)
        # content_musk = tf.reshape(
        #     tf.sign(tf.abs(self.placeholders['p'] - self.vocab.get_id(self.vocab.pad_token))),
        #     [tf.shape(self.placeholders['content_label'])[0],self.batch_size, -1]
        # )
        # content_musk = tf.cast(content_musk, tf.float32)
        self.content_loss = tf.reduce_mean(ce_loss(self.concat_content_score, self.content_label), 0)
        self.verif_loss = tf.reduce_mean(sparse_nll_loss(self.reshaped_ans_verif_score, self.para_label))
        self.loss = 0
        if self.use_boundary_loss:
            self.loss += 0.4 * self.boundary_loss
        if self.use_content_loss:
            self.loss += 0.3 * self.content_loss
        if self.use_verif_loss:
            self.loss += 0.3 * self.verif_loss

        if self.weight_decay > 0:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name])
            self.loss += self.weight_decay * l2_loss

    def _train_epoch(self, train_batches, print_every_n_batch=0, dropout_keep_prob=None):
        total_num, total_loss = 0, 0
        log_every_n_batch, n_batch_loss = 50, 0
        for bitx, batch in enumerate(train_batches, 1):
            feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['question_length'],
                         self.start_label: batch['start_id'],
                         self.end_label: batch['end_id'],
                         self.dropout_keep_prob: dropout_keep_prob,
                         self.em: batch['exact_match']}
            batch_size = len(batch['start_id'])
            padded_p_len = len(batch['passage_token_ids'][0])
            padded_p_num = len(batch['passage_token_ids']) / batch_size
            para_ids = []
            for start_id in batch['start_id']:
                para_ids.append(start_id // padded_p_len)
            feed_dict[self.para_label] = para_ids
            content_label = np.zeros([batch_size, padded_p_num * padded_p_len], dtype=int)
            for s_idx, (start_id, end_id) in enumerate(zip(batch['start_id'], batch['end_id'])):
                content_label[s_idx, start_id: end_id+1] = 1
            feed_dict[self.content_label] = content_label
            _, loss = self.sess.run([self.train_op, self.loss], feed_dict)
            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])
            n_batch_loss += loss
            if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                self.logger.info('Average loss from batch {} to {} is {}'.format(
                    bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch))
                n_batch_loss = 0
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
        for epoch in range(1, epochs + 1):
            self.logger.info('Training the model for epoch {}'.format(epoch))
            train_batches = data.gen_mini_batches('train', batch_size, pad_id, shuffle=True)
            train_loss = self._train_epoch(train_batches, print_every_n_batch=1, dropout_keep_prob=dropout_keep_prob)
            self.logger.info('Average train loss for epoch {} is {}'.format(epoch, train_loss))

            if evaluate:
                self.logger.info('Evaluating the model after epoch {}'.format(epoch))
                if data.dev_set is not None:
                    eval_batches = data.gen_mini_batches('dev', batch_size, pad_id, shuffle=False)
                    if self.use_ema:
                        self.save(save_dir, 'ema_temp')
                        with self.ema_test_graph.as_default():
                            if self.ema_test_model is None:
                                self.ema_test_model = MTRCModel(self.vocab, self.args)
                            self.ema_test_model.restore(save_dir, 'ema_temp')
                            eval_loss, bleu_rouge = self.evaluate(eval_batches)
                    else:
                        eval_loss, bleu_rouge = self.evaluate(eval_batches)
                    self.logger.info('Dev eval loss {}'.format(eval_loss))
                    self.logger.info('Dev eval result: {}'.format(bleu_rouge))

                    if bleu_rouge['Bleu-4'] > max_bleu_4:
                        self.save(save_dir, save_prefix)
                        max_bleu_4 = bleu_rouge['Bleu-4']
                else:
                    self.logger.warning('No dev set is loaded for evaluation in the dataset!')
            else:
                self.save(save_dir, save_prefix + '_' + str(epoch))

    def evaluate(self, eval_batches, result_dir=None, result_prefix=None, save_full_info=False):
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
        for b_itx, batch in enumerate(eval_batches):
            feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['question_length'],
                         self.start_label: batch['start_id'],
                         self.end_label: batch['end_id'],
                         self.dropout_keep_prob: 1.0,
                         self.em: batch['exact_match']}
            batch_size = len(batch['start_id'])
            padded_p_len = len(batch['passage_token_ids'][0])
            padded_p_num = len(batch['passage_token_ids']) / batch_size
            para_ids = []
            for start_id in batch['start_id']:
                para_ids.append(start_id // padded_p_len)
            feed_dict[self.para_label] = para_ids
            content_label = np.zeros([batch_size, padded_p_num * padded_p_len], dtype=int)
            for s_idx, (start_id, end_id) in enumerate(zip(batch['start_id'], batch['end_id'])):
                content_label[s_idx, start_id: end_id+1] = 1
            feed_dict[self.content_label] = content_label
            start_probs, end_probs, content_scores, verif_scores, loss = self.sess.run([self.start_probs, self.end_probs,
                                                                                        self.concat_content_score, self.reshaped_ans_verif_score,
                                                                                        self.loss], feed_dict)

            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])

            padded_p_len = len(batch['passage_token_ids'][0])
            for s_idx, sample in enumerate(batch['raw_data']):
                start_prob = start_probs[s_idx]
                end_prob = end_probs[s_idx]
                content_score = content_scores[s_idx]
                verif_score = verif_scores[s_idx]
                best_answer = self.find_best_answer_with_verif(sample, start_prob, end_prob,
                                                               content_score, verif_score, padded_p_len)
                if save_full_info:
                    sample['pred_answers'] = [best_answer]
                    pred_answers.append(sample)
                else:
                    pred_answers.append({'question_id': sample['question_id'],
                                         'question_type': sample['question_type'],
                                         'answers': [best_answer],
                                         'entity_answers': [[]],
                                         'yesno_answers': []})
                if 'answers' in sample:
                    ref_answers.append({'question_id': sample['question_id'],
                                         'question_type': sample['question_type'],
                                         'answers': sample['answers'],
                                         'entity_answers': [[]],
                                         'yesno_answers': []})

        if result_dir is not None and result_prefix is not None:
            result_file = os.path.join(result_dir, result_prefix + '.json')
            with open(result_file, 'w') as fout:
                for pred_answer in pred_answers:
                    fout.write(json.dumps(pred_answer, encoding='utf8', ensure_ascii=False) + '\n')

            self.logger.info('Saving {} results to {}'.format(result_prefix, result_file))

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

    def find_best_answer_with_verif(self, sample, start_probs, end_probs, content_scores, verif_scores, padded_p_len):
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
                start_probs[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                end_probs[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                passage_len)
            if answer_span[0] >= 0:
                mean_content_score = np.mean(content_scores[p_idx * padded_p_len + answer_span[0]: p_idx * padded_p_len + answer_span[1] + 1])
            else:
                mean_content_score = 0
            score = score * mean_content_score * verif_scores[p_idx]
            if score > best_score:
                best_score = score
                best_p_idx = p_idx
                best_span = answer_span
        if best_p_idx is None or best_span is None:
            best_answer = ''
        else:
            best_answer = ''.join(
                sample['passages'][best_p_idx]['passage_tokens'][best_span[0]: best_span[1] + 1])
        return best_answer
