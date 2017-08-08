# -*- coding: utf-8 -*-
import copy
import logging
import time
from collections import OrderedDict

import gc
import numpy as np
import theano
import theano.tensor as T
from multiprocessing import freeze_support

from src import default_initializer
from src.activations import Activation
from src.similarity import cosine_similarity_batch
from src.utils import shared_rand_matrix, align_batch_size, progress_bar_str, load_dev_test_loss, save_dev_test_loss
from src.utils import shared_zero_matrix, load_random_state, save_random_state
from src.data_utils.phrase_utils import read_phrase_list, PARA_INDEX, TRAN_INDEX, WORD_INDEX, TEXT_INDEX

__author__ = 'roger'

logger = logging.getLogger(__name__)

# 定义常用变量
align_len = 7
zero_int32 = T.constant(0, dtype=np.int32)
zero_float32 = T.constant(0, dtype=theano.config.floatX)
one_int32 = T.constant(1, dtype=np.int32)
one_float32 = T.constant(1, dtype=theano.config.floatX)


def sem_distance(p1, w1, b1, p2):
    """
    计算p1投影后的p1'与p2的语义距离
    :param p1: (batch, p1_dim)
    :param w1: (p2_dim, p1_dim)
    :param b1: (p2_dim, )
    :param p2: (batch, p2_dim)
    :return:
    """
    # (batch, p1_dim) (p2_dim, p1_dim)T (p2_dim, ) -> (batch, p2_dim)
    transform_p1 = T.tanh(T.dot(p1, w1.T) + b1)
    # (batch, p2_dim) (batch, p2_dim) -> (batch, )
    return T.sum((p2 - transform_p1) ** 2, axis=1) / 2


def bilinear_score(p1, w1, p2):
    """
    计算p1投影后的p1'与p2的语义距离
    :param p1: (batch, p1_dim)
    :param w1: (p1_dim, p2_dim)
    :param p2: (batch, p2_dim)
    :return:
    """
    # (batch, p1_dim) (p1_dim, p2_dim) -> (batch, p2_dim)
    transform_p1 = T.dot(p1, w1)
    # (batch, p2_dim) (batch, p2_dim) -> (batch, )
    return T.sum(transform_p1 * p2, axis=1)


def sem_sim_distance(p1, w1, b1, p2):
    """
    计算p1投影后的p1'与p2的余弦距离
    :param p1: (batch, p1_dim)
    :param w1: (p2_dim, p1_dim)
    :param b1: (p2_dim, )
    :param p2: (batch, p2_dim)
    :return:
    """
    # (batch, p1_dim) (p2_dim, p1_dim)T (p2_dim, ) -> (batch, p2_dim)
    transform_p1 = T.tanh(T.dot(p1, w1.T) + b1)
    # (batch, p2_dim) (batch, p2_dim) -> (batch, )
    return cosine_similarity_batch(transform_p1, p2), transform_p1


def conditional_probabilities(p1, w1, b1, p2):
    """
    计算p1投影后的p1'与p2的余弦距离
    :param p1: (batch, p1_dim)
    :param w1: (p2_dim, p1_dim)
    :param b1: (p2_dim, )
    :param p2: (batch, p2_dim)
    :return:
    """
    # (batch, p1_dim) (p2_dim, p1_dim)T (p2_dim, ) -> (batch, p2_dim)
    transform_p1 = T.tanh(T.dot(p1, w1.T) + b1)
    # (batch, p2_dim) (batch, p2_dim) -> (batch, )
    return T.exp(-T.sum(T.sqr(transform_p1 - p2), axis=1))


class RecursiveAutoEncoder(object):
    def __init__(self, batch, embedding, initializer=default_initializer, normalize=True, dropout=0,
                 activation="tanh", verbose=True):
        """
        递归自编码器的初始化
        :param batch:           批大小
        :param embedding:       Word Embedding
        :param initializer:     随机初始化器
        :param normalize:       是否归一化
        :param dropout:         dropout率
        :param activation:      激活函数
        :param verbose:         是否输出Debug日志内容
        :return:
        """
        self.embedding = embedding
        self.dim = self.embedding.dim
        self.batch = batch
        self.initializer = initializer
        self.normalize = normalize
        self.dropout = dropout
        self.verbose = verbose
        self.act = Activation(activation)
        # Composition Function Weight
        # (dim, 2 * dim)
        self.W = shared_rand_matrix((self.dim, 2 * self.dim), 'W', initializer=initializer)
        # (dim, )
        self.b = shared_zero_matrix((self.dim,), 'b')
        # Reconstruction Function Weight
        # (2 * dim, dim)
        self.Wr = shared_rand_matrix((2 * self.dim, self.dim), 'Wr', initializer=initializer)
        # (2 * dim, )
        self.br = shared_zero_matrix((self.dim * 2,), 'br')
        self.params = [self.W, self.b, self.Wr, self.br]

        self.l1_norm = sum([T.sum(T.abs_(param)) for param in self.params])
        self.l2_norm = sum([T.sum(param ** 2) for param in self.params])

        # 短语表示循环初始值
        self.zeros_pre = shared_zero_matrix((batch, self.dim))
        # 重构误差循环初始值
        self.zeros_rec = shared_zero_matrix((batch, ))
        # 最大长度用以减少多余训练计算
        self.max_len = T.iscalar()
        # 短语生成序列
        self.input_seqs = T.itensor3()  # (batch, word - 1, 3)
        # 长度掩码
        self.input_mask = T.matrix()  # (batch, word - 1)
        # 输入词在词表中的索引，注 0 为保留索引，用于为生成的子串留空
        self.input_index = T.imatrix()  # (batch, 2 * word - 1)
        # 构建短语表示生成前馈过程
        self.output, self.loss_recs = self.build_phrase_graph(index=self.input_index, seqs=self.input_seqs,
                                                              max_len=self.max_len, name="batch_phrase_generate")

        if verbose:
            logger.debug('Architecture of RAE built finished, summarized as below: ')
            logger.debug('Hidden dimension: %d' % self.dim)
            logger.debug('Normalize:        %s' % self.normalize)
            logger.debug('Dropout Rate:     %s' % self.dropout)

    def compose(self, left, right, W, b, Wr, br):
        """
        合成函数代表一个Batch中的其中一个合成过程
        :param left:  (batch, dim)
        :param right: (batch, dim)
        :param W:     (dim, dim)
        :param b:     (dim, )
        :param Wr:    (dim, dim)
        :param br:    (dim,)
        :return:
        """
        v = T.concatenate([left, right], axis=1)  # [(batch, dim) (batch, dim)] -> (batch, 2 * dim)
        z = self.act.activate(b + T.dot(v, W.T))  # (batch, 2 * dim) dot (dim, 2 * dim)T -> (batch, dim)
        if self.normalize:
            z = z / z.norm(2, axis=1)[:, None]  # (batch, dim) -> (batch, dim) normalize by row
        r = self.act.activate(br + T.dot(z, Wr.T))  # (batch, dim) dot (2 * dim, dim)T -> (batch, 2 * dim)
        left_r, right_r = r[:, :self.dim], r[:, self.dim:]  # (batch, 2 * dim) -> [(batch, dim) (batch. dim)]
        if self.normalize:
            # (batch, dim) -> (batch, dim) normalize by row
            left_r /= left_r.norm(2, axis=1)[:, None]
            # (batch, dim) -> (batch, dim) normalize by row
            right_r /= right_r.norm(2, axis=1)[:, None]
        # (batch, )
        loss_rec = T.sum((left_r - left) ** 2, axis=1) + T.sum((right_r - right) ** 2, axis=1)
        # (batch, dim) (batch)
        return z, loss_rec

    def encode(self, _seq, _mask, _input, _pre, loss_rec, W, b, Wr, br):
        """
        batch合成短语表示过程中 单词循环执行的函数
        :param _seq:   (batch, 3)
        :param _mask:  (batch, )
        :param _input: (batch, word * 2 - 1, dim)
        :param _pre:   (batch, dim)
        :param loss_rec: (batch, )
        :param W:      (dim, dim)
        :param b:      (dim, )
        :param Wr:     (dim, dim)
        :param br:     (dim,)
        :return:       (batch, dim)
        """
        left = _seq[:, 0]
        right = _seq[:, 1]
        # (batch, dim)
        # left_vec = _input[T.arange(self.batch), left]
        left_vec = _input[T.arange(_input.shape[0]), left]
        # (batch, dim)
        right_vec = _input[T.arange(_input.shape[0]), right]
        # (batch, dim) (batch, dim) -> (batch, 2 * dim), (batch, )
        left_right, loss_rec = self.compose(left_vec, right_vec, W, b, Wr, br)
        # (batch, 2 * dim)
        # 若掩码已为0 则代表已经超出原短语长度 此为多余计算 直接去上一轮结果作为该轮结果
        left_right = _mask[:, None] * left_right + (1 - _mask[:, None]) * _pre
        # (batch, )
        # 若掩码已为0 则代表已经超出原短语长度 此为多余计算 用0掩码消去
        loss_rec *= _mask
        # (batch, word * 2 - 1, dim), (batch, dim), (batch, )
        return T.set_subtensor(_input[T.arange(_input.shape[0]), _seq[:, 2]], left_right), left_right, loss_rec

    def build_phrase_graph(self, index, seqs, max_len, name):
        # 从Lookup Table中取出索引对应的词向量构造输入矩阵
        vector = self.embedding[index]  # (batch, 2 * word - 1, dim)
        # scan仅能循环扫描张量的第一维 故转置输入的张量
        seqs = T.transpose(seqs, axes=(1, 0, 2))  # (word - 1, batch, 3)
        mask = (T.transpose(index, axes=(1, 0)) > 0) * one_float32
        result, _ = theano.scan(fn=self.encode,              # 编码函数，对batch数量的短语进行合成
                                sequences=[seqs, mask[1:]],  # 扫描合成路径和掩码
                                                             # 因合成次数为短语长度-1 所以对于长度为1的短语，掩码第一次循环即为0
                                                             # 故取vector的第0维（第一个词）作为初始值，直接返回
                                outputs_info=[vector, vector[:, 0, :], self.zeros_rec],
                                non_sequences=[self.W, self.b, self.Wr, self.br],
                                n_steps=T.maximum(max_len - 1, 1),  # 执行的最小次数为1次，scan无法扫描0次。
                                name=name + "_scan")
        phrases, pres, loss_recs = result
        # (word - 1, batch, dim) -> (batch, dim)
        # 最后一次合成扫描返回的结果为最终表示
        phrases = pres[-1]
        sum_loss_recs = T.sum(loss_recs, axis=0)
        # (batch, dim)
        # 归一化
        if self.normalize:
            phrases = phrases / phrases.norm(2, axis=1)[:, None]
        return phrases, sum_loss_recs

"""    def build_phrase_graph_no_scan(self, index, seqs, max_len, batch_size, name):
        # 从Lookup Table中取出索引对应的词向量构造输入矩阵
        vector = self.embedding[index]  # (batch, 2 * word - 1, dim)
        phrases = vector[:, 0, :]
        sum_loss_recs = shared_zero_matrix((batch_size,), dtype=theano.config.floatX)
        mask = index[:, 1:] > 0 * one_float32
        for i in np.arange(align_len - 1, dtype="int32"):
            vector, phrases, loss_rec = self.encode(seqs[:, i, :], mask[:, i], vector, phrases, sum_loss_recs,
                                                    self.W, self.b, self.Wr, self.br)
            sum_loss_recs += loss_rec
        if self.normalize:
            phrases = phrases / phrases.norm(2, axis=1)[:, None]
        return phrases, sum_loss_recs"""


class NegativeRAE(RecursiveAutoEncoder):
    def __init__(self, batch, embedding, initializer=default_initializer, normalize=True,
                 dropout=0, activation="tanh", verbose=True):
        super(NegativeRAE, self).__init__(batch, embedding, initializer, normalize, dropout, activation, verbose)
        # 定义负例，原理同正例
        self.neg_seqs = T.itensor3()  # (batch, word - 1, 3)
        self.neg_index = T.imatrix()  # (batch. 2 * word - 1)
        # 负例忽略其重构误差
        self.neg_output, _ = self.build_phrase_graph(index=self.neg_index, seqs=self.neg_seqs,
                                                     max_len=self.max_len, name="batch_neg_phrase_generate")


class CrossLingualTransformer(object):
    # 跨语言映射器 用于保存映射的参数
    def __init__(self, src_dim, tar_dim, name, initializer=default_initializer):
        self.src_dim = src_dim
        self.tar_dim = tar_dim
        self.name = name
        self.Wl = shared_rand_matrix(shape=(self.src_dim, self.tar_dim),
                                     name=self.name + "Wl", initializer=initializer)
        self.bl = shared_zero_matrix((self.tar_dim,), 'bl')
        self.params = [self.Wl, self.bl]
        self.l1_norm = sum([T.sum(T.abs_(param)) for param in [self.Wl]])
        self.l2_norm = sum([T.sum(param ** 2) for param in [self.Wl]])


class BilinearTransformer(object):
    # 跨语言映射器 用于保存映射的参数
    def __init__(self, src_dim, tar_dim, name, initializer=default_initializer):
        self.src_dim = src_dim
        self.tar_dim = tar_dim
        self.name = name
        self.W = shared_rand_matrix(shape=(self.src_dim, self.tar_dim),
                                    name=self.name + "Bilinear", initializer=initializer)
        self.params = [self.W]
        self.l1_norm = sum([T.sum(T.abs_(param)) for param in [self.W]])
        self.l2_norm = sum([T.sum(param ** 2) for param in [self.W]])


class BilingualPhraseRAE(object):
    def __init__(self, src_embedding, tar_embedding, initializer=default_initializer, config=None, verbose=True):
        self.length = align_len
        self.src_embedding = src_embedding
        self.tar_embedding = tar_embedding
        self.source_dim = src_embedding.dim
        self.target_dim = tar_embedding.dim
        self.alpha = config.alpha
        self.normalize = config.normalize
        self.lambda_rec = config.weight_rec
        self.lambda_sem = config.weight_sem
        self.lambda_l2 = config.weight_l2
        self.dropout = config.dropout
        self.verbose = verbose
        self.learning_rate = config.optimizer.param["lr"]
        self.batch_size = config.batch_size

        # 构造源语言编码器 目标语言编码器
        self.src_encoder = NegativeRAE(batch=self.batch_size, embedding=self.src_embedding,
                                       initializer=initializer, normalize=self.normalize,
                                       dropout=self.dropout, activation=config.activation, verbose=self.verbose)
        self.tar_encoder = NegativeRAE(batch=self.batch_size, embedding=self.tar_embedding,
                                       initializer=initializer, normalize=self.normalize,
                                       dropout=self.dropout, activation=config.activation, verbose=self.verbose)
        self.source_pos = self.src_encoder.output  # (batch, source_dim)
        self.source_neg = self.src_encoder.neg_output  # (batch, source_dim)
        self.target_pos = self.tar_encoder.output  # (batch, target_dim)
        self.target_neg = self.tar_encoder.neg_output  # (batch, target_dim)

        # 定义跨语言映射参数
        self.s2t = CrossLingualTransformer(self.target_dim, self.source_dim, 's2t_s', initializer=initializer)
        self.t2s = CrossLingualTransformer(self.source_dim, self.target_dim, 't2s_t', initializer=initializer)

        # 定义该模型组件 组件顺序决定参数保存顺序
        self.component = [self.src_embedding, self.tar_embedding,
                          self.src_encoder, self.tar_encoder, self.s2t, self.t2s]
        # 正例语义距离
        self.source_pos_sem = sem_distance(self.source_pos, self.s2t.Wl, self.s2t.bl, self.target_pos)  # (batch, )
        self.target_pos_sem = sem_distance(self.target_pos, self.t2s.Wl, self.t2s.bl, self.source_pos)  # (batch, )
        # 负例语义距离
        self.source_neg_sem = sem_distance(self.source_pos, self.s2t.Wl, self.s2t.bl, self.target_neg)  # (batch, )
        self.target_neg_sem = sem_distance(self.target_pos, self.t2s.Wl, self.t2s.bl, self.source_neg)  # (batch, )
        # 跨语言语义相似度（余弦距离） 跨语言映射表示
        self.source_tar_sim, self.source_in_tar = sem_sim_distance(self.source_pos, self.s2t.Wl,
                                                                   self.s2t.bl, self.target_pos)  # (batch, )
        self.target_src_sim, self.target_in_src = sem_sim_distance(self.target_pos, self.t2s.Wl,
                                                                   self.t2s.bl, self.source_pos)  # (batch, )
        # max_margin间隔
        self.max_margin_source = T.maximum(0.0, self.source_pos_sem - self.source_neg_sem + 1.0)  # (batch, )
        self.max_margin_target = T.maximum(0.0, self.target_pos_sem - self.target_neg_sem + 1.0)  # (batch, )
        # 该batch的语义距离误差
        self.cost_sem = T.sum(self.max_margin_source) + T.sum(self.max_margin_target)  # scalar
        # 该batch的重构误差
        self.cost_rec = T.sum(self.src_encoder.loss_recs) + T.sum(self.tar_encoder.loss_recs)  # scalar

        # 定义代价函数
        self.loss_rec = self.cost_rec
        self.loss_sem = self.cost_sem
        loss_l2_word = self.lambda_l2 * (self.src_embedding.l2_norm + self.tar_embedding.l2_norm)
        '''
        self.src_norm_word_index = T.extra_ops.Unique()(T.concatenate([T.extra_ops.Unique()(self.src_encoder.input_index),
                                                                     T.extra_ops.Unique()(self.src_encoder.neg_index)]))
        self.tar_norm_word_index = T.extra_ops.Unique()(T.concatenate([T.extra_ops.Unique()(self.tar_encoder.input_index),
                                                                     T.extra_ops.Unique()(self.tar_encoder.neg_index)]))
        loss_l2_word = self.lambda_l2 * (T.sum(self.src_embedding[self.src_norm_word_index] ** 2)
                                         + T.sum(self.tar_embedding[self.tar_norm_word_index] ** 2))
        '''
        loss_l2_sem = self.lambda_sem * (self.s2t.l2_norm + self.t2s.l2_norm)
        loss_l2_rec = self.lambda_rec * (self.src_encoder.l2_norm + self.tar_encoder.l2_norm)
        self.loss_l2 = loss_l2_word + loss_l2_rec + loss_l2_sem
        if self.alpha > 1:
            import sys
            logger.error("alpha is bigger than 1 %f" % self.alpha)
            sys.stderr.write("alpha is bigger than 1 %f" % self.alpha)
            exit(-1)
        self.loss_brae = self.alpha * self.loss_rec + (1 - self.alpha) * self.loss_sem + self.loss_l2
        loss = self.loss_brae / config.batch_size

        # 定义模型参数
        self.params = self.src_encoder.params + self.tar_encoder.params + self.t2s.params + self.s2t.params
        self.params += self.src_embedding.params
        self.params += self.tar_embedding.params

        # 模型求导
        grads = T.grad(loss, self.params)
        # 定义更新梯度
        updates = OrderedDict()
        for param, grad in zip(self.params, grads):
            updates[param] = param - grad * self.learning_rate

        self.src_indexs = T.ivector()  # 源短语在源短语列表中的位置 批大小
        self.tar_indexs = T.ivector()  # 源短语在源短语列表中的位置 批大小

        self.get_indexs_embedding = theano.function(inputs=[self.src_indexs, self.tar_indexs],
                                                    outputs=[self.src_embedding[self.src_indexs],
                                                             self.tar_embedding[self.tar_indexs]])

        self.get_src_indexs_embedding = theano.function(inputs=[self.src_indexs],
                                                        outputs=self.src_embedding[self.src_indexs])
        self.get_tar_indexs_embedding = theano.function(inputs=[self.tar_indexs],
                                                        outputs=self.tar_embedding[self.tar_indexs])

        # 训练一个Batch
        self.train_batch = theano.function([self.src_encoder.input_index, self.tar_encoder.input_index,
                                            self.src_encoder.input_seqs, self.tar_encoder.input_seqs,
                                            self.src_encoder.neg_index, self.tar_encoder.neg_index,
                                            self.src_encoder.neg_seqs, self.tar_encoder.neg_seqs,
                                            self.src_encoder.max_len, self.tar_encoder.max_len],
                                           outputs=[loss, self.loss_rec / config.batch_size,
                                                    self.loss_sem / config.batch_size,
                                                    self.loss_l2 / config.batch_size],
                                           updates=updates,
                                           name="brae_objective"
                                           )

        self.predict_batch = theano.function([self.src_encoder.input_index, self.tar_encoder.input_index,
                                              self.src_encoder.input_seqs, self.tar_encoder.input_seqs,
                                              self.src_encoder.neg_index, self.tar_encoder.neg_index,
                                              self.src_encoder.neg_seqs, self.tar_encoder.neg_seqs,
                                              self.src_encoder.max_len, self.tar_encoder.max_len],
                                             outputs=[loss, self.loss_rec / config.batch_size,
                                                      self.loss_sem / config.batch_size,
                                                      self.loss_l2 / config.batch_size],
                                             name="brae_objective_predict"
                                             )

        if verbose:
            logger.debug('Architecture of BRAE built finished, summarized as below: ')
            logger.debug('Alpha:            %f' % self.alpha)
            logger.debug('Lambda Rec:       %f' % self.lambda_rec)
            logger.debug('Lambda Sem:       %f' % self.lambda_sem)
            logger.debug('Lambda Word:      %f' % self.lambda_l2)
            logger.debug('Learning Rate:    %f' % self.learning_rate)

    @staticmethod
    def generate_train_array(pos_phrases, dict_size):
        pos_instance = np.zeros(shape=(len(pos_phrases), 2 * align_len - 1), dtype=np.int32)
        neg_instance = np.zeros(shape=(len(pos_phrases), 2 * align_len - 1), dtype=np.int32)
        for p, pos_ins, neg_ins in zip(pos_phrases, pos_instance, neg_instance):
            len_p = len(p)
            pos_ins[:len_p] = p
            neg_ins[:len_p] = p
            ran_time = np.random.randint(len_p) + 1
            for i in xrange(ran_time):
                index = np.random.randint(len_p)
                neg_ins[index] = np.random.randint(1, dict_size)
        return np.array(pos_instance, dtype=np.int32), np.array(neg_instance, dtype=np.int32)

    @staticmethod
    def generate_train_array_align(pos_phrases, dict_size, phrases_id=None):
        pos_instance = np.zeros(shape=(len(pos_phrases), align_len), dtype=np.int32)
        neg_instance = np.zeros(shape=(len(pos_phrases), align_len), dtype=np.int32)
        if phrases_id is None:
            phrases_id = range(len(pos_instance))
        for phrases_index in phrases_id:
            p, pos_ins, neg_ins = pos_phrases[phrases_index], pos_instance[phrases_index], neg_instance[phrases_index]
            len_p = len(p)
            pos_ins[:len_p] = p
            neg_ins[:len_p] = p
            ran_time = np.random.randint(len_p) + 1
            for i in xrange(ran_time):
                index = np.random.randint(len_p)
                neg_ins[index] = np.random.randint(1, dict_size)
        return np.array(pos_instance, dtype=np.int32), np.array(neg_instance, dtype=np.int32)

    def save_model(self, filename):
        import cPickle
        # Default Component: src_embedding, tar_embedding, src_encoder, tar_encoder, s2t, t2s
        with file(filename, 'wb') as fout:
            cPickle.dump(self.src_embedding.word_idx, fout)
            cPickle.dump(self.tar_embedding.word_idx, fout)
            for comp in self.component:
                for p in comp.params:
                    cPickle.dump(p.get_value(), fout)

        logger.info("Save Model to %s" % filename)

    def load_model(self, filename):
        import cPickle
        # Default Component: src_embedding, tar_embedding, src_encoder, tar_encoder, s2t, t2s
        with file(filename, 'rb') as fin:
            self.src_embedding.word_idx = cPickle.load(fin)
            self.tar_embedding.word_idx = cPickle.load(fin)
            for comp in self.component:
                for p in comp.params:
                    pre_shape = p.get_value().shape
                    p.set_value(cPickle.load(fin))
                    after_shape = p.get_value().shape
                    if p.get_value().shape != pre_shape:
                        raise RuntimeError("Shape load from %s has different "
                                           "shape, %s != %s!\n" % (filename, pre_shape, after_shape))
        logger.info("Load Model from %s" % filename)

    def train(self, src_train, tar_train, src_tar_pair, config, model_name, start_iter=1, end_iter=26):
        n_epoch = config.n_epoch
        batch_size = config.batch_size
        size_src_word = self.src_embedding.size
        size_tar_word = self.tar_embedding.size
        train_index = align_batch_size(range(len(src_tar_pair)), batch_size)

        # 依长度对短语进行排序
        len_src_tar_pair = [(len(src_train[src_tar_pair[index][0]]), len(tar_train[src_tar_pair[index][1]]), index)
                            for index in train_index]
        len_src_tar_pair.sort()
        len_sort_index = np.array([index[2] for index in len_src_tar_pair], np.int32)
        src_len_index = np.array([index[0] for index in len_src_tar_pair], np.int32)
        tar_len_index = np.array([index[1] for index in len_src_tar_pair], np.int32)
        num_batch = len(len_sort_index) / batch_size

        if self.verbose:
            logger.debug("Train Details")
            logger.debug("Number of epochs: %d" % n_epoch)
            logger.debug("Size of Batches:  %d" % batch_size)
            logger.debug("Size phrase pair: %d" % len(src_tar_pair))
            logger.debug("Size Source word: %d" % size_src_word)
            logger.debug("Size Target word: %d" % size_tar_word)

        logger.info("Start training...")
        greedy_time = 0
        train_time = 0
        history_loss = []
        random_batch_index = np.arange(num_batch)
        rae_util = RAEUtil(self.source_dim, normalize=self.normalize)

        for i in xrange(1, n_epoch + 1):
            if i < start_iter:
                continue
            if i == start_iter and start_iter != 1:
                self.load_model("%s_iter%d.model" % (model_name, i - 1))
                load_random_state("%s_iter%d.model.rs" % (model_name, i - 1))
            if i >= end_iter:
                logger.info("Reach End the Iter")
                exit(1)

            # 每一轮迭代都随机生成负例
            src_pos, src_neg = self.generate_train_array_align(src_train, size_src_word)
            tar_pos, tar_neg = self.generate_train_array_align(tar_train, size_tar_word)
            src_pos_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
            src_neg_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
            tar_pos_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
            tar_neg_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
            src_seq = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)
            tar_seq = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)
            src_seq_neg = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)
            tar_seq_neg = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)

            random_batch_index = np.random.permutation(random_batch_index)

            k = 0
            epoch_start = time.time()
            loss = np.zeros(shape=(num_batch, 4), dtype=theano.config.floatX)
            for j in random_batch_index:
                # 清空临时路径
                # self.init_batch()
                if self.verbose:
                    k += 1
                    print progress_bar_str(k, num_batch) + "\r",
                    if k == num_batch:
                        print

                # 依据长度选择该batch训练的实例
                indexs = len_sort_index[j * batch_size: (j + 1) * batch_size]
                src_indexs = [src_tar_pair[index][0] for index in indexs]
                tar_indexs = [src_tar_pair[index][1] for index in indexs]
                src_len_max = np.max(src_len_index[j * batch_size: (j + 1) * batch_size])
                tar_len_max = np.max(tar_len_index[j * batch_size: (j + 1) * batch_size])

                src_pos_train[:, :align_len], src_neg_train[:, :align_len] = src_pos[src_indexs], src_neg[src_indexs]
                tar_pos_train[:, :align_len], tar_neg_train[:, :align_len] = tar_pos[tar_indexs], tar_neg[tar_indexs]

                # 生成贪婪路径
                time_temp = time.time()
                src_w = self.src_encoder.W.get_value(borrow=True)
                src_b = self.src_encoder.b.get_value(borrow=True)
                src_wr = self.src_encoder.Wr.get_value(borrow=True)
                src_br = self.src_encoder.br.get_value(borrow=True)
                tar_w = self.tar_encoder.W.get_value(borrow=True)
                tar_b = self.tar_encoder.b.get_value(borrow=True)
                tar_wr = self.tar_encoder.Wr.get_value(borrow=True)
                tar_br = self.tar_encoder.br.get_value(borrow=True)
                for sub_index in xrange(batch_size):
                    # Pos Path Generate
                    src_word_index = src_pos_train[sub_index]
                    tar_word_index = tar_pos_train[sub_index]
                    src_word_vec, tar_word_vec = self.get_indexs_embedding(src_word_index, tar_word_index)
                    src_seq[sub_index] = rae_util.build_greed_path(src_word_vec, src_w, src_b, src_wr, src_br,
                                                                   src_len_index[j * batch_size + sub_index])
                    tar_seq[sub_index] = rae_util.build_greed_path(tar_word_vec, tar_w, tar_b, tar_wr, tar_br,
                                                                   tar_len_index[j * batch_size + sub_index])
                    # Neg Path Generate
                    src_word_index = src_neg_train[sub_index]
                    tar_word_index = tar_neg_train[sub_index]
                    src_word_vec, tar_word_vec = self.get_indexs_embedding(src_word_index, tar_word_index)
                    src_seq_neg[sub_index] = rae_util.build_greed_path(src_word_vec, src_w, src_b, src_wr, src_br,
                                                                       src_len_index[j * batch_size + sub_index])
                    tar_seq_neg[sub_index] = rae_util.build_greed_path(tar_word_vec, tar_w, tar_b, tar_wr, tar_br,
                                                                       tar_len_index[j * batch_size + sub_index])
                greedy_time += (time.time() - time_temp)

                # 依据贪婪路径进行 前馈 + 反馈 训练
                time_temp = time.time()
                result = self.train_batch(src_pos_train, tar_pos_train,
                                          src_seq, tar_seq,
                                          src_neg_train, tar_neg_train,
                                          src_seq_neg, tar_seq_neg,
                                          src_len_max, tar_len_max)
                logger.debug(result)
                if result[0] == np.inf or result[0] == np.nan:
                    logger.error("Detect nan or inf")
                    exit(-1)
                loss[j] = result
                train_time += (time.time() - time_temp)

            logger.info("epoch %d time: %f, Greedy %f, Train %f" % (i, time.time() - epoch_start, greedy_time, train_time))
            logger.info("epoch %d rec_err %f sem_err %f" % (i, np.mean(loss, axis=0)[1], np.mean(loss, axis=0)[2]))
            logger.info("epoch %d l2_err  %f sum_err %f" % (i, np.mean(loss, axis=0)[3], np.mean(loss, axis=0)[0]))
            history_loss.append(np.mean(loss, axis=0))
            self.save_model("%s_iter%d.model" % (model_name, i))
            save_random_state("%s_iter%d.model.rs" % (model_name, i))
            if len(history_loss) > 1 and abs(history_loss[-1][0] - history_loss[-2][0]) < 10e-6:
                logger.info("joint error reaches a local minima")
                break

            # 清空临时内存
            del loss, src_pos, src_neg, tar_pos, tar_neg
            gc.collect()

    def tune_hyper_parameter(self, src_train, tar_train, train_pair, dev_pair, test_pair, config, model_name, start_iter=1, end_iter=26):
        n_epoch = config.n_epoch
        batch_size = config.batch_size
        size_src_word = self.src_embedding.size
        size_tar_word = self.tar_embedding.size
        train_index = align_batch_size(range(len(train_pair)), batch_size)

        src_phrase_id_set = set([src for src, tar in train_pair]) | set([src for src, tar in dev_pair]) | set([src for src, tar in test_pair])
        tar_phrase_id_set = set([tar for src, tar in train_pair]) | set([tar for src, tar in dev_pair]) | set([tar for src, tar in test_pair])

        # 依长度对短语进行排序
        len_src_tar_pair = [(len(src_train[train_pair[index][0]]), len(tar_train[train_pair[index][1]]), index)
                            for index in train_index]
        len_src_tar_pair.sort()
        len_sort_index = np.array([index[2] for index in len_src_tar_pair], np.int32)
        src_len_index = np.array([index[0] for index in len_src_tar_pair], np.int32)
        tar_len_index = np.array([index[1] for index in len_src_tar_pair], np.int32)
        num_batch = len(len_sort_index) / batch_size

        if self.verbose:
            logger.debug("Train Details")
            logger.debug("Number of epochs: %d" % n_epoch)
            logger.debug("Size of Batches:  %d" % batch_size)
            logger.debug("Size phrase pair: %d" % len(train_pair))
            logger.debug("Size Source word: %d" % size_src_word)
            logger.debug("Size Target word: %d" % size_tar_word)

        logger.info("Start training...")
        history_loss = []
        dev_history_loss = []
        test_history_loss = []
        random_batch_index = np.arange(num_batch)
        rae_util = RAEUtil(self.source_dim, normalize=self.normalize)

        for i in xrange(1, n_epoch + 1):
            if i < start_iter:
                continue
            if i == start_iter and start_iter != 1:
                self.load_model("%s_iter%d.model" % (model_name, i - 1))
                load_random_state("%s_iter%d.model.rs" % (model_name, i - 1))
            if i >= end_iter:
                logger.info("Reach End the Iter")
                exit(1)

            # 每一轮迭代都随机生成负例
            src_pos, src_neg = self.generate_train_array_align(src_train, size_src_word,
                                                               phrases_id=src_phrase_id_set)
            tar_pos, tar_neg = self.generate_train_array_align(tar_train, size_tar_word,
                                                               phrases_id=tar_phrase_id_set)
            src_pos_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
            src_neg_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
            tar_pos_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
            tar_neg_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
            src_seq = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)
            tar_seq = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)
            src_seq_neg = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)
            tar_seq_neg = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)

            random_batch_index = np.random.permutation(random_batch_index)

            k = 0
            epoch_start = time.time()
            greedy_time = 0
            train_time = 0
            loss = np.zeros(shape=(num_batch, 4), dtype=theano.config.floatX)
            for j in random_batch_index:
                # 清空临时路径
                # self.init_batch()
                if self.verbose:
                    k += 1
                    print progress_bar_str(k, num_batch) + "\r",
                    if k == num_batch:
                        print

                # 依据长度选择该batch训练的实例
                indexs = len_sort_index[j * batch_size: (j + 1) * batch_size]
                src_indexs = [train_pair[index][0] for index in indexs]
                tar_indexs = [train_pair[index][1] for index in indexs]
                src_len_max = np.max(src_len_index[j * batch_size: (j + 1) * batch_size])
                tar_len_max = np.max(tar_len_index[j * batch_size: (j + 1) * batch_size])

                src_pos_train[:, :align_len], src_neg_train[:, :align_len] = src_pos[src_indexs], src_neg[src_indexs]
                tar_pos_train[:, :align_len], tar_neg_train[:, :align_len] = tar_pos[tar_indexs], tar_neg[tar_indexs]

                # 生成贪婪路径
                time_temp = time.time()
                src_w = self.src_encoder.W.get_value(borrow=True)
                src_b = self.src_encoder.b.get_value(borrow=True)
                src_wr = self.src_encoder.Wr.get_value(borrow=True)
                src_br = self.src_encoder.br.get_value(borrow=True)
                tar_w = self.tar_encoder.W.get_value(borrow=True)
                tar_b = self.tar_encoder.b.get_value(borrow=True)
                tar_wr = self.tar_encoder.Wr.get_value(borrow=True)
                tar_br = self.tar_encoder.br.get_value(borrow=True)
                for sub_index in xrange(batch_size):
                    # Pos Path Generate
                    src_word_index = src_pos_train[sub_index]
                    tar_word_index = tar_pos_train[sub_index]
                    src_word_vec, tar_word_vec = self.get_indexs_embedding(src_word_index, tar_word_index)
                    src_seq[sub_index] = rae_util.build_greed_path(src_word_vec, src_w, src_b, src_wr, src_br,
                                                                   src_len_index[j * batch_size + sub_index])
                    tar_seq[sub_index] = rae_util.build_greed_path(tar_word_vec, tar_w, tar_b, tar_wr, tar_br,
                                                                   tar_len_index[j * batch_size + sub_index])
                    # Neg Path Generate
                    src_word_index = src_neg_train[sub_index]
                    tar_word_index = tar_neg_train[sub_index]
                    src_word_vec, tar_word_vec = self.get_indexs_embedding(src_word_index, tar_word_index)
                    src_seq_neg[sub_index] = rae_util.build_greed_path(src_word_vec, src_w, src_b, src_wr, src_br,
                                                                       src_len_index[j * batch_size + sub_index])
                    tar_seq_neg[sub_index] = rae_util.build_greed_path(tar_word_vec, tar_w, tar_b, tar_wr, tar_br,
                                                                       tar_len_index[j * batch_size + sub_index])
                greedy_time += (time.time() - time_temp)

                # 依据贪婪路径进行 前馈 + 反馈 训练
                time_temp = time.time()
                result = self.train_batch(src_pos_train, tar_pos_train,
                                          src_seq, tar_seq,
                                          src_neg_train, tar_neg_train,
                                          src_seq_neg, tar_seq_neg,
                                          src_len_max, tar_len_max)
                logger.debug(result)
                if result[0] == np.inf or result[0] == np.nan:
                    logger.error("Detect nan or inf")
                    exit(-1)
                loss[j] = result
                train_time += (time.time() - time_temp)

            logger.info(
                "epoch %d time: %f, Greedy %f, Train %f" % (i, time.time() - epoch_start, greedy_time, train_time))
            logger.info("epoch %d rec_err %f sem_err %f" % (i, np.mean(loss, axis=0)[1], np.mean(loss, axis=0)[2]))
            logger.info("epoch %d l2_err  %f sum_err %f" % (i, np.mean(loss, axis=0)[3], np.mean(loss, axis=0)[0]))
            history_loss.append(np.mean(loss, axis=0))
            dev_history_loss.append(self.predict_loss(src_train, tar_train, dev_pair,
                                                      src_pos, src_neg, tar_pos, tar_neg, config,
                                                      pref="epoch %d Dev" % i))
            test_history_loss.append(self.predict_loss(src_train, tar_train, test_pair,
                                                       src_pos, src_neg, tar_pos, tar_neg, config,
                                                       pref="epoch %d Test" % i))
            self.save_model("%s_iter%d.model" % (model_name, i))
            save_random_state("%s_iter%d.model.rs" % (model_name, i))
            save_dev_test_loss("%s_iter%d.tune.model.loss" % (model_name, i), dev_history_loss, test_history_loss)
            if len(history_loss) > 1 and abs(history_loss[-1][0] - history_loss[-2][0]) < 10e-6:
                logger.info("joint error reaches a local minima")
                dev_final_err = [losses[0] for losses in dev_history_loss]
                test_final_err = [losses[0] for losses in test_history_loss]
                min_iter = np.argmin(dev_final_err)
                logger.info("[MinErr] at Iter %d" % min_iter)
                logger.info("[DevErr] %f" % dev_final_err[min_iter])
                logger.info("[TestErr] %f" % test_final_err[min_iter])
                break
            if i == n_epoch:
                logger.info("joint error reaches a local minima")
                dev_final_err = [losses[0] for losses in dev_history_loss]
                test_final_err = [losses[0] for losses in test_history_loss]
                min_iter = np.argmin(dev_final_err)
                logger.info("[MinErr] at Iter %d" % min_iter)
                logger.info("[DevErr] %f" % dev_final_err[min_iter])
                logger.info("[TestErr] %f" % test_final_err[min_iter])
                break

            # 清空临时内存
            del loss, src_pos, src_neg, tar_pos, tar_neg
            gc.collect()

    def predict_loss(self, src_train, tar_train, test_pair, src_pos, src_neg, tar_pos, tar_neg, config, pref=""):
        batch_size = config.batch_size
        train_index = align_batch_size(range(len(test_pair)), batch_size)

        # 依长度对短语进行排序
        len_src_tar_pair = [(len(src_train[test_pair[index][0]]), len(tar_train[test_pair[index][1]]), index)
                            for index in train_index]
        len_src_tar_pair.sort()
        len_sort_index = np.array([index[2] for index in len_src_tar_pair], np.int32)
        src_len_index = np.array([index[0] for index in len_src_tar_pair], np.int32)
        tar_len_index = np.array([index[1] for index in len_src_tar_pair], np.int32)
        num_batch = len(len_sort_index) / batch_size
        rae_util = RAEUtil(self.source_dim, normalize=self.normalize)

        # 每一轮迭代都随机生成负例
        src_pos_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
        src_neg_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
        tar_pos_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
        tar_neg_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
        src_seq = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)
        tar_seq = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)
        src_seq_neg = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)
        tar_seq_neg = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)

        k = 0
        loss = np.zeros(shape=(num_batch, 4), dtype=theano.config.floatX)
        for j in xrange(num_batch):
            # 清空临时路径
            # self.init_batch()
            if self.verbose:
                k += 1
                print progress_bar_str(k, num_batch) + "\r",
                if k == num_batch:
                    print

            # 依据长度选择该batch训练的实例
            indexs = len_sort_index[j * batch_size: (j + 1) * batch_size]
            src_indexs = [test_pair[index][0] for index in indexs]
            tar_indexs = [test_pair[index][1] for index in indexs]
            src_len_max = np.max(src_len_index[j * batch_size: (j + 1) * batch_size])
            tar_len_max = np.max(tar_len_index[j * batch_size: (j + 1) * batch_size])

            src_pos_train[:, :align_len], src_neg_train[:, :align_len] = src_pos[src_indexs], src_neg[src_indexs]
            tar_pos_train[:, :align_len], tar_neg_train[:, :align_len] = tar_pos[tar_indexs], tar_neg[tar_indexs]

            # 生成贪婪路径
            src_w = self.src_encoder.W.get_value(borrow=True)
            src_b = self.src_encoder.b.get_value(borrow=True)
            src_wr = self.src_encoder.Wr.get_value(borrow=True)
            src_br = self.src_encoder.br.get_value(borrow=True)
            tar_w = self.tar_encoder.W.get_value(borrow=True)
            tar_b = self.tar_encoder.b.get_value(borrow=True)
            tar_wr = self.tar_encoder.Wr.get_value(borrow=True)
            tar_br = self.tar_encoder.br.get_value(borrow=True)
            for sub_index in xrange(batch_size):
                # Pos Path Generate
                src_word_index = src_pos_train[sub_index]
                tar_word_index = tar_pos_train[sub_index]
                src_word_vec, tar_word_vec = self.get_indexs_embedding(src_word_index, tar_word_index)
                src_seq[sub_index] = rae_util.build_greed_path(src_word_vec, src_w, src_b, src_wr, src_br,
                                                               src_len_index[j * batch_size + sub_index])
                tar_seq[sub_index] = rae_util.build_greed_path(tar_word_vec, tar_w, tar_b, tar_wr, tar_br,
                                                               tar_len_index[j * batch_size + sub_index])
                # Neg Path Generate
                src_word_index = src_neg_train[sub_index]
                tar_word_index = tar_neg_train[sub_index]
                src_word_vec, tar_word_vec = self.get_indexs_embedding(src_word_index, tar_word_index)
                src_seq_neg[sub_index] = rae_util.build_greed_path(src_word_vec, src_w, src_b, src_wr, src_br,
                                                                   src_len_index[j * batch_size + sub_index])
                tar_seq_neg[sub_index] = rae_util.build_greed_path(tar_word_vec, tar_w, tar_b, tar_wr, tar_br,
                                                                   tar_len_index[j * batch_size + sub_index])

            # 依据贪婪路径进行 前馈 + 反馈 训练
            result = self.predict_batch(src_pos_train, tar_pos_train,
                                        src_seq, tar_seq,
                                        src_neg_train, tar_neg_train,
                                        src_seq_neg, tar_seq_neg,
                                        src_len_max, tar_len_max)

            loss[j] = result

        logger.info("rec_err %f sem_err %f" % (np.mean(loss, axis=0)[1], np.mean(loss, axis=0)[2]))
        logger.info("l2_err  %f sum_err %f" % (np.mean(loss, axis=0)[3], np.mean(loss, axis=0)[0]))
        return np.mean(loss, axis=0)


class BilingualPhraseRAEBiLinear(object):
    def __init__(self, src_embedding, tar_embedding, initializer=default_initializer, config=None, verbose=True):
        self.length = align_len
        self.src_embedding = src_embedding
        self.tar_embedding = tar_embedding
        self.source_dim = src_embedding.dim
        self.target_dim = tar_embedding.dim
        self.alpha = config.alpha
        self.normalize = config.normalize
        self.lambda_rec = config.weight_rec
        self.lambda_sem = config.weight_sem
        self.lambda_l2 = config.weight_l2
        self.dropout = config.dropout
        self.verbose = verbose
        self.learning_rate = config.optimizer.param["lr"]
        self.batch_size = config.batch_size

        # 构造源语言编码器 目标语言编码器
        self.src_encoder = NegativeRAE(batch=self.batch_size, embedding=self.src_embedding,
                                       initializer=initializer, normalize=self.normalize,
                                       dropout=self.dropout, activation=config.activation, verbose=self.verbose)
        self.tar_encoder = NegativeRAE(batch=self.batch_size, embedding=self.tar_embedding,
                                       initializer=initializer, normalize=self.normalize,
                                       dropout=self.dropout, activation=config.activation, verbose=self.verbose)
        self.source_pos = self.src_encoder.output  # (batch, source_dim)
        self.source_neg = self.src_encoder.neg_output  # (batch, source_dim)
        self.target_pos = self.tar_encoder.output  # (batch, target_dim)
        self.target_neg = self.tar_encoder.neg_output  # (batch, target_dim)

        self.bilinear_w = BilinearTransformer(self.src_embedding.dim, self.tar_embedding.dim,
                                              name="bilinear", initializer=initializer)

        # 定义该模型组件 组件顺序决定参数保存顺序
        self.component = [self.src_embedding, self.tar_embedding,
                          self.src_encoder, self.tar_encoder, self.bilinear_w]
        # 正例语义距离
        self.pos_score = bilinear_score(self.source_pos, self.bilinear_w.W, self.target_pos)
        self.src_neg_score = bilinear_score(self.source_neg, self.bilinear_w.W, self.target_pos)
        self.tar_neg_score = bilinear_score(self.source_pos, self.bilinear_w.W, self.target_neg)

        # max_margin间隔
        self.max_margin_source = T.maximum(0.0, self.src_neg_score - self.pos_score + 1.0)  # (batch, )
        self.max_margin_target = T.maximum(0.0, self.tar_neg_score - self.pos_score + 1.0)  # (batch, )
        # 该batch的语义距离误差
        self.cost_sem = T.sum(self.max_margin_source) + T.sum(self.max_margin_target)  # scalar
        # 该batch的重构误差
        self.cost_rec = T.sum(self.src_encoder.loss_recs) + T.sum(self.tar_encoder.loss_recs)  # scalar

        # 定义代价函数
        self.loss_rec = self.cost_rec
        self.loss_sem = self.cost_sem
        loss_l2_word = self.lambda_l2 * (self.src_embedding.l2_norm + self.tar_embedding.l2_norm)
        loss_l2_sem = self.lambda_sem * self.bilinear_w.l2_norm
        loss_l2_rec = self.lambda_rec * (self.src_encoder.l2_norm + self.tar_encoder.l2_norm)
        self.loss_l2 = loss_l2_word + loss_l2_rec + loss_l2_sem
        if self.alpha > 1 or self.alpha < 0:
            import sys
            logger.error("alpha is out-of-range [0, 1]. %f" % self.alpha)
            sys.stderr.write("alpha is out-of-range [0, 1]. %f" % self.alpha)
            exit(-1)
        self.loss_brae = self.alpha * self.loss_rec + (1 - self.alpha) * self.loss_sem + self.loss_l2
        loss = self.loss_brae / config.batch_size

        # 定义模型参数
        self.params = self.src_encoder.params + self.tar_encoder.params + self.bilinear_w.params
        self.params += self.src_embedding.params
        self.params += self.tar_embedding.params

        # 模型求导
        grads = T.grad(loss, self.params)
        # 定义更新梯度
        updates = OrderedDict()
        for param, grad in zip(self.params, grads):
            updates[param] = param - grad * self.learning_rate

        self.src_indexs = T.ivector()  # 源短语在源短语列表中的位置 批大小
        self.tar_indexs = T.ivector()  # 源短语在源短语列表中的位置 批大小

        self.get_indexs_embedding = theano.function(inputs=[self.src_indexs, self.tar_indexs],
                                                    outputs=[self.src_embedding[self.src_indexs],
                                                             self.tar_embedding[self.tar_indexs]])

        self.get_src_indexs_embedding = theano.function(inputs=[self.src_indexs],
                                                        outputs=self.src_embedding[self.src_indexs])
        self.get_tar_indexs_embedding = theano.function(inputs=[self.tar_indexs],
                                                        outputs=self.tar_embedding[self.tar_indexs])

        # 训练一个Batch
        self.train_batch = theano.function([self.src_encoder.input_index, self.tar_encoder.input_index,
                                            self.src_encoder.input_seqs, self.tar_encoder.input_seqs,
                                            self.src_encoder.neg_index, self.tar_encoder.neg_index,
                                            self.src_encoder.neg_seqs, self.tar_encoder.neg_seqs,
                                            self.src_encoder.max_len, self.tar_encoder.max_len],
                                           outputs=[loss, self.loss_rec / config.batch_size,
                                                    self.loss_sem / config.batch_size,
                                                    self.loss_l2 / config.batch_size],
                                           updates=updates,
                                           )

        if verbose:
            logger.debug('Architecture of BRAE built finished, summarized as below: ')
            logger.debug('Alpha:            %f' % self.alpha)
            logger.debug('Lambda Rec:       %f' % self.lambda_rec)
            logger.debug('Lambda Sem:       %f' % self.lambda_sem)
            logger.debug('Lambda Word:      %f' % self.lambda_l2)
            logger.debug('Learning Rate:    %f' % self.learning_rate)

    @staticmethod
    def generate_train_array(pos_phrases, dict_size):
        pos_instance = np.zeros(shape=(len(pos_phrases), 2 * align_len - 1), dtype=np.int32)
        neg_instance = np.zeros(shape=(len(pos_phrases), 2 * align_len - 1), dtype=np.int32)
        for p, pos_ins, neg_ins in zip(pos_phrases, pos_instance, neg_instance):
            len_p = len(p)
            pos_ins[:len_p] = p
            neg_ins[:len_p] = p
            ran_time = np.random.randint(len_p) + 1
            for i in xrange(ran_time):
                index = np.random.randint(len_p)
                neg_ins[index] = np.random.randint(1, dict_size)
        return np.array(pos_instance, dtype=np.int32), np.array(neg_instance, dtype=np.int32)

    def save_model(self, filename):
        import cPickle
        # Default Component: src_embedding, tar_embedding, src_encoder, tar_encoder, s2t, t2s
        with file(filename, 'wb') as fout:
            cPickle.dump(self.src_embedding.word_idx, fout)
            cPickle.dump(self.tar_embedding.word_idx, fout)
            for comp in self.component:
                for p in comp.params:
                    cPickle.dump(p.get_value(), fout)

        logger.info("Save Model to %s" % filename)

    def load_model(self, filename):
        import cPickle
        # Default Component: src_embedding, tar_embedding, src_encoder, tar_encoder, s2t, t2s
        with file(filename, 'rb') as fin:
            self.src_embedding.word_idx = cPickle.load(fin)
            self.tar_embedding.word_idx = cPickle.load(fin)
            for comp in self.component:
                for p in comp.params:
                    pre_shape = p.get_value().shape
                    p.set_value(cPickle.load(fin))
                    after_shape = p.get_value().shape
                    if p.get_value().shape != pre_shape:
                        raise RuntimeError("Shape load from %s has different "
                                           "shape, %s != %s!\n" % (filename, pre_shape, after_shape))
        logger.info("Load Model from %s" % filename)

    def train(self, src_train, tar_train, src_tar_pair, config, model_name, start_iter=1, end_iter=26):
        n_epoch = config.n_epoch
        batch_size = config.batch_size
        size_src_word = self.src_embedding.size
        size_tar_word = self.tar_embedding.size

        train_index = align_batch_size(range(len(src_tar_pair)), batch_size)

        # 依长度对短语进行排序
        len_src_tar_pair = [(len(src_train[src_tar_pair[index][0]]), len(tar_train[src_tar_pair[index][1]]), index)
                            for index in train_index]
        len_src_tar_pair.sort()
        len_sort_index = np.array([index[2] for index in len_src_tar_pair], np.int32)
        src_len_index = np.array([index[0] for index in len_src_tar_pair], np.int32)
        tar_len_index = np.array([index[1] for index in len_src_tar_pair], np.int32)
        num_batch = len(len_sort_index) / batch_size

        if self.verbose:
            logger.debug("Train Details")
            logger.debug("Number of epochs: %d" % n_epoch)
            logger.debug("Size of Batches:  %d" % batch_size)
            logger.debug("Size phrase pair: %d" % len(src_tar_pair))
            logger.debug("Size Source word: %d" % size_src_word)
            logger.debug("Size Target word: %d" % size_tar_word)

        logger.info("Start training...")
        greedy_time = 0
        train_time = 0
        history_loss = []
        rae_util = RAEUtil(self.source_dim, normalize=self.normalize)
        random_batch_index = np.arange(num_batch)
        for i in xrange(1, n_epoch + 1):

            if i < start_iter:
                continue
            if i == start_iter and start_iter != 1:
                self.load_model("%s_iter%d.model" % (model_name, i - 1))
                load_random_state("%s_iter%d.model.rs" % (model_name, i - 1))
            if i >= end_iter:
                logger.info("Reach End the Iter")
                exit(1)

            # 每一轮迭代都随机生成负例
            src_pos, src_neg = BilingualPhraseRAE.generate_train_array_align(src_train, size_src_word)
            tar_pos, tar_neg = BilingualPhraseRAE.generate_train_array_align(tar_train, size_tar_word)
            src_pos_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
            src_neg_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
            tar_pos_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
            tar_neg_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
            src_seq = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)
            tar_seq = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)
            src_seq_neg = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)
            tar_seq_neg = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)

            # 初始化Theano变量
            random_batch_index = np.random.permutation(random_batch_index)

            k = 0
            epoch_start = time.time()
            loss = np.zeros(shape=(num_batch, 4), dtype=theano.config.floatX)
            for j in random_batch_index:
                # 清空临时路径
                if self.verbose:
                    k += 1
                    print progress_bar_str(k, num_batch) + "\r",
                    if k == num_batch:
                        print

                # 依据长度选择该batch训练的实例
                indexs = len_sort_index[j * batch_size: (j + 1) * batch_size]
                src_indexs = [src_tar_pair[index][0] for index in indexs]
                tar_indexs = [src_tar_pair[index][1] for index in indexs]
                src_len_max = np.max(src_len_index[j * batch_size: (j + 1) * batch_size])
                tar_len_max = np.max(tar_len_index[j * batch_size: (j + 1) * batch_size])

                src_pos_train[:, :align_len], src_neg_train[:, :align_len] = src_pos[src_indexs], src_neg[src_indexs]
                tar_pos_train[:, :align_len], tar_neg_train[:, :align_len] = tar_pos[tar_indexs], tar_neg[tar_indexs]

                # 生成贪婪路径
                time_temp = time.time()
                src_w = self.src_encoder.W.get_value(borrow=True)
                src_b = self.src_encoder.b.get_value(borrow=True)
                src_wr = self.src_encoder.Wr.get_value(borrow=True)
                src_br = self.src_encoder.br.get_value(borrow=True)
                tar_w = self.tar_encoder.W.get_value(borrow=True)
                tar_b = self.tar_encoder.b.get_value(borrow=True)
                tar_wr = self.tar_encoder.Wr.get_value(borrow=True)
                tar_br = self.tar_encoder.br.get_value(borrow=True)
                for sub_index in xrange(batch_size):
                    # Pos Path Generate
                    src_word_index = src_pos_train[sub_index]
                    tar_word_index = tar_pos_train[sub_index]
                    src_word_vec, tar_word_vec = self.get_indexs_embedding(src_word_index, tar_word_index)
                    src_seq[sub_index] = rae_util.build_greed_path(src_word_vec, src_w, src_b, src_wr, src_br,
                                                                   src_len_index[j * batch_size + sub_index])
                    tar_seq[sub_index] = rae_util.build_greed_path(tar_word_vec, tar_w, tar_b, tar_wr, tar_br,
                                                                   tar_len_index[j * batch_size + sub_index])
                    # Neg Path Generate
                    src_word_index = src_neg_train[sub_index]
                    tar_word_index = tar_neg_train[sub_index]
                    src_word_vec, tar_word_vec = self.get_indexs_embedding(src_word_index, tar_word_index)
                    src_seq_neg[sub_index] = rae_util.build_greed_path(src_word_vec, src_w, src_b, src_wr, src_br,
                                                                       src_len_index[j * batch_size + sub_index])
                    tar_seq_neg[sub_index] = rae_util.build_greed_path(tar_word_vec, tar_w, tar_b, tar_wr, tar_br,
                                                                       tar_len_index[j * batch_size + sub_index])
                greedy_time += (time.time() - time_temp)

                # 依据贪婪路径进行 前馈 + 反馈 训练
                time_temp = time.time()
                result = self.train_batch(src_pos_train, tar_pos_train,
                                          src_seq, tar_seq,
                                          src_neg_train, tar_neg_train,
                                          src_seq_neg, tar_seq_neg,
                                          src_len_max, tar_len_max)
                logger.debug(result)
                if result[0] == np.inf or result[0] == np.nan:
                    logger.error("Detect nan or inf")
                    exit(-1)
                loss[j] = result
                train_time += (time.time() - time_temp)

            logger.info("epoch %d time: %f, Greedy %f, Train %f" % (i, time.time() - epoch_start, greedy_time, train_time))
            logger.info("epoch %d rec_err %f sem_err %f" % (i, np.mean(loss, axis=0)[1], np.mean(loss, axis=0)[2]))
            logger.info("epoch %d l2_err  %f sum_err %f" % (i, np.mean(loss, axis=0)[3], np.mean(loss, axis=0)[0]))
            history_loss.append(np.mean(loss, axis=0))
            self.save_model("%s_iter%d.model" % (model_name, i))
            save_random_state("%s_iter%d.model.rs" % (model_name, i))
            if len(history_loss) > 1 and abs(history_loss[-1][0] - history_loss[-2][0]) < 10e-6:
                logger.info("joint error reaches a local minima")
                break

            # 清空临时内存
            del loss, src_pos, src_neg, tar_pos, tar_neg
            gc.collect()


class BilingualPhraseRAELLE(BilingualPhraseRAEBiLinear):
    def __init__(self, src_embedding, tar_embedding, config=None, verbose=True):
        super(BilingualPhraseRAELLE, self).__init__(src_embedding, tar_embedding, config=config, verbose=verbose)
        self.beta = config.beta

        # 最大长度用以减少多余训练计算
        self.src_para_max_len = T.iscalar()
        self.src_para_input_seqs = T.itensor3()  # (batch, word - 1, 3)
        self.src_para_input_index = T.imatrix()  # (batch, 2 * word - 1)
        self.src_para_input_weight = T.fvector()  # (batch, )
        self.tar_para_max_len = T.iscalar()
        self.tar_para_input_seqs = T.itensor3()  # (batch, word - 1, 3)
        self.tar_para_input_index = T.imatrix()  # (batch, 2 * word - 1)
        self.tar_para_input_weight = T.fvector()  # (batch, )

        # (batch, dim)
        self.src_para_output, _ = self.src_encoder.build_phrase_graph(index=self.src_para_input_index,
                                                                      seqs=self.src_para_input_seqs,
                                                                      max_len=self.src_para_max_len,
                                                                      name="batch_src_para_phrase_generate")
        # (batch, dim)
        self.tar_para_output, _ = self.tar_encoder.build_phrase_graph(index=self.tar_para_input_index,
                                                                      seqs=self.tar_para_input_seqs,
                                                                      max_len=self.tar_para_max_len,
                                                                      name="batch_tar_para_phrase_generate")
        # (batch, dim) (batch, dim) -> (batch, 1)
        self.loss_le_src = self.src_para_input_weight * T.sum((self.source_pos - self.src_para_output) ** 2, axis=1)
        self.loss_le_tar = self.tar_para_input_weight * T.sum((self.target_pos - self.tar_para_output) ** 2, axis=1)
        self.loss_le = T.sum(self.loss_le_src) + T.sum(self.loss_le_tar)
        if self.alpha + self.beta > 1 or self.alpha < 0 or self.beta < 0:
            import sys
            logger.error("alpha and beta is out-of-range. alpha: %f, beta: %f." % (self.alpha, self.beta))
            sys.stderr.write("alpha and beta is out-of-range. alpha: %f, beta: %f." % (self.alpha, self.beta))
            exit(-1)
        loss = (self.alpha * self.loss_rec + (1 - self.alpha - self.beta) * self.loss_sem + self.loss_l2 + self.beta * self.loss_le) / config.batch_size
        grads = T.grad(loss, self.params)

        # Define Grad and Update
        updates = OrderedDict()
        for param, grad in zip(self.params, grads):
            updates[param] = param - grad * self.learning_rate

        self.train_batch = theano.function([self.src_encoder.input_index, self.tar_encoder.input_index,
                                            self.src_encoder.input_seqs, self.tar_encoder.input_seqs,
                                            self.src_encoder.neg_index, self.tar_encoder.neg_index,
                                            self.src_encoder.neg_seqs, self.tar_encoder.neg_seqs,
                                            self.src_encoder.max_len, self.tar_encoder.max_len,
                                            self.src_para_input_index, self.tar_para_input_index,
                                            self.src_para_input_seqs, self.tar_para_input_seqs,
                                            self.src_para_input_weight, self.tar_para_input_weight,
                                            self.src_para_max_len, self.tar_para_max_len],
                                           outputs=[loss, self.loss_rec / config.batch_size,
                                                    self.loss_sem / config.batch_size, self.loss_l2 / config.batch_size,
                                                    self.loss_le / config.batch_size],
                                           updates=updates,
                                           )

        if verbose:
            logger.debug('Architecture of BRAE with LE built finished, summarized as below: ')
            logger.debug('Beta:             %f' % self.beta)

    @staticmethod
    def generate_para_train(phrases, train_indexs):
        """
        :param phrases:     短语表
        :param train_indexs: 训练过程中短语索引顺序
        :return:
        """
        para_tables = np.zeros((len(train_indexs)), dtype=np.int32)
        weight = np.zeros((len(train_indexs, )), dtype=theano.config.floatX)
        for p_index, index in zip(train_indexs, xrange(len(train_indexs))):
            if phrases[p_index][PARA_INDEX] is None or len(phrases[p_index][PARA_INDEX]) == 0:
                # 若不存在复述 则取weight=0
                phrase_index = p_index
                w = 0
            else:
                para_index = np.random.randint(len(phrases[p_index][PARA_INDEX]))
                phrase_index, w = phrases[p_index][PARA_INDEX][para_index]
            para_tables[index] = phrase_index
            weight[index] = w
        return np.array(para_tables, dtype=np.int32), weight

    @staticmethod
    def generate_para_train2(phrases, train_indexs, num):
        para_table = np.zeros((len(train_indexs), num), dtype=np.int32)
        weight = np.zeros((len(train_indexs), num), dtype=np.float32)
        nums = np.zeros((len(train_indexs)), dtype=np.int32)
        for p_index, index in zip(train_indexs, xrange(len(train_indexs))):
            if phrases[p_index][PARA_INDEX] is None or len(phrases[p_index][PARA_INDEX]) == 0:
                para_table[index][0] = p_index
                weight[index][0] = 1
                nums[index] = 0
            else:
                this_num = min(num, len(phrases[p_index][PARA_INDEX]))
                for i in xrange(this_num):
                    para_table[index][i], weight[index][i] = phrases[p_index][PARA_INDEX][i]
                nums[index] = this_num
                if np.sum(weight[index]) != 0:
                    weight[index] /= np.sum(weight[index])
        return para_table, weight, nums

    @staticmethod
    def get_err_le(encoder, pos_phrases, weight, para_input_index, para_input_seqs, para_max_len, para_input_num,
                   max_num, batch_size, name=""):
        para_err = shared_zero_matrix((batch_size, max_num), name="para_dis")
        for i in range(max_num):
            rep, _ = encoder.build_phrase_graph(index=para_input_index[:, i, :],
                                                seqs=para_input_seqs[:, i, :, :],
                                                max_len=para_max_len[i],
                                                name=name+"para_graph")
            err = T.exp(-T.sum(T.sqr(pos_phrases - rep), axis=1))
            mask = (para_input_num > i) * one_float32
            para_err = T.set_subtensor(para_err[:, i], err * mask)
        sum_err = T.sum(para_err, axis=1)
        new_sum_err = T.eq(sum_err, 0) * one_float32 + sum_err
        para_err /= new_sum_err[:, None]

        def kl_div(score, wei, num):
            a = T.sum(wei[:num] * T.log(wei[:num] / score[:num]))
            return a

        le_err, _ = theano.scan(fn=kl_div,
                                sequences=[para_err, weight, para_input_num]
                                )
        mask = (para_input_num > 0) * one_float32
        return mask * le_err

    def train(self, src_phrases, tar_phrases, src_tar_pair, config, model_name, start_iter=1, end_iter=26):
        """

        :param src_phrases:
        :param tar_phrases:
        :param src_tar_pair:
        :param config:
        :param model_name:
        :param start_iter:
        :param end_iter:
        :return:
        """
        n_epoch = config.n_epoch
        batch_size = config.batch_size
        size_src_word = self.src_embedding.size
        size_tar_word = self.tar_embedding.size

        train_index = align_batch_size(range(len(src_tar_pair)), batch_size)
        src_train = [phrase[WORD_INDEX] for phrase in src_phrases]
        tar_train = [phrase[WORD_INDEX] for phrase in tar_phrases]

        # 依长度对短语进行排序
        len_src_tar_pair = [(len(src_train[src_tar_pair[index][0]]), len(tar_train[src_tar_pair[index][1]]), index)
                            for index in train_index]
        len_src_tar_pair.sort()
        len_sort_index = np.array([index[2] for index in len_src_tar_pair], np.int32)
        src_len_index = np.array([index[0] for index in len_src_tar_pair], np.int32)
        tar_len_index = np.array([index[1] for index in len_src_tar_pair], np.int32)
        num_batch = len(len_sort_index) / batch_size

        if self.verbose:
            logger.debug("Train Details")
            logger.debug("Number of epochs: %d" % n_epoch)
            logger.debug("Size of Batches:  %d" % batch_size)
            logger.debug("Size phrase pair: %d" % len(src_tar_pair))
            logger.debug("Size Source word: %d" % size_src_word)
            logger.debug("Size Target word: %d" % size_tar_word)

        logger.info("Start training...")
        greedy_time = 0
        train_time = 0
        history_loss = []
        rae_util = RAEUtil(self.source_dim, normalize=self.normalize)
        random_batch_index = np.arange(num_batch)
        for i in xrange(1, n_epoch + 1):

            if i < start_iter:
                continue
            if i == start_iter and start_iter != 1:
                self.load_model("%s_iter%d.model" % (model_name, i - 1))
                load_random_state("%s_iter%d.model.rs" % (model_name, i - 1))
            if i >= end_iter:
                logger.info("Reach End the Iter")
                exit(1)

            # 每一轮迭代都随机生成负例
            src_pos, src_neg = BilingualPhraseRAE.generate_train_array_align(src_train, size_src_word)
            tar_pos, tar_neg = BilingualPhraseRAE.generate_train_array_align(tar_train, size_tar_word)
            src_pos_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
            src_neg_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
            tar_pos_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
            tar_neg_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
            src_para_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
            tar_para_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
            src_seq = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)
            tar_seq = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)
            src_seq_neg = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)
            tar_seq_neg = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)

            # 与训练的train_index大小相同
            # FOR LE
            src_seq_para = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)
            tar_seq_para = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)
            random_batch_index = np.random.permutation(random_batch_index)
            loss = []
            epoch_start = time.time()
            k = 0
            for j in random_batch_index:
                # 清空临时路径
                if self.verbose:
                    k += 1
                    print progress_bar_str(k, num_batch) + "\r",
                indexs = len_sort_index[j * batch_size: (j + 1) * batch_size]
                src_indexs = [src_tar_pair[index][0] for index in indexs]
                tar_indexs = [src_tar_pair[index][1] for index in indexs]
                src_len_max = np.max(src_len_index[j * batch_size: (j + 1) * batch_size])
                tar_len_max = np.max(tar_len_index[j * batch_size: (j + 1) * batch_size])
                # FOR LE
                src_para, src_para_weight = BilingualPhraseRAELLE.generate_para_train(src_phrases, src_indexs)
                tar_para, tar_para_weight = BilingualPhraseRAELLE.generate_para_train(tar_phrases, tar_indexs)
                src_para_len_index = np.array([len(src_train[index]) for index in src_para], np.int32)
                tar_para_len_index = np.array([len(tar_train[index]) for index in tar_para], np.int32)
                src_para_len_max = np.max(src_para_len_index[indexs])
                tar_para_len_max = np.max(tar_para_len_index[indexs])

                src_pos_train[:, :align_len], src_neg_train[:, :align_len] = src_pos[src_indexs], src_neg[src_indexs]
                tar_pos_train[:, :align_len], tar_neg_train[:, :align_len] = tar_pos[tar_indexs], tar_neg[tar_indexs]

                src_para_train[:, :align_len], src_para_weight_train = src_pos[src_para], src_para_weight
                tar_para_train[:, :align_len], tar_para_weight_train = tar_pos[tar_para], tar_para_weight

                # Generate greedy path
                time_temp = time.time()
                src_w = self.src_encoder.W.get_value(borrow=True)
                src_b = self.src_encoder.b.get_value(borrow=True)
                src_wr = self.src_encoder.Wr.get_value(borrow=True)
                src_br = self.src_encoder.br.get_value(borrow=True)
                tar_w = self.tar_encoder.W.get_value(borrow=True)
                tar_b = self.tar_encoder.b.get_value(borrow=True)
                tar_wr = self.tar_encoder.Wr.get_value(borrow=True)
                tar_br = self.tar_encoder.br.get_value(borrow=True)
                for sub_index in xrange(batch_size):
                    # Pos Path Generate
                    src_word_index = src_pos_train[sub_index]
                    tar_word_index = tar_pos_train[sub_index]
                    src_word_vec, tar_word_vec = self.get_indexs_embedding(src_word_index, tar_word_index)
                    src_seq[sub_index] = rae_util.build_greed_path(src_word_vec, src_w, src_b, src_wr, src_br,
                                                                   src_len_index[j * batch_size + sub_index])
                    tar_seq[sub_index] = rae_util.build_greed_path(tar_word_vec, tar_w, tar_b, tar_wr, tar_br,
                                                                   tar_len_index[j * batch_size + sub_index])
                    # Neg Path Generate
                    src_word_index = src_neg_train[sub_index]
                    tar_word_index = tar_neg_train[sub_index]
                    src_word_vec, tar_word_vec = self.get_indexs_embedding(src_word_index, tar_word_index)
                    src_seq_neg[sub_index] = rae_util.build_greed_path(src_word_vec, src_w, src_b, src_wr, src_br,
                                                                       src_len_index[j * batch_size + sub_index])
                    tar_seq_neg[sub_index] = rae_util.build_greed_path(tar_word_vec, tar_w, tar_b, tar_wr, tar_br,
                                                                       tar_len_index[j * batch_size + sub_index])
                    # Para Path Generate
                    src_word_index = src_para_train[sub_index]
                    tar_word_index = tar_para_train[sub_index]
                    src_word_vec, tar_word_vec = self.get_indexs_embedding(src_word_index, tar_word_index)
                    src_seq_para[sub_index] = rae_util.build_greed_path(src_word_vec, src_w, src_b, src_wr, src_br,
                                                                        src_para_len_index[indexs[sub_index]])
                    tar_seq_para[sub_index] = rae_util.build_greed_path(tar_word_vec, tar_w, tar_b, tar_wr, tar_br,
                                                                        tar_para_len_index[indexs[sub_index]])
                greedy_time += (time.time() - time_temp)
                # Train
                time_temp = time.time()
                # FOR LE
                result = self.train_batch(src_pos_train, tar_pos_train,
                                          src_seq, tar_seq,
                                          src_neg_train, tar_neg_train,
                                          src_seq_neg, tar_seq_neg,
                                          src_len_max, tar_len_max,
                                          src_para_train, tar_para_train,
                                          src_seq_para, tar_seq_para,
                                          src_para_weight_train, tar_para_weight_train,
                                          src_para_len_max, tar_para_len_max)
                logger.debug(result)
                if result[0] == np.inf or result[0] == np.nan:
                    logger.error("Detect nan or inf")
                    exit(-1)
                loss.append(result)
                del result
                train_time += (time.time() - time_temp)
            logger.info("epoch %d time: %f, Greedy %f, Train %f" % (i, time.time() - epoch_start, greedy_time, train_time))
            logger.info("epoch %d rec_err %f sem_err %f" % (i, np.mean(loss, axis=0)[1], np.mean(loss, axis=0)[2]))
            logger.info("epoch %d l2_err  %f sum_err %f" % (i, np.mean(loss, axis=0)[3], np.mean(loss, axis=0)[0]))
            # FOR LE
            logger.info("epoch %d le_err  %f" % (i, np.mean(loss, axis=0)[4]))
            history_loss.append(np.mean(loss, axis=0))
            del loss
            self.save_model("%s_iter%d.model" % (model_name, i))
            if len(history_loss) > 1 and abs(history_loss[-1][0] - history_loss[-2][0]) < 10e-6:
                logger.info("joint error reaches a local minima")
                break


class BilingualPhraseRAEISOMAP(BilingualPhraseRAE):
    def __init__(self, src_embedding, tar_embedding, initializer=default_initializer, config=None, verbose=True):
        super(BilingualPhraseRAEISOMAP, self).__init__(src_embedding, tar_embedding, initializer=initializer,
                                                       config=config, verbose=verbose)
        self.beta = config.beta
        self.trans_num = config.trans_num

        self.src_tran_max_len = T.ivector()  # (batch, )
        self.src_tran_input_seqs = T.itensor4()  # (batch, trans_num, word - 1, 3)
        self.src_tran_input_index = T.itensor3()  # (batch, trans_num, 2 * word - 1)
        self.src_tran_input_weight = T.fvector()  # (batch, )
        self.src_tran_input_num = T.ivector()  # (batch, )
        self.tar_tran_max_len = T.ivector()  # (batch, )
        self.tar_tran_input_seqs = T.itensor4()  # (batch, trans_num, word - 1, 3)
        self.tar_tran_input_index = T.itensor3()  # (batch, trans_num, 2 * word - 1)
        self.tar_tran_input_weight = T.fvector()  # (batch, )
        self.tar_tran_input_num = T.ivector()  # (batch, )

        self.src_loss_isomap = BilingualPhraseRAEISOMAP.get_isomap_err(encoder=self.tar_encoder,
                                                                       cross_w=self.t2s.Wl,
                                                                       cross_b=self.t2s.bl,
                                                                       src_pos_phrase=self.src_encoder.output,
                                                                       tar_pos_phrase=self.tar_encoder.output,
                                                                       indexs=self.src_tran_input_index,
                                                                       seqs=self.src_tran_input_seqs,
                                                                       max_len=self.src_tran_max_len,
                                                                       weight=self.src_tran_input_weight,
                                                                       nums=self.src_tran_input_num,
                                                                       max_num=self.trans_num,
                                                                       batch_size=self.batch_size
                                                                       )
        self.tar_loss_isomap = BilingualPhraseRAEISOMAP.get_isomap_err(encoder=self.src_encoder,
                                                                       cross_w=self.s2t.Wl,
                                                                       cross_b=self.s2t.bl,
                                                                       src_pos_phrase=self.tar_encoder.output,
                                                                       tar_pos_phrase=self.src_encoder.output,
                                                                       indexs=self.tar_tran_input_index,
                                                                       seqs=self.tar_tran_input_seqs,
                                                                       max_len=self.tar_tran_max_len,
                                                                       weight=self.tar_tran_input_weight,
                                                                       nums=self.tar_tran_input_num,
                                                                       max_num=self.trans_num,
                                                                       batch_size=self.batch_size
                                                                       )

        self.isomap_err = T.sum(self.src_loss_isomap) + T.sum(self.tar_loss_isomap)
        if self.alpha + self.beta > 1:
            import sys
            logger.error("alpha and beta is bigger than 1 %f" % self.alpha + self.beta)
            sys.stderr.write("alpha and beta is bigger than 1 %f" % self.alpha + self.beta)
            exit(-1)
        loss = (self.alpha * self.loss_rec + (1 - self.alpha - self.beta) * self.loss_sem + self.loss_l2 + self.beta * self.isomap_err) / config.batch_size
        grads = T.grad(loss, self.params)

        # Define Grad and Update
        updates = OrderedDict()
        for param, grad in zip(self.params, grads):
            updates[param] = param - grad * self.learning_rate

        self.train_batch = theano.function([self.src_encoder.input_index, self.tar_encoder.input_index,
                                            self.src_encoder.input_seqs, self.tar_encoder.input_seqs,
                                            self.src_encoder.neg_index, self.tar_encoder.neg_index,
                                            self.src_encoder.neg_seqs, self.tar_encoder.neg_seqs,
                                            self.src_encoder.max_len, self.tar_encoder.max_len,
                                            self.src_tran_input_index, self.tar_tran_input_index,
                                            self.src_tran_input_seqs, self.tar_tran_input_seqs,
                                            self.src_tran_input_weight, self.tar_tran_input_weight,
                                            self.src_tran_input_num, self.tar_tran_input_num,
                                            self.src_tran_max_len, self.tar_tran_max_len,
                                            ],
                                           outputs=[loss, self.loss_rec / config.batch_size,
                                                    self.loss_sem / config.batch_size, self.loss_l2 / config.batch_size,
                                                    self.isomap_err / config.batch_size,
                                                    ],
                                           updates=updates,
                                           )

        if verbose:
            logger.debug('Architecture of BRAE with ISOMAP built finished, summarized as below: ')
            logger.debug('Beta:             %f' % self.beta)
            logger.debug('Trans Num:        %d' % config.trans_num)

    def train(self, src_phrases, tar_phrases, src_tar_pair, config, model_name, start_iter=1, end_iter=26):
        """

        :param src_phrases:
        :param tar_phrases:
        :param src_tar_pair:
        :param config:
        :param model_name:
        :param start_iter:
        :param end_iter:
        :return:
        """
        n_epoch = config.n_epoch
        batch_size = config.batch_size
        size_src_word = self.src_embedding.size
        size_tar_word = self.tar_embedding.size

        train_index = align_batch_size(range(len(src_tar_pair)), batch_size)
        src_train = [phrase[WORD_INDEX] for phrase in src_phrases]
        tar_train = [phrase[WORD_INDEX] for phrase in tar_phrases]

        # 依长度对短语进行排序
        len_src_tar_pair = [(len(src_train[src_tar_pair[index][0]]), len(tar_train[src_tar_pair[index][1]]), index)
                            for index in train_index]
        len_src_tar_pair.sort()
        len_sort_index = np.array([index[2] for index in len_src_tar_pair], np.int32)
        src_len_index = np.array([index[0] for index in len_src_tar_pair], np.int32)
        tar_len_index = np.array([index[1] for index in len_src_tar_pair], np.int32)
        num_batch = len(len_sort_index) / batch_size

        if self.verbose:
            logger.debug("Train Details")
            logger.debug("Number of epochs: %d" % n_epoch)
            logger.debug("Size of Batches:  %d" % batch_size)
            logger.debug("Size phrase pair: %d" % len(src_tar_pair))
            logger.debug("Size Source word: %d" % size_src_word)
            logger.debug("Size Target word: %d" % size_tar_word)

        logger.info("Start training...")
        greedy_time = 0
        train_time = 0
        history_loss = []
        rae_util = RAEUtil(self.source_dim, normalize=self.normalize)
        random_batch_index = np.arange(num_batch)
        for i in xrange(1, n_epoch + 1):
            if i < start_iter:
                continue
            if i == start_iter and start_iter != 1:
                self.load_model("%s_iter%d.model" % (model_name, i - 1))
                load_random_state("%s_iter%d.model.rs" % (model_name, i - 1))
            if i >= end_iter:
                logger.info("Reach End the Iter")
                exit(1)
            # 每一轮迭代都随机生成负例
            src_pos, src_neg = BilingualPhraseRAE.generate_train_array_align(src_train, size_src_word)
            tar_pos, tar_neg = BilingualPhraseRAE.generate_train_array_align(tar_train, size_tar_word)
            src_pos_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
            src_neg_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
            tar_pos_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
            tar_neg_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
            src_trans_train = np.zeros((batch_size, self.trans_num, 2 * align_len - 1), dtype=np.int32)
            tar_trans_train = np.zeros((batch_size, self.trans_num, 2 * align_len - 1), dtype=np.int32)
            src_seq = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)
            tar_seq = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)
            src_seq_neg = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)
            tar_seq_neg = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)

            # 与训练的train_index大小相同
            # FOR ISOMAP
            # (train_size, trans_num, 2 * word - 1), (train_size), (train_size)
            src_seq_trans = np.zeros((batch_size, self.trans_num, self.length - 1, 3), dtype=np.int32)
            tar_seq_trans = np.zeros((batch_size, self.trans_num, self.length - 1, 3), dtype=np.int32)
            random_batch_index = np.random.permutation(random_batch_index)
            loss = []
            epoch_start = time.time()
            k = 0
            for j in random_batch_index:
                # 清空临时路径
                if self.verbose:
                    k += 1
                    print progress_bar_str(k, num_batch) + "\r",
                indexs = len_sort_index[j * batch_size: (j + 1) * batch_size]
                src_indexs = [src_tar_pair[index][0] for index in indexs]
                tar_indexs = [src_tar_pair[index][1] for index in indexs]
                src_len_max = np.max(src_len_index[j * batch_size: (j + 1) * batch_size])
                tar_len_max = np.max(tar_len_index[j * batch_size: (j + 1) * batch_size])
                # FOR ISOMAP
                src_trans, src_trans_weight, src_trans_num = BilingualPhraseRAEISOMAP.generate_trans_train(src_phrases,
                                                                                                           src_indexs,
                                                                                                           tar_indexs,
                                                                                                           self.trans_num)
                tar_trans, tar_trans_weight, tar_trans_num = BilingualPhraseRAEISOMAP.generate_trans_train(tar_phrases,
                                                                                                           tar_indexs,
                                                                                                           src_indexs,
                                                                                                           self.trans_num)
                src_seq_trans *= 0
                tar_seq_trans *= 0
                src_trans_len_index = np.array([[len(tar_train[index]) for index in tran_indexs]
                                                for tran_indexs in src_trans], np.int32)
                tar_trans_len_index = np.array([[len(src_train[index]) for index in tran_indexs]
                                                for tran_indexs in tar_trans], np.int32)
                src_trans_len_max = np.max(src_trans_len_index, axis=0)
                tar_trans_len_max = np.max(tar_trans_len_index, axis=0)

                src_pos_train[:, :align_len], src_neg_train[:, :align_len] = src_pos[src_indexs], src_neg[src_indexs]
                tar_pos_train[:, :align_len], tar_neg_train[:, :align_len] = tar_pos[tar_indexs], tar_neg[tar_indexs]

                src_trans_train[:, :, :align_len], tar_trans_train[:, :, :align_len] = tar_pos[src_trans], src_pos[tar_trans]
                src_trans_weight_train, src_trans_num_train = src_trans_weight, src_trans_num
                tar_trans_weight_train, tar_trans_num_train = tar_trans_weight, tar_trans_num

                # Generate greedy path
                time_temp = time.time()
                src_w = self.src_encoder.W.get_value(borrow=True)
                src_b = self.src_encoder.b.get_value(borrow=True)
                src_wr = self.src_encoder.Wr.get_value(borrow=True)
                src_br = self.src_encoder.br.get_value(borrow=True)
                tar_w = self.tar_encoder.W.get_value(borrow=True)
                tar_b = self.tar_encoder.b.get_value(borrow=True)
                tar_wr = self.tar_encoder.Wr.get_value(borrow=True)
                tar_br = self.tar_encoder.br.get_value(borrow=True)
                for sub_index in xrange(batch_size):
                    # Pos Path Generate
                    src_word_index = src_pos_train[sub_index]
                    tar_word_index = tar_pos_train[sub_index]
                    src_word_vec, tar_word_vec = self.get_indexs_embedding(src_word_index, tar_word_index)
                    src_seq[sub_index] = rae_util.build_greed_path(src_word_vec, src_w, src_b, src_wr, src_br,
                                                                   src_len_index[j * batch_size + sub_index])
                    tar_seq[sub_index] = rae_util.build_greed_path(tar_word_vec, tar_w, tar_b, tar_wr, tar_br,
                                                                   tar_len_index[j * batch_size + sub_index])
                    # Neg Path Generate
                    src_word_index = src_neg_train[sub_index]
                    tar_word_index = tar_neg_train[sub_index]
                    src_word_vec, tar_word_vec = self.get_indexs_embedding(src_word_index, tar_word_index)
                    src_seq_neg[sub_index] = rae_util.build_greed_path(src_word_vec, src_w, src_b, src_wr, src_br,
                                                                       src_len_index[j * batch_size + sub_index])
                    tar_seq_neg[sub_index] = rae_util.build_greed_path(tar_word_vec, tar_w, tar_b, tar_wr, tar_br,
                                                                       tar_len_index[j * batch_size + sub_index])
                    # Trans Path Generate
                    len_index = indexs[sub_index]
                    for tran_index in xrange(src_trans_num_train[sub_index]):
                        # src trans -> tar
                        tar_word_index = src_trans_train[sub_index][tran_index]
                        # output is list or dict
                        tar_word_vec = self.get_tar_indexs_embedding(tar_word_index)
                        src_seq_trans[sub_index][tran_index] = rae_util.build_greed_path(tar_word_vec,
                                                                                         tar_w, tar_b, tar_wr, tar_br,
                                                                                         src_trans_len_index[len_index][tran_index])
                    for tran_index in xrange(tar_trans_num_train[sub_index]):
                        # tar trans -> src
                        src_word_index = tar_trans_train[sub_index][tran_index]
                        # output is list or dict
                        src_word_vec = self.get_src_indexs_embedding(src_word_index)
                        tar_seq_trans[sub_index][tran_index] = rae_util.build_greed_path(src_word_vec,
                                                                                         src_w, src_b, src_wr, src_br,
                                                                                         tar_trans_len_index[len_index][tran_index])

                greedy_time += (time.time() - time_temp)
                # Train
                time_temp = time.time()
                # FOR ISOMAP
                result = self.train_batch(src_pos_train, tar_pos_train,
                                          src_seq, tar_seq,
                                          src_neg_train, tar_neg_train,
                                          src_seq_neg, tar_seq_neg,
                                          src_len_max, tar_len_max,
                                          src_trans_train, tar_trans_train,
                                          src_seq_trans, tar_seq_trans,
                                          src_trans_weight_train, tar_trans_weight_train,
                                          src_trans_num_train, tar_trans_num_train,
                                          src_trans_len_max, tar_trans_len_max)
                logger.debug(result)
                if result[0] == np.inf or result[0] == np.nan:
                    logger.info("Detect nan or inf")
                    exit(-1)
                loss.append(result)
                del result
                train_time += (time.time() - time_temp)
            logger.info("epoch %d time: %f, Greedy %f, Train %f" % (i, time.time() - epoch_start, greedy_time, train_time))
            logger.info("epoch %d rec_err %f sem_err %f" % (i, np.mean(loss, axis=0)[1], np.mean(loss, axis=0)[2]))
            logger.info("epoch %d l2_err  %f sum_err %f" % (i, np.mean(loss, axis=0)[3], np.mean(loss, axis=0)[0]))
            # FOR ISOMAP
            logger.info("epoch %d isomap_err  %f" % (i, np.mean(loss, axis=0)[4]))
            history_loss.append(np.mean(loss, axis=0))
            del loss
            self.save_model("%s_iter%d.model" % (model_name, i))
            save_random_state("%s_iter%d.model.rs" % (model_name, i))
            if len(history_loss) > 1 and abs(history_loss[-1][0] - history_loss[-2][0]) < 10e-6:
                logger.info("joint error reaches a local minima")
                break

    @staticmethod
    def generate_trans_train(src_phrases, src_train_index, tar_train_index, max_trans_num):
        trans_words = np.zeros((len(src_train_index), max_trans_num), dtype=np.int32)
        # 第0维代表正例的权重 其余维数代表翻译候选项的权重
        trans_weight = np.zeros((len(src_train_index), max_trans_num + 1), dtype=theano.config.floatX)
        trans_weight[:, 0] = 1  # 防止翻译概率为0
        trans_nums = np.zeros((len(src_train_index), ), dtype=np.int32)
        for ind in xrange(len(src_train_index)):
            src_ind = src_train_index[ind]
            tar_ind = tar_train_index[ind]
            src_trans_candidate = src_phrases[src_ind][TRAN_INDEX]
            if src_trans_candidate is None or len(src_trans_candidate) == 0 or tar_ind not in src_trans_candidate:
                continue
            # 从大到小选择trans_num个翻译候选项
            trans_num = min(max_trans_num, len(src_phrases[src_ind][TRAN_INDEX]) - 1)
            if trans_num < 1:
                continue
            trans_weight[ind][0] = src_trans_candidate[tar_ind]
            trans_i = 0
            for key, value in src_trans_candidate.iteritems():
                if key == tar_ind:
                    continue
                else:
                    trans_words[ind][trans_i] = key
                    trans_weight[ind][trans_i + 1] = value
                    trans_i += 1
                    if trans_i >= trans_num:
                        break
            if np.sum(trans_weight[ind]) != 0:
                trans_weight[ind] /= np.sum(trans_weight[ind])
            else:
                trans_weight[ind] *= 0
                trans_weight[ind][0] = 1
            trans_nums[ind] = trans_num
        return trans_words, trans_weight, trans_nums

    @staticmethod
    def get_isomap_err(encoder, cross_w, cross_b, src_pos_phrase, tar_pos_phrase, indexs, seqs, max_len,
                       weight, nums, max_num, batch_size, name=""):

        sem_dis = conditional_probabilities(src_pos_phrase, cross_w, cross_b, tar_pos_phrase)
        neg_sem_scores = shared_zero_matrix((batch_size, max_num), 'isomap_dis_src')
        for i in xrange(max_num):
            neg_rep, _ = encoder.build_phrase_graph(indexs[:, i, :], seqs[:, i, :, :], max_len[i], name + "trans_graph")
            mask = (nums > i) * one_float32
            neg_sem_score = conditional_probabilities(src_pos_phrase, cross_w, cross_b, neg_rep)
            neg_sem_scores = T.set_subtensor(neg_sem_scores[:, i], neg_sem_score * mask)
        sum_exp_scores = sem_dis + T.sum(neg_sem_scores, axis=1)
        sem_dis /= sum_exp_scores
        neg_sem_scores /= sum_exp_scores[:, None]

        def kl_div(score, wei, num, pos_score, pos_weight):
            a = T.sum(wei[:num] * T.log(wei[:num] / score[:num]))
            b = pos_weight * T.log(pos_weight / pos_score)
            return a + b

        mask = (nums > 0) * one_float32
        err, _ = theano.scan(kl_div,
                             sequences=[neg_sem_scores,
                                        weight[:, 1:],
                                        nums,
                                        sem_dis,
                                        weight[:, 0]
                                        ]
                             )
        return err * mask


class GraphBRAE(BilingualPhraseRAE):
    def __init__(self, src_embedding, tar_embedding, initializer=default_initializer, config=None, verbose=True):
        super(GraphBRAE, self).__init__(src_embedding, tar_embedding,
                                        initializer=initializer, config=config, verbose=verbose)
        self.beta = config.beta
        self.gama = config.gama
        self.para = config.para
        self.trans = config.trans
        self.para_num = config.para_num
        self.trans_num = config.trans_num

        # For LE
        self.src_para_max_len = T.ivector()
        self.src_para_input_seqs = T.itensor4()  # (batch, word - 1, 3)
        self.src_para_input_index = T.itensor3()  # (batch, 2 * word - 1)
        self.src_para_input_weight = T.fmatrix()  # (batch, )
        self.src_para_input_num = T.ivector()
        self.tar_para_max_len = T.ivector()
        self.tar_para_input_seqs = T.itensor4()  # (batch, word - 1, 3)
        self.tar_para_input_index = T.itensor3()  # (batch, 2 * word - 1)
        self.tar_para_input_weight = T.fmatrix()  # (batch, )
        self.tar_para_input_num = T.ivector()

        # For ISOMAP
        self.src_tran_max_len = T.ivector()  # (batch, )
        self.src_tran_input_seqs = T.itensor4()  # (batch, trans_num, word - 1, 3)
        self.src_tran_input_index = T.itensor3()  # (batch, trans_num, 2 * word - 1)
        self.src_tran_input_weight = T.fmatrix()  # (batch, )
        self.src_tran_input_num = T.ivector()  # (batch, )
        self.tar_tran_max_len = T.ivector()  # (batch, )
        self.tar_tran_input_seqs = T.itensor4()  # (batch, trans_num, word - 1, 3)
        self.tar_tran_input_index = T.itensor3()  # (batch, trans_num, 2 * word - 1)
        self.tar_tran_input_weight = T.fmatrix()  # (batch, )
        self.tar_tran_input_num = T.ivector()  # (batch, )

        # (batch, dim)
        self.src_loss_le = BilingualPhraseRAELLE.get_err_le(encoder=self.src_encoder,
                                                            pos_phrases=self.src_encoder.output,
                                                            weight=self.src_para_input_weight,
                                                            para_input_index=self.src_para_input_index,
                                                            para_input_seqs=self.src_para_input_seqs,
                                                            para_max_len=self.src_para_max_len,
                                                            para_input_num=self.src_para_input_num,
                                                            max_num=self.para_num,
                                                            batch_size=config.batch_size,
                                                            name="batch_src")
        self.tar_loss_le = BilingualPhraseRAELLE.get_err_le(encoder=self.tar_encoder,
                                                            pos_phrases=self.tar_encoder.output,
                                                            weight=self.tar_para_input_weight,
                                                            para_input_index=self.tar_para_input_index,
                                                            para_input_seqs=self.tar_para_input_seqs,
                                                            para_max_len=self.tar_para_max_len,
                                                            para_input_num=self.tar_para_input_num,
                                                            max_num=self.para_num,
                                                            batch_size=config.batch_size,
                                                            name="batch_tar")
        self.loss_le = T.sum(self.src_loss_le) + T.sum(self.tar_loss_le)

        self.src_loss_isomap = BilingualPhraseRAEISOMAP.get_isomap_err(encoder=self.tar_encoder,
                                                                       cross_w=self.t2s.Wl,
                                                                       cross_b=self.t2s.bl,
                                                                       src_pos_phrase=self.src_encoder.output,
                                                                       tar_pos_phrase=self.tar_encoder.output,
                                                                       indexs=self.src_tran_input_index,
                                                                       seqs=self.src_tran_input_seqs,
                                                                       max_len=self.src_tran_max_len,
                                                                       weight=self.src_tran_input_weight,
                                                                       nums=self.src_tran_input_num,
                                                                       max_num=self.trans_num,
                                                                       batch_size=self.batch_size
                                                                       )
        self.tar_loss_isomap = BilingualPhraseRAEISOMAP.get_isomap_err(encoder=self.src_encoder,
                                                                       cross_w=self.s2t.Wl,
                                                                       cross_b=self.s2t.bl,
                                                                       src_pos_phrase=self.tar_encoder.output,
                                                                       tar_pos_phrase=self.src_encoder.output,
                                                                       indexs=self.tar_tran_input_index,
                                                                       seqs=self.tar_tran_input_seqs,
                                                                       max_len=self.tar_tran_max_len,
                                                                       weight=self.tar_tran_input_weight,
                                                                       nums=self.tar_tran_input_num,
                                                                       max_num=self.trans_num,
                                                                       batch_size=self.batch_size
                                                                       )
        self.loss_isomap = T.sum(self.src_loss_isomap) + T.sum(self.tar_loss_isomap)

        loss = self.alpha * self.loss_rec
        loss += self.beta * self.loss_sem
        loss_para = loss
        loss_trans = loss
        loss += self.gama * self.loss_le
        loss_para += self.gama * self.loss_le
        loss += self.gama * self.loss_isomap
        loss_trans += self.gama * self.loss_isomap
        '''
        self.src_norm_word_index = T.extra_ops.Unique()(T.concatenate([T.extra_ops.Unique()(self.src_encoder.input_index),
                                                      T.extra_ops.Unique()(self.src_encoder.neg_index),
                                                      T.extra_ops.Unique()(self.src_para_input_index),
                                                      T.extra_ops.Unique()(self.src_tran_input_index)]))
        self.tar_norm_word_index = T.extra_ops.Unique()(T.concatenate([T.extra_ops.Unique()(self.tar_encoder.input_index),
                                                                     T.extra_ops.Unique()(self.tar_encoder.neg_index),
                                                                     T.extra_ops.Unique()(self.tar_para_input_index),
                                                                     T.extra_ops.Unique()(self.tar_tran_input_index)]))
        '''
        loss += self.loss_l2
        loss_para += self.loss_l2
        loss_trans += self.loss_l2
        loss /= config.batch_size
        loss_para /= config.batch_size
        loss_trans /= config.batch_size

        # Define Grad and Update
        grads = T.grad(loss, self.params)
        updates = OrderedDict()
        for param, grad in zip(self.params, grads):
            updates[param] = param - grad * self.learning_rate

        para_grads = T.grad(loss_para, self.params)
        para_updates = OrderedDict()
        for param, grad in zip(self.params, para_grads):
            para_updates[param] = param - grad * self.learning_rate

        trans_grads = T.grad(loss_trans, self.params)
        trans_updates = OrderedDict()
        for param, grad in zip(self.params, trans_grads):
            trans_updates[param] = param - grad * self.learning_rate

        self.train_batch = theano.function([self.src_encoder.input_index, self.tar_encoder.input_index,
                                            self.src_encoder.input_seqs, self.tar_encoder.input_seqs,
                                            self.src_encoder.neg_index, self.tar_encoder.neg_index,
                                            self.src_encoder.neg_seqs, self.tar_encoder.neg_seqs,
                                            self.src_encoder.max_len, self.tar_encoder.max_len,
                                            self.src_para_input_index, self.tar_para_input_index,
                                            self.src_para_input_seqs, self.tar_para_input_seqs,
                                            self.src_para_input_weight, self.tar_para_input_weight,
                                            self.src_para_max_len, self.tar_para_max_len,
                                            self.src_para_input_num, self.tar_para_input_num,
                                            self.src_tran_input_index, self.tar_tran_input_index,
                                            self.src_tran_input_seqs, self.tar_tran_input_seqs,
                                            self.src_tran_input_weight, self.tar_tran_input_weight,
                                            self.src_tran_input_num, self.tar_tran_input_num,
                                            self.src_tran_max_len, self.tar_tran_max_len,
                                            ],
                                           outputs=[loss, self.loss_rec / config.batch_size,
                                                    self.loss_sem / config.batch_size,
                                                    self.loss_l2 / config.batch_size,
                                                    self.loss_isomap / config.batch_size,
                                                    self.loss_le / config.batch_size,
                                                    ],
                                           updates=updates,
                                           #mode=NanGuardMode(nan_is_error=True, inf_is_error=True,big_is_error=True),
                                           name="gbrae_objective")
        self.train_para_batch = theano.function([self.src_encoder.input_index, self.tar_encoder.input_index,
                                                 self.src_encoder.input_seqs, self.tar_encoder.input_seqs,
                                                 self.src_encoder.neg_index, self.tar_encoder.neg_index,
                                                 self.src_encoder.neg_seqs, self.tar_encoder.neg_seqs,
                                                 self.src_encoder.max_len, self.tar_encoder.max_len,
                                                 self.src_para_input_index, self.tar_para_input_index,
                                                 self.src_para_input_seqs, self.tar_para_input_seqs,
                                                 self.src_para_input_weight, self.tar_para_input_weight,
                                                 self.src_para_max_len, self.tar_para_max_len,
                                                 self.src_para_input_num, self.tar_para_input_num,
                                                 ],
                                                outputs=[loss_para, self.loss_rec / config.batch_size,
                                                         self.loss_sem / config.batch_size,
                                                         self.loss_l2 / config.batch_size,
                                                         self.loss_le / config.batch_size,
                                                         ],
                                                updates=para_updates,
                                                name="gbrae_para_objective")
        self.train_trans_batch = theano.function([self.src_encoder.input_index, self.tar_encoder.input_index,
                                                  self.src_encoder.input_seqs, self.tar_encoder.input_seqs,
                                                  self.src_encoder.neg_index, self.tar_encoder.neg_index,
                                                  self.src_encoder.neg_seqs, self.tar_encoder.neg_seqs,
                                                  self.src_encoder.max_len, self.tar_encoder.max_len,
                                                  self.src_tran_input_index, self.tar_tran_input_index,
                                                  self.src_tran_input_seqs, self.tar_tran_input_seqs,
                                                  self.src_tran_input_weight, self.tar_tran_input_weight,
                                                  self.src_tran_input_num, self.tar_tran_input_num,
                                                  self.src_tran_max_len, self.tar_tran_max_len,
                                                  ],
                                                 outputs=[loss_trans, self.loss_rec / config.batch_size,
                                                          self.loss_sem / config.batch_size,
                                                          self.loss_l2 / config.batch_size,
                                                          self.loss_isomap / config.batch_size,
                                                          ],
                                                 updates=trans_updates,
                                                 name="gbrae_trans_objective")

        self.predict_batch = theano.function([self.src_encoder.input_index, self.tar_encoder.input_index,
                                              self.src_encoder.input_seqs, self.tar_encoder.input_seqs,
                                              self.src_encoder.neg_index, self.tar_encoder.neg_index,
                                              self.src_encoder.neg_seqs, self.tar_encoder.neg_seqs,
                                              self.src_encoder.max_len, self.tar_encoder.max_len,
                                              self.src_para_input_index, self.tar_para_input_index,
                                              self.src_para_input_seqs, self.tar_para_input_seqs,
                                              self.src_para_input_weight, self.tar_para_input_weight,
                                              self.src_para_max_len, self.tar_para_max_len,
                                              self.src_para_input_num, self.tar_para_input_num,
                                              self.src_tran_input_index, self.tar_tran_input_index,
                                              self.src_tran_input_seqs, self.tar_tran_input_seqs,
                                              self.src_tran_input_weight, self.tar_tran_input_weight,
                                              self.src_tran_input_num, self.tar_tran_input_num,
                                              self.src_tran_max_len, self.tar_tran_max_len,
                                              ],
                                             outputs=[loss, self.loss_rec / config.batch_size,
                                                      self.loss_sem / config.batch_size,
                                                      self.loss_l2 / config.batch_size,
                                                      self.loss_isomap / config.batch_size,
                                                      self.loss_le / config.batch_size,
                                                      ],
                                             name="gbrae_objective_predict")

        self.predict_para_batch = theano.function([self.src_encoder.input_index, self.tar_encoder.input_index,
                                                   self.src_encoder.input_seqs, self.tar_encoder.input_seqs,
                                                   self.src_encoder.neg_index, self.tar_encoder.neg_index,
                                                   self.src_encoder.neg_seqs, self.tar_encoder.neg_seqs,
                                                   self.src_encoder.max_len, self.tar_encoder.max_len,
                                                   self.src_para_input_index, self.tar_para_input_index,
                                                   self.src_para_input_seqs, self.tar_para_input_seqs,
                                                   self.src_para_input_weight, self.tar_para_input_weight,
                                                   self.src_para_max_len, self.tar_para_max_len,
                                                   self.src_para_input_num, self.tar_para_input_num,
                                                   ],
                                                  outputs=[loss_para, self.loss_rec / config.batch_size,
                                                           self.loss_sem / config.batch_size,
                                                           self.loss_l2 / config.batch_size,
                                                           self.loss_le / config.batch_size,
                                                           ],
                                                  name="gbrae_para_predict")

        self.predict_trans_batch = theano.function([self.src_encoder.input_index, self.tar_encoder.input_index,
                                                    self.src_encoder.input_seqs, self.tar_encoder.input_seqs,
                                                    self.src_encoder.neg_index, self.tar_encoder.neg_index,
                                                    self.src_encoder.neg_seqs, self.tar_encoder.neg_seqs,
                                                    self.src_encoder.max_len, self.tar_encoder.max_len,
                                                    self.src_tran_input_index, self.tar_tran_input_index,
                                                    self.src_tran_input_seqs, self.tar_tran_input_seqs,
                                                    self.src_tran_input_weight, self.tar_tran_input_weight,
                                                    self.src_tran_input_num, self.tar_tran_input_num,
                                                    self.src_tran_max_len, self.tar_tran_max_len,
                                                    ],
                                                   outputs=[loss_trans, self.loss_rec / config.batch_size,
                                                            self.loss_sem / config.batch_size,
                                                            self.loss_l2 / config.batch_size,
                                                            self.loss_isomap / config.batch_size,
                                                            ],
                                                   name="gbrae_trans_predict")

        if verbose:
            logger.debug('Architecture of {} built finished'.format(self.__class__.__name__))
            logger.debug('Alpha: %f' % self.alpha)
            logger.debug('Beta:  %f' % self.beta)
            logger.debug('Gama:  %f' % self.gama)

    def train(self, src_phrases, tar_phrases, src_tar_pair, config, model_name, start_iter=1, end_iter=26):
        """

        :param src_phrases:
        :param tar_phrases:
        :param src_tar_pair:
        :param config:
        :param model_name:
        :param start_iter:
        :param end_iter:
        :return:
        """
        n_epoch = config.n_epoch
        batch_size = config.batch_size
        size_src_word = self.src_embedding.size
        size_tar_word = self.tar_embedding.size

        train_index = align_batch_size(range(len(src_tar_pair)), batch_size)
        src_train = [phrase[WORD_INDEX] for phrase in src_phrases]
        tar_train = [phrase[WORD_INDEX] for phrase in tar_phrases]

        # 依长度对短语进行排序
        len_src_tar_pair = [(len(src_train[src_tar_pair[index][0]]), len(tar_train[src_tar_pair[index][1]]), index)
                            for index in train_index]
        len_src_tar_pair.sort()
        len_sort_index = np.array([index[2] for index in len_src_tar_pair], np.int32)
        src_len_index = np.array([index[0] for index in len_src_tar_pair], np.int32)
        tar_len_index = np.array([index[1] for index in len_src_tar_pair], np.int32)
        num_batch = len(len_sort_index) / batch_size

        if self.verbose:
            logger.debug("Train Details")
            logger.debug("Number of epochs: %d" % n_epoch)
            logger.debug("Size of Batches:  %d" % batch_size)
            logger.debug("Size phrase pair: %d" % len(src_tar_pair))
            logger.debug("Size Source word: %d" % size_src_word)
            logger.debug("Size Target word: %d" % size_tar_word)

        logger.info("Start training...")
        history_loss = []
        random_batch_index = np.arange(num_batch)
        for i in xrange(1, n_epoch + 1):

            if i < start_iter:
                continue
            if i == start_iter and start_iter != 1:
                self.load_model("%s_iter%d.model" % (model_name, i - 1))
                load_random_state("%s_iter%d.model.rs" % (model_name, i - 1))
            if i >= end_iter:
                logger.info("Reach End the Iter")
                exit(1)

            rae_util = RAEUtil(self.source_dim, normalize=self.normalize)
            # 每一轮迭代都随机生成负例
            logger.info("Generate Neg Instance ...")
            src_pos, src_neg = BilingualPhraseRAE.generate_train_array_align(src_train, size_src_word)
            tar_pos, tar_neg = BilingualPhraseRAE.generate_train_array_align(tar_train, size_tar_word)
            src_pos_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
            src_neg_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
            tar_pos_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
            tar_neg_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
            src_para_train = np.zeros((batch_size, self.para_num, 2 * align_len - 1), dtype=np.int32)
            tar_para_train = np.zeros((batch_size, self.para_num, 2 * align_len - 1), dtype=np.int32)
            src_trans_train = np.zeros((batch_size, self.trans_num, 2 * align_len - 1), dtype=np.int32)
            tar_trans_train = np.zeros((batch_size, self.trans_num, 2 * align_len - 1), dtype=np.int32)
            src_seq = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)
            tar_seq = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)
            src_seq_neg = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)
            tar_seq_neg = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)
            # FOR LE
            src_seq_para = np.zeros((batch_size, self.para_num, self.length - 1, 3), dtype=np.int32)
            tar_seq_para = np.zeros((batch_size, self.para_num, self.length - 1, 3), dtype=np.int32)

            # FOR ISOMAP
            src_seq_trans = np.zeros((batch_size, self.trans_num, self.length - 1, 3), dtype=np.int32)
            tar_seq_trans = np.zeros((batch_size, self.trans_num, self.length - 1, 3), dtype=np.int32)
            random_batch_index = np.random.permutation(random_batch_index)

            loss = []
            epoch_start = time.time()
            generate_time = 0
            greedy_time = 0
            train_time = 0
            k = 0
            for j in random_batch_index:
                k += 1
                if k % 5000 == 0:
                    while gc.collect() > 0:
                        pass
                # 清空临时路径
                if self.verbose:
                    print progress_bar_str(k, num_batch) + "\r",
                temp_time = time.time()
                indexs = len_sort_index[j * batch_size: (j + 1) * batch_size]
                src_indexs = [src_tar_pair[index][0] for index in indexs]
                tar_indexs = [src_tar_pair[index][1] for index in indexs]
                src_len_max = np.max(src_len_index[j * batch_size: (j + 1) * batch_size])
                tar_len_max = np.max(tar_len_index[j * batch_size: (j + 1) * batch_size])
                # FOR LE
                if self.para:
                    src_para, src_para_weight, src_para_num = BilingualPhraseRAELLE.generate_para_train2(src_phrases,
                                                                                                         src_indexs,
                                                                                                         self.para_num)
                    tar_para, tar_para_weight, tar_para_num = BilingualPhraseRAELLE.generate_para_train2(tar_phrases,
                                                                                                         tar_indexs,
                                                                                                         self.para_num)
                    src_seq_para *= 0
                    tar_seq_para *= 0
                    src_para_len_index = np.array([[len(src_train[index]) for index in indexs]
                                                   for indexs in src_para], np.int32)
                    tar_para_len_index = np.array([[len(tar_train[index]) for index in indexs]
                                                   for indexs in tar_para], np.int32)
                    src_para_len_max = np.max(src_para_len_index, axis=0)
                    tar_para_len_max = np.max(tar_para_len_index, axis=0)


                # FOR ISOMAP
                if self.trans:
                    src_trans, src_trans_weight, src_trans_num = BilingualPhraseRAEISOMAP.generate_trans_train(
                        src_phrases,
                        src_indexs,
                        tar_indexs,
                        self.trans_num)
                    tar_trans, tar_trans_weight, tar_trans_num = BilingualPhraseRAEISOMAP.generate_trans_train(
                        tar_phrases,
                        tar_indexs,
                        src_indexs,
                        self.trans_num)
                    src_seq_trans *= 0
                    tar_seq_trans *= 0
                    src_trans_len_index = np.array([[len(tar_train[index]) for index in tran_indexs]
                                                    for tran_indexs in src_trans], np.int32)
                    tar_trans_len_index = np.array([[len(src_train[index]) for index in tran_indexs]
                                                    for tran_indexs in tar_trans], np.int32)
                    src_trans_len_max = np.max(src_trans_len_index, axis=0)
                    tar_trans_len_max = np.max(tar_trans_len_index, axis=0)

                src_pos_train[:, :align_len], src_neg_train[:, :align_len] = src_pos[src_indexs], src_neg[src_indexs]
                tar_pos_train[:, :align_len], tar_neg_train[:, :align_len] = tar_pos[tar_indexs], tar_neg[tar_indexs]

                if self.para:
                    src_para_train[:, :, :align_len], src_para_weight_train = src_pos[src_para], src_para_weight
                    tar_para_train[:, :, :align_len], tar_para_weight_train = tar_pos[tar_para], tar_para_weight
                    src_para_num_train, tar_para_num_train = src_para_num, tar_para_num

                if self.trans:
                    src_trans_train[:, :, :align_len], tar_trans_train[:, :, :align_len] = tar_pos[src_trans], src_pos[tar_trans]
                    src_trans_weight_train, src_trans_num_train = src_trans_weight, src_trans_num
                    tar_trans_weight_train, tar_trans_num_train = tar_trans_weight, tar_trans_num

                generate_time += (time.time() - temp_time)
                # Generate greedy path
                time_temp = time.time()
                src_w = self.src_encoder.W.get_value(borrow=True)
                src_b = self.src_encoder.b.get_value(borrow=True)
                src_wr = self.src_encoder.Wr.get_value(borrow=True)
                src_br = self.src_encoder.br.get_value(borrow=True)
                tar_w = self.tar_encoder.W.get_value(borrow=True)
                tar_b = self.tar_encoder.b.get_value(borrow=True)
                tar_wr = self.tar_encoder.Wr.get_value(borrow=True)
                tar_br = self.tar_encoder.br.get_value(borrow=True)
                for sub_index in xrange(batch_size):
                    # Pos Path Generate
                    src_word_index = src_pos_train[sub_index]
                    tar_word_index = tar_pos_train[sub_index]
                    src_word_vec, tar_word_vec = self.get_indexs_embedding(src_word_index, tar_word_index)
                    src_seq[sub_index] = rae_util.build_greed_path(src_word_vec, src_w, src_b, src_wr, src_br,
                                                                   src_len_index[j * batch_size + sub_index])
                    tar_seq[sub_index] = rae_util.build_greed_path(tar_word_vec, tar_w, tar_b, tar_wr, tar_br,
                                                                   tar_len_index[j * batch_size + sub_index])
                    # Neg Path Generate
                    src_word_index = src_neg_train[sub_index]
                    tar_word_index = tar_neg_train[sub_index]
                    src_word_vec, tar_word_vec = self.get_indexs_embedding(src_word_index, tar_word_index)
                    src_seq_neg[sub_index] = rae_util.build_greed_path(src_word_vec, src_w, src_b, src_wr, src_br,
                                                                       src_len_index[j * batch_size + sub_index])
                    tar_seq_neg[sub_index] = rae_util.build_greed_path(tar_word_vec, tar_w, tar_b, tar_wr, tar_br,
                                                                       tar_len_index[j * batch_size + sub_index])
                    # Para Path Generate
                    if self.para:
                        for para_index in xrange(src_para_num_train[sub_index]):
                            src_word_index = src_para_train[sub_index][para_index]
                            src_word_vec = self.get_src_indexs_embedding(src_word_index)
                            src_seq_para[sub_index][para_index] = rae_util.build_greed_path(src_word_vec,
                                                                                            src_w, src_b, src_wr, src_br,
                                                                                            src_para_len_index[sub_index][para_index])
                        for para_index in xrange(tar_para_num_train[sub_index]):
                            tar_word_index = tar_para_train[sub_index][para_index]
                            tar_word_vec = self.get_tar_indexs_embedding(tar_word_index)
                            tar_seq_para[sub_index][para_index] = rae_util.build_greed_path(tar_word_vec,
                                                                                            tar_w, tar_b, tar_wr, tar_br,
                                                                                            tar_para_len_index[sub_index][para_index])

                    if self.trans:
                        for tran_index in xrange(src_trans_num_train[sub_index]):
                            # src trans -> tar
                            tar_word_index = src_trans_train[sub_index][tran_index]
                            # output is list or dict
                            tar_word_vec = self.get_tar_indexs_embedding(tar_word_index)
                            src_seq_trans[sub_index][tran_index] = rae_util.build_greed_path(tar_word_vec,
                                                                                             tar_w, tar_b, tar_wr,
                                                                                             tar_br,
                                                                                             src_trans_len_index[
                                                                                                 sub_index][tran_index])
                        for tran_index in xrange(tar_trans_num_train[sub_index]):
                            # tar trans -> src
                            src_word_index = tar_trans_train[sub_index][tran_index]
                            # output is list or dict
                            src_word_vec = self.get_src_indexs_embedding(src_word_index)
                            tar_seq_trans[sub_index][tran_index] = rae_util.build_greed_path(src_word_vec,
                                                                                             src_w, src_b, src_wr,
                                                                                             src_br,
                                                                                             tar_trans_len_index[
                                                                                                 sub_index][tran_index])

                greedy_time += (time.time() - time_temp)
                # Train
                time_temp = time.time()
                if self.para and self.trans:
                    result = self.train_batch(src_pos_train, tar_pos_train,
                                              src_seq, tar_seq,
                                              src_neg_train, tar_neg_train,
                                              src_seq_neg, tar_seq_neg,
                                              src_len_max, tar_len_max,
                                              src_para_train, tar_para_train,
                                              src_seq_para, tar_seq_para,
                                              src_para_weight_train, tar_para_weight_train,
                                              src_para_len_max, tar_para_len_max,
                                              src_para_num_train, tar_para_num_train,
                                              src_trans_train, tar_trans_train,
                                              src_seq_trans, tar_seq_trans,
                                              src_trans_weight_train, tar_trans_weight_train,
                                              src_trans_num_train, tar_trans_num_train,
                                              src_trans_len_max, tar_trans_len_max)
                elif self.para:
                    result = self.train_para_batch(src_pos_train, tar_pos_train,
                                                   src_seq, tar_seq,
                                                   src_neg_train, tar_neg_train,
                                                   src_seq_neg, tar_seq_neg,
                                                   src_len_max, tar_len_max,
                                                   src_para_train, tar_para_train,
                                                   src_seq_para, tar_seq_para,
                                                   src_para_weight_train, tar_para_weight_train,
                                                   src_para_len_max, tar_para_len_max,
                                                   src_para_num_train, tar_para_num_train,
                                                   )
                elif self.trans:
                    result = self.train_trans_batch(src_pos_train, tar_pos_train,
                                                    src_seq, tar_seq,
                                                    src_neg_train, tar_neg_train,
                                                    src_seq_neg, tar_seq_neg,
                                                    src_len_max, tar_len_max,
                                                    src_trans_train, tar_trans_train,
                                                    src_seq_trans, tar_seq_trans,
                                                    src_trans_weight_train, tar_trans_weight_train,
                                                    src_trans_num_train, tar_trans_num_train,
                                                    src_trans_len_max, tar_trans_len_max)

                logger.debug(result)
                if result[0] == np.inf or result[0] == np.nan:
                    logger.info("Detect nan or inf")
                    exit(-1)
                loss.append(result)
                del result
                train_time += (time.time() - time_temp)
            logger.info("epoch %d time: %f, Greedy %f, Train %f" % (i, time.time() - epoch_start, greedy_time, train_time))
            logger.info("epoch %d time: Generate %f" % (i, generate_time))
            logger.info("epoch %d rec_err %f sem_err %f" % (i, np.mean(loss, axis=0)[1], np.mean(loss, axis=0)[2]))
            logger.info("epoch %d l2_err  %f sum_err %f" % (i, np.mean(loss, axis=0)[3], np.mean(loss, axis=0)[0]))
            if self.para and self.trans:
                logger.info("epoch %d isomap_err  %f" % (i, np.mean(loss, axis=0)[4]))
                logger.info("epoch %d le_err  %f" % (i, np.mean(loss, axis=0)[5]))
            elif self.para:
                logger.info("epoch %d le_err  %f" % (i, np.mean(loss, axis=0)[4]))
            elif self.trans:
                logger.info("epoch %d isomap_err  %f" % (i, np.mean(loss, axis=0)[4]))

            history_loss.append(np.mean(loss, axis=0))
            del loss
            self.save_model("%s_iter%d.model" % (model_name, i))
            save_random_state("%s_iter%d.model.rs" % (model_name, i))
            if len(history_loss) > 1 and abs(history_loss[-1][0] - history_loss[-2][0]) < 10e-6:
                logger.info("joint error reaches a local minima")
                break

    def tune_hyper_parameter(self, src_phrases, tar_phrases, train_pair, dev_pair, test_pair, config, model_name,
                             start_iter=1, end_iter=26):
        """
        :param src_phrases:
        :param tar_phrases:
        :param train_pair:
        :param dev_pair:
        :param test_pair:
        :param config:
        :param model_name:
        :param start_iter:
        :param end_iter:
        :return:
        """
        n_epoch = config.n_epoch
        batch_size = config.batch_size
        size_src_word = self.src_embedding.size
        size_tar_word = self.tar_embedding.size

        train_index = align_batch_size(range(len(train_pair)), batch_size)
        src_train = [phrase[WORD_INDEX] for phrase in src_phrases]
        tar_train = [phrase[WORD_INDEX] for phrase in tar_phrases]

        src_phrase_id_set = set([src for src, tar in train_pair]) | set([src for src, tar in dev_pair]) | set([src for src, tar in test_pair])
        tar_phrase_id_set = set([tar for src, tar in train_pair]) | set([tar for src, tar in dev_pair]) | set([tar for src, tar in test_pair])

        # 依长度对短语进行排序 Train
        len_src_tar_pair = [(len(src_train[train_pair[index][0]]), len(tar_train[train_pair[index][1]]), index)
                            for index in train_index]
        len_src_tar_pair.sort()
        len_sort_index = np.array([index[2] for index in len_src_tar_pair], np.int32)
        src_len_index = np.array([index[0] for index in len_src_tar_pair], np.int32)
        tar_len_index = np.array([index[1] for index in len_src_tar_pair], np.int32)
        num_batch = len(len_sort_index) / batch_size

        if self.verbose:
            logger.debug("Train Details")
            logger.debug("Number of epochs: %d" % n_epoch)
            logger.debug("Size of Batches:  %d" % batch_size)
            logger.debug("Size phrase pair: %d" % len(train_pair))
            logger.debug("Size Source word: %d" % size_src_word)
            logger.debug("Size Target word: %d" % size_tar_word)

        logger.info("Start training...")
        history_loss = []
        dev_history_loss = []
        test_history_loss = []
        random_batch_index = np.arange(num_batch)
        for i in xrange(1, n_epoch + 1):

            if i < start_iter:
                continue
            if i == start_iter and start_iter != 1:
                self.load_model("%s_iter%d.tune.model" % (model_name, i - 1))
                # 历史误差信息存放在这里
                load_random_state("%s_iter%d.tune.model.rs" % (model_name, i - 1))
                dev_history_loss, test_history_loss = load_dev_test_loss("%s_iter%d.tune.model.loss"
                                                                         % (model_name, i - 1))
            if i >= end_iter:
                logger.info("Reach End the Iter")
                exit(1)

            rae_util = RAEUtil(self.source_dim, normalize=self.normalize)
            # 每一轮迭代都随机生成负例
            logger.info("Generate Neg Instance ...")
            src_pos, src_neg = BilingualPhraseRAE.generate_train_array_align(src_train, size_src_word,
                                                                             phrases_id=src_phrase_id_set)
            tar_pos, tar_neg = BilingualPhraseRAE.generate_train_array_align(tar_train, size_tar_word,
                                                                             phrases_id=tar_phrase_id_set)
            src_pos_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
            src_neg_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
            tar_pos_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
            tar_neg_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
            src_para_train = np.zeros((batch_size, self.para_num, 2 * align_len - 1), dtype=np.int32)
            tar_para_train = np.zeros((batch_size, self.para_num, 2 * align_len - 1), dtype=np.int32)
            src_trans_train = np.zeros((batch_size, self.trans_num, 2 * align_len - 1), dtype=np.int32)
            tar_trans_train = np.zeros((batch_size, self.trans_num, 2 * align_len - 1), dtype=np.int32)
            src_seq = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)
            tar_seq = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)
            src_seq_neg = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)
            tar_seq_neg = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)
            # FOR LE
            src_seq_para = np.zeros((batch_size, self.para_num, self.length - 1, 3), dtype=np.int32)
            tar_seq_para = np.zeros((batch_size, self.para_num, self.length - 1, 3), dtype=np.int32)
            # FOR ISOMAP
            src_seq_trans = np.zeros((batch_size, self.trans_num, self.length - 1, 3), dtype=np.int32)
            tar_seq_trans = np.zeros((batch_size, self.trans_num, self.length - 1, 3), dtype=np.int32)
            random_batch_index = np.random.permutation(random_batch_index)

            loss = []
            epoch_start = time.time()
            generate_time = 0
            greedy_time = 0
            train_time = 0
            k = 0
            # For Train
            for j in random_batch_index:
                k += 1
                if k % 5000 == 0:
                    while gc.collect() > 0:
                        pass
                # 清空临时路径
                if self.verbose:
                    print progress_bar_str(k, num_batch) + "\r",
                temp_time = time.time()
                indexs = len_sort_index[j * batch_size: (j + 1) * batch_size]
                src_indexs = [train_pair[index][0] for index in indexs]
                tar_indexs = [train_pair[index][1] for index in indexs]
                src_len_max = np.max(src_len_index[j * batch_size: (j + 1) * batch_size])
                tar_len_max = np.max(tar_len_index[j * batch_size: (j + 1) * batch_size])
                # FOR LE
                if self.para:
                    src_para, src_para_weight, src_para_num = BilingualPhraseRAELLE.generate_para_train2(src_phrases,
                                                                                                         src_indexs,
                                                                                                         self.para_num)
                    tar_para, tar_para_weight, tar_para_num = BilingualPhraseRAELLE.generate_para_train2(tar_phrases,
                                                                                                         tar_indexs,
                                                                                                         self.para_num)
                    src_para_len_index = np.array([[len(src_train[index]) for index in indexs]
                                                   for indexs in src_para], np.int32)
                    tar_para_len_index = np.array([[len(tar_train[index]) for index in indexs]
                                                   for indexs in tar_para], np.int32)
                    src_para_len_max = np.max(src_para_len_index, axis=0)
                    tar_para_len_max = np.max(tar_para_len_index, axis=0)

                # FOR ISOMAP
                if self.trans:
                    src_trans, src_trans_weight, src_trans_num = BilingualPhraseRAEISOMAP.generate_trans_train(
                        src_phrases,
                        src_indexs,
                        tar_indexs,
                        self.trans_num)
                    tar_trans, tar_trans_weight, tar_trans_num = BilingualPhraseRAEISOMAP.generate_trans_train(
                        tar_phrases,
                        tar_indexs,
                        src_indexs,
                        self.trans_num)
                    src_seq_trans *= 0
                    tar_seq_trans *= 0
                    src_trans_len_index = np.array([[len(tar_train[index]) for index in tran_indexs]
                                                    for tran_indexs in src_trans], np.int32)
                    tar_trans_len_index = np.array([[len(src_train[index]) for index in tran_indexs]
                                                    for tran_indexs in tar_trans], np.int32)
                    src_trans_len_max = np.max(src_trans_len_index, axis=0)
                    tar_trans_len_max = np.max(tar_trans_len_index, axis=0)

                src_pos_train[:, :align_len], src_neg_train[:, :align_len] = src_pos[src_indexs], src_neg[src_indexs]
                tar_pos_train[:, :align_len], tar_neg_train[:, :align_len] = tar_pos[tar_indexs], tar_neg[tar_indexs]

                if self.para:
                    src_para_train[:, :, :align_len], src_para_weight_train = src_pos[src_para], src_para_weight
                    tar_para_train[:, :, :align_len], tar_para_weight_train = tar_pos[tar_para], tar_para_weight
                    src_para_num_train, tar_para_num_train = src_para_num, tar_para_num

                if self.trans:
                    src_trans_train[:, :, :align_len], tar_trans_train[:, :, :align_len] = tar_pos[src_trans], src_pos[
                        tar_trans]
                    src_trans_weight_train, src_trans_num_train = src_trans_weight, src_trans_num
                    tar_trans_weight_train, tar_trans_num_train = tar_trans_weight, tar_trans_num

                generate_time += (time.time() - temp_time)
                # Generate greedy path
                time_temp = time.time()
                src_w = self.src_encoder.W.get_value(borrow=True)
                src_b = self.src_encoder.b.get_value(borrow=True)
                src_wr = self.src_encoder.Wr.get_value(borrow=True)
                src_br = self.src_encoder.br.get_value(borrow=True)
                tar_w = self.tar_encoder.W.get_value(borrow=True)
                tar_b = self.tar_encoder.b.get_value(borrow=True)
                tar_wr = self.tar_encoder.Wr.get_value(borrow=True)
                tar_br = self.tar_encoder.br.get_value(borrow=True)
                for sub_index in xrange(batch_size):
                    # Pos Path Generate
                    src_word_index = src_pos_train[sub_index]
                    tar_word_index = tar_pos_train[sub_index]
                    src_word_vec, tar_word_vec = self.get_indexs_embedding(src_word_index, tar_word_index)
                    src_seq[sub_index] = rae_util.build_greed_path(src_word_vec, src_w, src_b, src_wr, src_br,
                                                                   src_len_index[j * batch_size + sub_index])
                    tar_seq[sub_index] = rae_util.build_greed_path(tar_word_vec, tar_w, tar_b, tar_wr, tar_br,
                                                                   tar_len_index[j * batch_size + sub_index])
                    # Neg Path Generate
                    src_word_index = src_neg_train[sub_index]
                    tar_word_index = tar_neg_train[sub_index]
                    src_word_vec, tar_word_vec = self.get_indexs_embedding(src_word_index, tar_word_index)
                    src_seq_neg[sub_index] = rae_util.build_greed_path(src_word_vec, src_w, src_b, src_wr, src_br,
                                                                       src_len_index[j * batch_size + sub_index])
                    tar_seq_neg[sub_index] = rae_util.build_greed_path(tar_word_vec, tar_w, tar_b, tar_wr, tar_br,
                                                                       tar_len_index[j * batch_size + sub_index])
                    # Para Path Generate
                    if self.para:
                        for para_index in xrange(src_para_num_train[sub_index]):
                            src_word_index = src_para_train[sub_index][para_index]
                            src_word_vec = self.get_src_indexs_embedding(src_word_index)
                            src_seq_para[sub_index][para_index] = rae_util.build_greed_path(src_word_vec,
                                                                                            src_w, src_b, src_wr,
                                                                                            src_br,
                                                                                            src_para_len_index[
                                                                                                sub_index][para_index])
                        for para_index in xrange(tar_para_num_train[sub_index]):
                            tar_word_index = tar_para_train[sub_index][para_index]
                            tar_word_vec = self.get_tar_indexs_embedding(tar_word_index)
                            tar_seq_para[sub_index][para_index] = rae_util.build_greed_path(tar_word_vec,
                                                                                            tar_w, tar_b, tar_wr,
                                                                                            tar_br,
                                                                                            tar_para_len_index[
                                                                                                sub_index][para_index])

                    if self.trans:
                        for tran_index in xrange(src_trans_num_train[sub_index]):
                            # src trans -> tar
                            tar_word_index = src_trans_train[sub_index][tran_index]
                            # output is list or dict
                            tar_word_vec = self.get_tar_indexs_embedding(tar_word_index)
                            src_seq_trans[sub_index][tran_index] = rae_util.build_greed_path(tar_word_vec,
                                                                                             tar_w, tar_b, tar_wr,
                                                                                             tar_br,
                                                                                             src_trans_len_index[
                                                                                                 sub_index][tran_index])
                        for tran_index in xrange(tar_trans_num_train[sub_index]):
                            # tar trans -> src
                            src_word_index = tar_trans_train[sub_index][tran_index]
                            # output is list or dict
                            src_word_vec = self.get_src_indexs_embedding(src_word_index)
                            tar_seq_trans[sub_index][tran_index] = rae_util.build_greed_path(src_word_vec,
                                                                                             src_w, src_b, src_wr,
                                                                                             src_br,
                                                                                             tar_trans_len_index[
                                                                                                 sub_index][tran_index])

                greedy_time += (time.time() - time_temp)
                # Train
                time_temp = time.time()
                if self.para and self.trans:
                    result = self.train_batch(src_pos_train, tar_pos_train,
                                              src_seq, tar_seq,
                                              src_neg_train, tar_neg_train,
                                              src_seq_neg, tar_seq_neg,
                                              src_len_max, tar_len_max,
                                              src_para_train, tar_para_train,
                                              src_seq_para, tar_seq_para,
                                              src_para_weight_train, tar_para_weight_train,
                                              src_para_len_max, tar_para_len_max,
                                              src_para_num_train, tar_para_num_train,
                                              src_trans_train, tar_trans_train,
                                              src_seq_trans, tar_seq_trans,
                                              src_trans_weight_train, tar_trans_weight_train,
                                              src_trans_num_train, tar_trans_num_train,
                                              src_trans_len_max, tar_trans_len_max)
                elif self.para:
                    result = self.train_para_batch(src_pos_train, tar_pos_train,
                                                   src_seq, tar_seq,
                                                   src_neg_train, tar_neg_train,
                                                   src_seq_neg, tar_seq_neg,
                                                   src_len_max, tar_len_max,
                                                   src_para_train, tar_para_train,
                                                   src_seq_para, tar_seq_para,
                                                   src_para_weight_train, tar_para_weight_train,
                                                   src_para_len_max, tar_para_len_max,
                                                   src_para_num_train, tar_para_num_train,
                                                   )
                elif self.trans:
                    result = self.train_trans_batch(src_pos_train, tar_pos_train,
                                                    src_seq, tar_seq,
                                                    src_neg_train, tar_neg_train,
                                                    src_seq_neg, tar_seq_neg,
                                                    src_len_max, tar_len_max,
                                                    src_trans_train, tar_trans_train,
                                                    src_seq_trans, tar_seq_trans,
                                                    src_trans_weight_train, tar_trans_weight_train,
                                                    src_trans_num_train, tar_trans_num_train,
                                                    src_trans_len_max, tar_trans_len_max)
                logger.debug(result)
                if result[0] == np.inf or result[0] == np.nan:
                    logger.info("Detect nan or inf")
                    exit(-1)
                loss.append(result)
                del result
                train_time += (time.time() - time_temp)
            logger.info("epoch %d time: %f, Greedy %f, Train %f" % (i, time.time() - epoch_start, greedy_time, train_time))
            logger.info("epoch %d time: Generate %f" % (i, generate_time))
            logger.info("epoch %d rec_err %f sem_err %f" % (i, np.mean(loss, axis=0)[1], np.mean(loss, axis=0)[2]))
            logger.info("epoch %d l2_err  %f sum_err %f" % (i, np.mean(loss, axis=0)[3], np.mean(loss, axis=0)[0]))
            if self.para and self.trans:
                logger.info("epoch %d isomap_err  %f" % (i, np.mean(loss, axis=0)[4]))
                logger.info("epoch %d le_err  %f" % (i, np.mean(loss, axis=0)[5]))
            elif self.para:
                logger.info("epoch %d le_err  %f" % (i, np.mean(loss, axis=0)[4]))
            elif self.trans:
                logger.info("epoch %d isomap_err  %f" % (i, np.mean(loss, axis=0)[4]))
            history_loss.append(np.mean(loss, axis=0))
            dev_history_loss.append(self.predict_loss(src_phrases, tar_phrases, src_train, tar_train, dev_pair,
                                                      src_pos, src_neg, tar_pos, tar_neg, config,
                                                      pref="epoch %d Dev" % i))
            test_history_loss.append(self.predict_loss(src_phrases, tar_phrases, src_train, tar_train, test_pair,
                                                       src_pos, src_neg, tar_pos, tar_neg, config,
                                                       pref="epoch %d Test" % i))
            del loss
            self.save_model("%s_iter%d.tune.model" % (model_name, i))
            save_random_state("%s_iter%d.tune.model.rs" % (model_name, i))
            save_dev_test_loss("%s_iter%d.tune.model.loss" % (model_name, i), dev_history_loss, test_history_loss)
            if len(history_loss) > 1 and abs(history_loss[-1][0] - history_loss[-2][0]) < 10e-6:
                logger.info("joint error reaches a local minima")
                dev_final_err = [losses[0] for losses in dev_history_loss]
                test_final_err = [losses[0] for losses in test_history_loss]
                min_iter = np.argmin(dev_final_err)
                logger.info("[MinErr] at Iter %d" % min_iter)
                logger.info("[DevErr] %f" % dev_final_err[min_iter])
                logger.info("[TestErr] %f" % test_final_err[min_iter])
                break
            if i == n_epoch:
                dev_final_err = [losses[0] for losses in dev_history_loss]
                test_final_err = [losses[0] for losses in test_history_loss]
                min_iter = np.argmin(dev_final_err)
                logger.info("[MinErr] at Iter %d" % min_iter)
                logger.info("[DevErr] %f" % dev_final_err[min_iter])
                logger.info("[TestErr] %f" % test_final_err[min_iter])
            
    def predict_loss(self, src_phrases, tar_phrases, src_train, tar_train, test_pair,
                     src_pos, src_neg, tar_pos, tar_neg, config, pref=""):
        batch_size = config.batch_size
        rae_util = RAEUtil(self.source_dim, normalize=self.normalize)
        test_index = align_batch_size(range(len(test_pair)), batch_size)
        test_len_src_tar_pair = [(len(src_train[test_pair[index][0]]), len(tar_train[test_pair[index][1]]), index)
                                 for index in test_index]
        test_len_src_tar_pair.sort()
        test_len_sort_index = np.array([index[2] for index in test_len_src_tar_pair], np.int32)
        test_src_len_index = np.array([index[0] for index in test_len_src_tar_pair], np.int32)
        test_tar_len_index = np.array([index[1] for index in test_len_src_tar_pair], np.int32)
        test_num_batch = len(test_len_sort_index) / batch_size
        src_pos_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
        src_neg_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
        tar_pos_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
        tar_neg_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
        src_para_train = np.zeros((batch_size, self.para_num, 2 * align_len - 1), dtype=np.int32)
        tar_para_train = np.zeros((batch_size, self.para_num, 2 * align_len - 1), dtype=np.int32)
        src_trans_train = np.zeros((batch_size, self.trans_num, 2 * align_len - 1), dtype=np.int32)
        tar_trans_train = np.zeros((batch_size, self.trans_num, 2 * align_len - 1), dtype=np.int32)
        src_seq = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)
        tar_seq = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)
        src_seq_neg = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)
        tar_seq_neg = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)
        # FOR LE
        src_seq_para = np.zeros((batch_size, self.para_num, self.length - 1, 3), dtype=np.int32)
        tar_seq_para = np.zeros((batch_size, self.para_num, self.length - 1, 3), dtype=np.int32)
        # FOR ISOMAP
        src_seq_trans = np.zeros((batch_size, self.trans_num, self.length - 1, 3), dtype=np.int32)
        tar_seq_trans = np.zeros((batch_size, self.trans_num, self.length - 1, 3), dtype=np.int32)
        k = 0
        losses = list()
        for j in xrange(test_num_batch):
            k += 1
            if k % 5000 == 0:
                while gc.collect() > 0:
                    pass
            # 清空临时路径
            if self.verbose:
                print progress_bar_str(k, test_num_batch) + "\r",
            indexs = test_len_sort_index[j * batch_size: (j + 1) * batch_size]
            src_indexs = [test_pair[index][0] for index in indexs]
            tar_indexs = [test_pair[index][1] for index in indexs]
            src_len_max = np.max(test_src_len_index[j * batch_size: (j + 1) * batch_size])
            tar_len_max = np.max(test_tar_len_index[j * batch_size: (j + 1) * batch_size])
            # FOR LE
            if self.para:
                src_para, src_para_weight, src_para_num = BilingualPhraseRAELLE.generate_para_train2(src_phrases,
                                                                                                     src_indexs,
                                                                                                     self.para_num)
                tar_para, tar_para_weight, tar_para_num = BilingualPhraseRAELLE.generate_para_train2(tar_phrases,
                                                                                                     tar_indexs,
                                                                                                     self.para_num)
                src_para_len_index = np.array([[len(src_train[index]) for index in indexs]
                                               for indexs in src_para], np.int32)
                tar_para_len_index = np.array([[len(tar_train[index]) for index in indexs]
                                               for indexs in tar_para], np.int32)
                src_para_len_max = np.max(src_para_len_index, axis=0)
                tar_para_len_max = np.max(tar_para_len_index, axis=0)

            # FOR ISOMAP
            if self.trans:
                src_trans, src_trans_weight, src_trans_num = BilingualPhraseRAEISOMAP.generate_trans_train(src_phrases,
                                                                                                           src_indexs,
                                                                                                           tar_indexs,
                                                                                                           self.trans_num)
                tar_trans, tar_trans_weight, tar_trans_num = BilingualPhraseRAEISOMAP.generate_trans_train(tar_phrases,
                                                                                                           tar_indexs,
                                                                                                           src_indexs,
                                                                                                           self.trans_num)
                src_seq_trans *= 0
                tar_seq_trans *= 0
                src_trans_len_index = np.array([[len(tar_train[index]) for index in tran_indexs]
                                                for tran_indexs in src_trans], np.int32)
                tar_trans_len_index = np.array([[len(src_train[index]) for index in tran_indexs]
                                                for tran_indexs in tar_trans], np.int32)
                src_trans_len_max = np.max(src_trans_len_index, axis=0)
                tar_trans_len_max = np.max(tar_trans_len_index, axis=0)

            src_pos_train[:, :align_len], src_neg_train[:, :align_len] = src_pos[src_indexs], src_neg[src_indexs]
            tar_pos_train[:, :align_len], tar_neg_train[:, :align_len] = tar_pos[tar_indexs], tar_neg[tar_indexs]

            if self.para:
                src_para_train[:, :, :align_len], src_para_weight_train = src_pos[src_para], src_para_weight
                tar_para_train[:, :, :align_len], tar_para_weight_train = tar_pos[tar_para], tar_para_weight
                src_para_num_train, tar_para_num_train = src_para_num, tar_para_num

            if self.trans:
                src_trans_train[:, :, :align_len], tar_trans_train[:, :, :align_len] = tar_pos[src_trans], src_pos[
                    tar_trans]
                src_trans_weight_train, src_trans_num_train = src_trans_weight, src_trans_num
                tar_trans_weight_train, tar_trans_num_train = tar_trans_weight, tar_trans_num

            # Generate greedy path
            src_w = self.src_encoder.W.get_value(borrow=True)
            src_b = self.src_encoder.b.get_value(borrow=True)
            src_wr = self.src_encoder.Wr.get_value(borrow=True)
            src_br = self.src_encoder.br.get_value(borrow=True)
            tar_w = self.tar_encoder.W.get_value(borrow=True)
            tar_b = self.tar_encoder.b.get_value(borrow=True)
            tar_wr = self.tar_encoder.Wr.get_value(borrow=True)
            tar_br = self.tar_encoder.br.get_value(borrow=True)
            for sub_index in xrange(batch_size):
                # Pos Path Generate
                src_word_index = src_pos_train[sub_index]
                tar_word_index = tar_pos_train[sub_index]
                src_word_vec, tar_word_vec = self.get_indexs_embedding(src_word_index, tar_word_index)
                src_seq[sub_index] = rae_util.build_greed_path(src_word_vec, src_w, src_b, src_wr, src_br,
                                                               test_src_len_index[j * batch_size + sub_index])
                tar_seq[sub_index] = rae_util.build_greed_path(tar_word_vec, tar_w, tar_b, tar_wr, tar_br,
                                                               test_tar_len_index[j * batch_size + sub_index])
                # Neg Path Generate
                src_word_index = src_neg_train[sub_index]
                tar_word_index = tar_neg_train[sub_index]
                src_word_vec, tar_word_vec = self.get_indexs_embedding(src_word_index, tar_word_index)
                src_seq_neg[sub_index] = rae_util.build_greed_path(src_word_vec, src_w, src_b, src_wr, src_br,
                                                                   test_src_len_index[j * batch_size + sub_index])
                tar_seq_neg[sub_index] = rae_util.build_greed_path(tar_word_vec, tar_w, tar_b, tar_wr, tar_br,
                                                                   test_tar_len_index[j * batch_size + sub_index])
                # Para Path Generate
                if self.para:
                    for para_index in xrange(src_para_num_train[sub_index]):
                        src_word_index = src_para_train[sub_index][para_index]
                        src_word_vec = self.get_src_indexs_embedding(src_word_index)
                        src_seq_para[sub_index][para_index] = rae_util.build_greed_path(src_word_vec,
                                                                                        src_w, src_b, src_wr, src_br,
                                                                                        src_para_len_index[sub_index][
                                                                                            para_index])
                    for para_index in xrange(tar_para_num_train[sub_index]):
                        tar_word_index = tar_para_train[sub_index][para_index]
                        tar_word_vec = self.get_tar_indexs_embedding(tar_word_index)
                        tar_seq_para[sub_index][para_index] = rae_util.build_greed_path(tar_word_vec,
                                                                                        tar_w, tar_b, tar_wr, tar_br,
                                                                                        tar_para_len_index[sub_index][
                                                                                            para_index])

                if self.trans:
                    for tran_index in xrange(src_trans_num_train[sub_index]):
                        # src trans -> tar
                        tar_word_index = src_trans_train[sub_index][tran_index]
                        # output is list or dict
                        tar_word_vec = self.get_tar_indexs_embedding(tar_word_index)
                        src_seq_trans[sub_index][tran_index] = rae_util.build_greed_path(tar_word_vec,
                                                                                         tar_w, tar_b, tar_wr, tar_br,
                                                                                         src_trans_len_index[sub_index][
                                                                                             tran_index])
                    for tran_index in xrange(tar_trans_num_train[sub_index]):
                        # tar trans -> src
                        src_word_index = tar_trans_train[sub_index][tran_index]
                        # output is list or dict
                        src_word_vec = self.get_src_indexs_embedding(src_word_index)
                        tar_seq_trans[sub_index][tran_index] = rae_util.build_greed_path(src_word_vec,
                                                                                         src_w, src_b, src_wr, src_br,
                                                                                         tar_trans_len_index[sub_index][
                                                                                             tran_index])

            # Predict
            if self.para and self.trans:
                result = self.predict_batch(src_pos_train, tar_pos_train,
                                            src_seq, tar_seq,
                                            src_neg_train, tar_neg_train,
                                            src_seq_neg, tar_seq_neg,
                                            src_len_max, tar_len_max,
                                            src_para_train, tar_para_train,
                                            src_seq_para, tar_seq_para,
                                            src_para_weight_train, tar_para_weight_train,
                                            src_para_len_max, tar_para_len_max,
                                            src_para_num_train, tar_para_num_train,
                                            src_trans_train, tar_trans_train,
                                            src_seq_trans, tar_seq_trans,
                                            src_trans_weight_train, tar_trans_weight_train,
                                            src_trans_num_train, tar_trans_num_train,
                                            src_trans_len_max, tar_trans_len_max)
            elif self.para:
                result = self.predict_para_batch(src_pos_train, tar_pos_train,
                                                 src_seq, tar_seq,
                                                 src_neg_train, tar_neg_train,
                                                 src_seq_neg, tar_seq_neg,
                                                 src_len_max, tar_len_max,
                                                 src_para_train, tar_para_train,
                                                 src_seq_para, tar_seq_para,
                                                 src_para_weight_train, tar_para_weight_train,
                                                 src_para_len_max, tar_para_len_max,
                                                 src_para_num_train, tar_para_num_train,
                                                 )
            elif self.trans:
                result = self.predict_trans_batch(src_pos_train, tar_pos_train,
                                                  src_seq, tar_seq,
                                                  src_neg_train, tar_neg_train,
                                                  src_seq_neg, tar_seq_neg,
                                                  src_len_max, tar_len_max,
                                                  src_trans_train, tar_trans_train,
                                                  src_seq_trans, tar_seq_trans,
                                                  src_trans_weight_train, tar_trans_weight_train,
                                                  src_trans_num_train, tar_trans_num_train,
                                                  src_trans_len_max, tar_trans_len_max)
            losses.append(result)
        logger.info("%s rec_err %f sem_err %f" % (pref, np.mean(losses, axis=0)[1], np.mean(losses, axis=0)[2]))
        logger.info("%s l2_err  %f sum_err %f" % (pref, np.mean(losses, axis=0)[3], np.mean(losses, axis=0)[0]))
        if self.para and self.trans:
            logger.info("isomap_err  %f" % np.mean(losses, axis=0)[4])
            logger.info("le_err  %f" % np.mean(losses, axis=0)[5])
        elif self.para:
            logger.info("le_err  %f" % np.mean(losses, axis=0)[4])
        elif self.trans:
            logger.info("isomap_err  %f" % np.mean(losses, axis=0)[4])
        return np.mean(losses, axis=0)

    def test_kl(self, src_phrases, tar_phrases, test_pair, config):
        batch_size = config.batch_size
        src_train = [phrase[WORD_INDEX] for phrase in src_phrases]
        tar_train = [phrase[WORD_INDEX] for phrase in tar_phrases]
        rae_util = RAEUtil(config.dim, normalize=self.normalize)
        test_index = align_batch_size(range(len(test_pair)), batch_size)
        test_len_src_tar_pair = [(len(src_train[test_pair[index][0]]), len(tar_train[test_pair[index][1]]), index)
                                 for index in test_index]
        test_len_src_tar_pair.sort()
        test_len_sort_index = np.array([index[2] for index in test_len_src_tar_pair], np.int32)
        test_src_len_index = np.array([index[0] for index in test_len_src_tar_pair], np.int32)
        test_tar_len_index = np.array([index[1] for index in test_len_src_tar_pair], np.int32)
        test_num_batch = len(test_len_sort_index) / batch_size
        src_pos_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
        src_neg_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
        tar_pos_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
        tar_neg_train = np.zeros((batch_size, 2 * align_len - 1), dtype=np.int32)
        src_para_train = np.zeros((batch_size, self.para_num, 2 * align_len - 1), dtype=np.int32)
        tar_para_train = np.zeros((batch_size, self.para_num, 2 * align_len - 1), dtype=np.int32)
        src_trans_train = np.zeros((batch_size, self.trans_num, 2 * align_len - 1), dtype=np.int32)
        tar_trans_train = np.zeros((batch_size, self.trans_num, 2 * align_len - 1), dtype=np.int32)
        src_seq = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)
        tar_seq = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)
        src_seq_neg = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)
        tar_seq_neg = np.zeros((batch_size, self.length - 1, 3), dtype=np.int32)
        # FOR LE
        src_seq_para = np.zeros((batch_size, self.para_num, self.length - 1, 3), dtype=np.int32)
        tar_seq_para = np.zeros((batch_size, self.para_num, self.length - 1, 3), dtype=np.int32)
        # FOR ISOMAP
        src_seq_trans = np.zeros((batch_size, self.trans_num, self.length - 1, 3), dtype=np.int32)
        tar_seq_trans = np.zeros((batch_size, self.trans_num, self.length - 1, 3), dtype=np.int32)

        size_src_word = self.src_embedding.size
        size_tar_word = self.tar_embedding.size
        src_phrase_id_set = set([src for src, tar in test_pair])
        tar_phrase_id_set = set([tar for src, tar in test_pair])
        src_pos, src_neg = BilingualPhraseRAE.generate_train_array_align(src_train, size_src_word,
                                                                         phrases_id=src_phrase_id_set)
        tar_pos, tar_neg = BilingualPhraseRAE.generate_train_array_align(tar_train, size_tar_word,
                                                                         phrases_id=tar_phrase_id_set)
        k = 0
        losses = list()
        for j in xrange(test_num_batch):
            k += 1
            if k % 5000 == 0:
                while gc.collect() > 0:
                    pass
            # 清空临时路径
            if self.verbose:
                print progress_bar_str(k, test_num_batch) + "\r",
            indexs = test_len_sort_index[j * batch_size: (j + 1) * batch_size]
            src_indexs = [test_pair[index][0] for index in indexs]
            tar_indexs = [test_pair[index][1] for index in indexs]
            src_len_max = np.max(test_src_len_index[j * batch_size: (j + 1) * batch_size])
            tar_len_max = np.max(test_tar_len_index[j * batch_size: (j + 1) * batch_size])
            # FOR LE
            if self.para:
                src_para, src_para_weight, src_para_num = BilingualPhraseRAELLE.generate_para_train2(src_phrases,
                                                                                                     src_indexs,
                                                                                                     self.para_num)
                tar_para, tar_para_weight, tar_para_num = BilingualPhraseRAELLE.generate_para_train2(tar_phrases,
                                                                                                     tar_indexs,
                                                                                                     self.para_num)
                src_para_len_index = np.array([[len(src_train[index]) for index in indexs]
                                               for indexs in src_para], np.int32)
                tar_para_len_index = np.array([[len(tar_train[index]) for index in indexs]
                                               for indexs in tar_para], np.int32)
                src_para_len_max = np.max(src_para_len_index, axis=0)
                tar_para_len_max = np.max(tar_para_len_index, axis=0)

            # FOR ISOMAP
            if self.trans:
                src_trans, src_trans_weight, src_trans_num = BilingualPhraseRAEISOMAP.generate_trans_train(src_phrases,
                                                                                                           src_indexs,
                                                                                                           tar_indexs,
                                                                                                           self.trans_num)
                tar_trans, tar_trans_weight, tar_trans_num = BilingualPhraseRAEISOMAP.generate_trans_train(tar_phrases,
                                                                                                           tar_indexs,
                                                                                                           src_indexs,
                                                                                                           self.trans_num)
                src_seq_trans *= 0
                tar_seq_trans *= 0
                src_trans_len_index = np.array([[len(tar_train[index]) for index in tran_indexs]
                                                for tran_indexs in src_trans], np.int32)
                tar_trans_len_index = np.array([[len(src_train[index]) for index in tran_indexs]
                                                for tran_indexs in tar_trans], np.int32)
                src_trans_len_max = np.max(src_trans_len_index, axis=0)
                tar_trans_len_max = np.max(tar_trans_len_index, axis=0)

            src_pos_train[:, :align_len], src_neg_train[:, :align_len] = src_pos[src_indexs], src_neg[src_indexs]
            tar_pos_train[:, :align_len], tar_neg_train[:, :align_len] = tar_pos[tar_indexs], tar_neg[tar_indexs]

            if self.para:
                src_para_train[:, :, :align_len], src_para_weight_train = src_pos[src_para], src_para_weight
                tar_para_train[:, :, :align_len], tar_para_weight_train = tar_pos[tar_para], tar_para_weight
                src_para_num_train, tar_para_num_train = src_para_num, tar_para_num

            if self.trans:
                src_trans_train[:, :, :align_len], tar_trans_train[:, :, :align_len] = tar_pos[src_trans], src_pos[
                    tar_trans]
                src_trans_weight_train, src_trans_num_train = src_trans_weight, src_trans_num
                tar_trans_weight_train, tar_trans_num_train = tar_trans_weight, tar_trans_num

            # Generate greedy path
            src_w = self.src_encoder.W.get_value(borrow=True)
            src_b = self.src_encoder.b.get_value(borrow=True)
            src_wr = self.src_encoder.Wr.get_value(borrow=True)
            src_br = self.src_encoder.br.get_value(borrow=True)
            tar_w = self.tar_encoder.W.get_value(borrow=True)
            tar_b = self.tar_encoder.b.get_value(borrow=True)
            tar_wr = self.tar_encoder.Wr.get_value(borrow=True)
            tar_br = self.tar_encoder.br.get_value(borrow=True)
            for sub_index in xrange(batch_size):
                # Pos Path Generate
                src_word_index = src_pos_train[sub_index]
                tar_word_index = tar_pos_train[sub_index]
                src_word_vec, tar_word_vec = self.get_indexs_embedding(src_word_index, tar_word_index)
                src_seq[sub_index] = rae_util.build_greed_path(src_word_vec, src_w, src_b, src_wr, src_br,
                                                               test_src_len_index[j * batch_size + sub_index])
                tar_seq[sub_index] = rae_util.build_greed_path(tar_word_vec, tar_w, tar_b, tar_wr, tar_br,
                                                               test_tar_len_index[j * batch_size + sub_index])
                # Neg Path Generate
                src_word_index = src_neg_train[sub_index]
                tar_word_index = tar_neg_train[sub_index]
                src_word_vec, tar_word_vec = self.get_indexs_embedding(src_word_index, tar_word_index)
                src_seq_neg[sub_index] = rae_util.build_greed_path(src_word_vec, src_w, src_b, src_wr, src_br,
                                                                   test_src_len_index[j * batch_size + sub_index])
                tar_seq_neg[sub_index] = rae_util.build_greed_path(tar_word_vec, tar_w, tar_b, tar_wr, tar_br,
                                                                   test_tar_len_index[j * batch_size + sub_index])
                # Para Path Generate
                if self.para:
                    for para_index in xrange(src_para_num_train[sub_index]):
                        src_word_index = src_para_train[sub_index][para_index]
                        src_word_vec = self.get_src_indexs_embedding(src_word_index)
                        src_seq_para[sub_index][para_index] = rae_util.build_greed_path(src_word_vec,
                                                                                        src_w, src_b, src_wr, src_br,
                                                                                        src_para_len_index[sub_index][
                                                                                            para_index])
                    for para_index in xrange(tar_para_num_train[sub_index]):
                        tar_word_index = tar_para_train[sub_index][para_index]
                        tar_word_vec = self.get_tar_indexs_embedding(tar_word_index)
                        tar_seq_para[sub_index][para_index] = rae_util.build_greed_path(tar_word_vec,
                                                                                        tar_w, tar_b, tar_wr, tar_br,
                                                                                        tar_para_len_index[sub_index][
                                                                                            para_index])

                if self.trans:
                    for tran_index in xrange(src_trans_num_train[sub_index]):
                        # src trans -> tar
                        tar_word_index = src_trans_train[sub_index][tran_index]
                        # output is list or dict
                        tar_word_vec = self.get_tar_indexs_embedding(tar_word_index)
                        src_seq_trans[sub_index][tran_index] = rae_util.build_greed_path(tar_word_vec,
                                                                                         tar_w, tar_b, tar_wr, tar_br,
                                                                                         src_trans_len_index[sub_index][
                                                                                             tran_index])
                    for tran_index in xrange(tar_trans_num_train[sub_index]):
                        # tar trans -> src
                        src_word_index = tar_trans_train[sub_index][tran_index]
                        # output is list or dict
                        src_word_vec = self.get_src_indexs_embedding(src_word_index)
                        tar_seq_trans[sub_index][tran_index] = rae_util.build_greed_path(src_word_vec,
                                                                                         src_w, src_b, src_wr, src_br,
                                                                                         tar_trans_len_index[sub_index][
                                                                                             tran_index])

            # Predict
            if self.para and self.trans:
                result = self.predict_batch(src_pos_train, tar_pos_train,
                                            src_seq, tar_seq,
                                            src_neg_train, tar_neg_train,
                                            src_seq_neg, tar_seq_neg,
                                            src_len_max, tar_len_max,
                                            src_para_train, tar_para_train,
                                            src_seq_para, tar_seq_para,
                                            src_para_weight_train, tar_para_weight_train,
                                            src_para_len_max, tar_para_len_max,
                                            src_para_num_train, tar_para_num_train,
                                            src_trans_train, tar_trans_train,
                                            src_seq_trans, tar_seq_trans,
                                            src_trans_weight_train, tar_trans_weight_train,
                                            src_trans_num_train, tar_trans_num_train,
                                            src_trans_len_max, tar_trans_len_max)
            elif self.para:
                result = self.predict_para_batch(src_pos_train, tar_pos_train,
                                                 src_seq, tar_seq,
                                                 src_neg_train, tar_neg_train,
                                                 src_seq_neg, tar_seq_neg,
                                                 src_len_max, tar_len_max,
                                                 src_para_train, tar_para_train,
                                                 src_seq_para, tar_seq_para,
                                                 src_para_weight_train, tar_para_weight_train,
                                                 src_para_len_max, tar_para_len_max,
                                                 src_para_num_train, tar_para_num_train,
                                                 )
            elif self.trans:
                result = self.predict_trans_batch(src_pos_train, tar_pos_train,
                                                  src_seq, tar_seq,
                                                  src_neg_train, tar_neg_train,
                                                  src_seq_neg, tar_seq_neg,
                                                  src_len_max, tar_len_max,
                                                  src_trans_train, tar_trans_train,
                                                  src_seq_trans, tar_seq_trans,
                                                  src_trans_weight_train, tar_trans_weight_train,
                                                  src_trans_num_train, tar_trans_num_train,
                                                  src_trans_len_max, tar_trans_len_max)
            losses.append(result)
        logger.info("batch%d rec_err %f sem_err %f" % (j, np.mean(losses, axis=0)[1], np.mean(losses, axis=0)[2]))
        logger.info("batch%d l2_err  %f sum_err %f" % (j, np.mean(losses, axis=0)[3], np.mean(losses, axis=0)[0]))
        if self.para and self.trans:
            logger.info("isomap_err  %f" % np.mean(losses, axis=0)[4])
            logger.info("le_err  %f" % np.mean(losses, axis=0)[5])
        elif self.para:
            logger.info("le_err  %f" % np.mean(losses, axis=0)[4])
        elif self.trans:
            logger.info("isomap_err  %f" % np.mean(losses, axis=0)[4])


class RAEUtil(object):
    def __init__(self, dim, normalize):
        self.dim = dim
        self.normalize = normalize

    @staticmethod
    def get_norm(x):
        if x.ndim == 2:
            return np.sqrt(np.sum(np.square(x), axis=1))
        elif x.ndim == 1:
            return np.sqrt(np.sum(np.square(x)))

    def compose(self, left, right, w, b, wr, br, ):
        v = np.concatenate([left, right], axis=1)  # [(word, dim) (word, dim)] -> (word, 2 * dim)
        z = np.tanh(b + np.dot(v, w.T))  # (word, 2 * dim) dot (dim, 2 * dim)T -> (word, dim)
        if self.normalize:
            z = z / self.get_norm(z)[:, None]  # (word, dim) -> (word, dim) normalize by row
        r = np.tanh(br + np.dot(z, wr.T))  # (word, dim) dot (2 * dim, dim)T -> (word, 2 * dim)
        left_r, right_r = r[:, :self.dim], r[:, self.dim:]  # (word, 2 * dim) -> [(word, dim) (word. dim)]
        if self.normalize:
            # (word, dim) -> (word, dim) normalize by row
            left_r = left_r / self.get_norm(left_r)[:, None]
            # (word, dim) -> (word, dim) normalize by row
            right_r = right_r / self.get_norm(right_r)[:, None]
        # (word, )
        loss_rec = np.sum((left_r - left) ** 2, axis=1) + np.sum((right_r - right) ** 2, axis=1)
        min_index = np.argmin(loss_rec)
        return z[min_index], min_index

    def build_greed_rep(self, vectors, w, b, wr, br):
        vec_len = vectors.shape[0]
        if vec_len == 1:
            return vectors[0]
        while vec_len > 2:
            min_vec, min_index = self.compose(vectors[:vec_len - 1], vectors[1:vec_len], w, b, wr, br)
            vectors[min_index:-1] = vectors[min_index + 1:]
            vectors[min_index] = min_vec
            vec_len -= 1
        return np.tanh(b + np.dot(np.concatenate([vectors[0], vectors[1]]), w.T))

    def build_greed_path(self, vectors, w, b, wr, br, phrase_len):
        path = np.zeros((align_len - 1, 3), dtype=np.int32)
        path[:] = align_len - 1 + phrase_len - 1
        if phrase_len == 1:
            pass
        elif phrase_len == 2:
            path[0][0] = 0
            path[0][1] = 1
        else:
            times = phrase_len - 2
            remain = range(phrase_len)
            for i in xrange(times):
                min_vec, min_index = self.compose(vectors[:phrase_len - 1], vectors[1:phrase_len], w, b, wr, br)
                path[i][0] = remain[min_index]
                path[i][1] = remain[min_index + 1]
                path[i][2] = align_len + i
                remain.pop(min_index)
                remain[min_index] = align_len + i
                vectors[min_index:-1] = vectors[min_index + 1:]
                vectors[min_index] = min_vec
                phrase_len -= 1
            path[times][0] = remain[0]
            path[times][1] = remain[1]
        return path


def brae_predict(phrase_file, output_file, model_file=None, model=None, normalize=True,
                 verbose=True, num_process=1, bilinear=False):
    """
    不使用Theano预测 仅利用NumPy计算 同时检验BRAE定义过程
    可与predict函数进行对比 predict
    :param phrase_file:
    :param output_file:
    :param model_file:
    :param model:11
    :param normalize:
    :param verbose:
    :param num_process:
    :param bilinear:
    :return:
    """
    # 获取模型参数
    # Default Component: src_embedding, tar_embedding, src_encoder, tar_encoder, s2t, t2s
    import multiprocessing as mp
    import ctypes
    bilinear_w = None
    s2t_b = None
    s2t_w = None
    t2s_b = None
    t2s_w = None
    print "Loading model ..."
    if model is not None:
        src_word_idx = model.src_embedding.word_idx
        tar_word_idx = model.tar_embedding.word_idx
        src_embedding = model.src_embedding.W.get_value()
        tar_embedding = model.tar_embedding.W.get_value()
        src_w = model.src_encoder.W.get_value()
        src_b = model.src_encoder.b.get_value()
        src_wr = model.src_encoder.Wr.get_value()
        src_br = model.src_encoder.br.get_value()
        tar_w = model.tar_encoder.W.get_value()
        tar_b = model.tar_encoder.b.get_value()
        tar_wr = model.tar_encoder.Wr.get_value()
        tar_br = model.tar_encoder.br.get_value()
        if bilinear:
            bilinear_w = model.bilinear_w.W.get_value()
        else:
            s2t_w = model.s2t.Wl.get_value()
            s2t_b = model.s2t.bl.get_value()
            t2s_w = model.t2s.Wl.get_value()
            t2s_b = model.t2s.bl.get_value()
    elif model_file is not None:
        import cPickle
        with file(model_file, 'rb') as fin:
            src_word_idx = cPickle.load(fin)
            tar_word_idx = cPickle.load(fin)
            src_embedding = cPickle.load(fin)
            tar_embedding = cPickle.load(fin)
            src_w = cPickle.load(fin)
            src_b = cPickle.load(fin)
            src_wr = cPickle.load(fin)
            src_br = cPickle.load(fin)
            tar_w = cPickle.load(fin)
            tar_b = cPickle.load(fin)
            tar_wr = cPickle.load(fin)
            tar_br = cPickle.load(fin)
            if bilinear:
                bilinear_w = cPickle.load(fin)
            else:
                s2t_w = cPickle.load(fin)
                s2t_b = cPickle.load(fin)
                t2s_w = cPickle.load(fin)
                t2s_b = cPickle.load(fin)
    else:
        raise RuntimeError("No Model file or Model to Predict!\n")

    dim = src_embedding.shape[1]
    rae_util = RAEUtil(dim=dim, normalize=normalize)

    global src_phrase, tar_phrase, bi_phrase_pair

    def calc_phrase_rep(start, end, embedding, w, b, wr, br, index_count, len_all):
        # 本函数仅子进程调用
        # 在子进程中删除暂时无用的全局变量节约内存
        global src_phrase, tar_phrase, bi_phrase_pair
        sub_phrases = copy.deepcopy(tar_phrase[start:end])
        del src_phrase, tar_phrase, bi_phrase_pair
        reps = np.frombuffer(mp_arr.get_obj()).reshape((len_all, embedding.shape[1]))
        for sub_phrase, j in zip(sub_phrases, xrange(start, end)):
            if verbose:
                counter_lock.acquire()
                index_count.value += 1
                if index_count.value % 1000 == 0:
                    print progress_bar_str(index_count.value + 1, len_all) + "\r",
                counter_lock.release()
            reps[j] = rae_util.build_greed_rep(embedding[sub_phrase[WORD_INDEX]], w, b, wr, br)
            reps[j] = reps[j] / rae_util.get_norm(reps[j])

    print "Loading phrase data ..."
    # 获取源短语集合 目标短语集合 双语短语对表
    src_phrase, tar_phrase, bi_phrase_pair = read_phrase_list(phrase_file, src_word_idx, tar_word_idx,
                                                              src_max_len=100, tar_max_len=100)

    # 获取目标短语向量表示 及 映射表示
    print "Calc target phrase embedding ..."
    if num_process > 1:
        len_tar = len(tar_phrase)
        mp_arr = mp.Array(ctypes.c_double, len_tar * tar_embedding.shape[1])
        counter_lock = mp.Lock()
        counter = mp.Value(ctypes.c_long, 0)
        p_list = list()
        freeze_support()
        for i in xrange(num_process):
            thread_size = len_tar / num_process
            if i != num_process - 1:
                end_index = (i + 1) * thread_size
            else:
                end_index = len_tar
            p = mp.Process(target=calc_phrase_rep, args=(i * thread_size, end_index, tar_embedding, tar_w, tar_b, tar_wr,
                                                         tar_br, counter, len_tar,))
            p.start()
            p_list.append(p)
        for p in p_list:
            p.join()
        tar_rep = np.frombuffer(mp_arr.get_obj()).reshape((len_tar, tar_embedding.shape[1]))
        print
    else:
        tar_rep = np.zeros((len(tar_phrase), tar_embedding.shape[1]))
        len_tar = len(tar_phrase)
        for phrase, i in zip(tar_phrase, xrange(len_tar)):
            if verbose:
                if i % 1000 == 0:
                    print progress_bar_str(i + 1000, len_tar) + "\r",
            tar_rep[i] = rae_util.build_greed_rep(tar_embedding[phrase[WORD_INDEX]], tar_w, tar_b, tar_wr, tar_br)
            tar_rep[i] = tar_rep[i] / rae_util.get_norm(tar_rep[i])
        print

    print "Calc source phrase embedding ..."
    # 获取源短语向量表示 及 映射表示
    src_rep = np.zeros((len(src_phrase), src_embedding.shape[1]))
    len_src = len(src_phrase)
    for phrase, i in zip(src_phrase, xrange(len_src)):
        if verbose:
            if i % 1000 == 0:
                print progress_bar_str(i + 1000, len_src) + "\r",
        src_rep[i] = rae_util.build_greed_rep(src_embedding[phrase[WORD_INDEX]], src_w, src_b, src_wr, src_br)
        src_rep[i] = src_rep[i] / rae_util.get_norm(src_rep[i])
    print

    if bilinear:
        src_bilinear_rep = np.dot(src_rep, bilinear_w)
        src_tar_score = list()
        len_bi = len(bi_phrase_pair)
        i = 0
        for s_ind, t_ind in bi_phrase_pair:
            if verbose:
                if i % 1000 == 0:
                    print progress_bar_str(i + 1000, len_bi) + "\r",
            i += 1
            bi_score = np.dot(src_bilinear_rep[s_ind], tar_rep[t_ind])
            """if bi_score > 0:
                src_tar_score.append(np.log(bi_score))
            elif bi_score < 0:
                src_tar_score.append(-np.log(-bi_score))
            else:
                src_tar_score.append(0)"""
            src_tar_score.append(bi_score)
        # 输出预测结果
        print "Write to %s ..." % output_file
        i = 0
        with open(output_file, "w") as fout:
            fout.write("%d %d\n" % (len(bi_phrase_pair), dim))
            for src_tar, bi_score in zip(bi_phrase_pair, src_tar_score):
                if verbose:
                    if i % 1000 == 0:
                        print progress_bar_str(i + 1, len_bi) + "\r",
                i += 1
                src_index, tar_index = src_tar
                output_text = "%s ||| %s" % (src_phrase[src_index][TEXT_INDEX].encode("utf-8"),
                                             tar_phrase[tar_index][TEXT_INDEX].encode("utf-8"))
                cos_text = "%.6f" % bi_score
                fout.write("%s ||| %s\n" % (output_text, cos_text))
        print
    else:
        src2tar_rep = np.tanh(np.dot(src_rep, s2t_w.T) + s2t_b)
        tar2src_rep = np.tanh(np.dot(tar_rep, t2s_w.T) + t2s_b)
        # 依据短语表计算cos距离
        print "Calc cosine distance ..."
        src_tar_cos = list()
        norm_src_rep = rae_util.get_norm(src_rep)
        norm_tar_rep = rae_util.get_norm(tar_rep)
        norm_src2tar_rep = rae_util.get_norm(src2tar_rep)
        norm_tar2src_rep = rae_util.get_norm(tar2src_rep)
        len_bi = len(bi_phrase_pair)
        i = 0
        for s_ind, t_ind in bi_phrase_pair:
            if verbose:
                if i % 1000 == 0:
                    print progress_bar_str(i + 1000, len_bi) + "\r",
            i += 1
            src_cos = np.sum(src_rep[s_ind] * tar2src_rep[t_ind]) / (norm_src_rep[s_ind] * norm_tar2src_rep[t_ind])
            tar_cos = np.sum(tar_rep[t_ind] * src2tar_rep[s_ind]) / (norm_tar_rep[t_ind] * norm_src2tar_rep[s_ind])
            src_tar_cos.append((src_cos, tar_cos))
        print

        # 输出预测结果
        print "Write to %s ..." % output_file
        i = 0
        with open(output_file, "w") as fout:
            fout.write("%d %d\n" % (len(bi_phrase_pair), dim))
            for src_tar, bi_cos in zip(bi_phrase_pair, src_tar_cos):
                if verbose:
                    if i % 1000 == 0:
                        print progress_bar_str(i + 1, len_bi) + "\r",
                i += 1
                src_index, tar_index = src_tar
                src_cos, tar_cos = bi_cos
                output_text = "%s ||| %s" % (src_phrase[src_index][TEXT_INDEX].encode("utf-8"),
                                             tar_phrase[tar_index][TEXT_INDEX].encode("utf-8"))
                cos_text = "%.6f ||| %.6f" % (src_cos, tar_cos)
                fout.write("%s ||| %s\n" % (output_text, cos_text))
        print
    fout.close()
