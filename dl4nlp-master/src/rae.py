import copy
import logging
from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse

from src import default_initializer
from src.activations import Activation
from src.similarity import cosine_similarity
from src.utils import shared_rand_matrix, progress_bar_str, get_train_sequence

__author__ = 'roger'

align_len = 7
logger = logging.getLogger(__name__)
np.set_printoptions(threshold=np.nan)


class RecursiveAutoEncoder(object):
    def __init__(self, vectors, dim, normalize, dropout, activation, initializer=default_initializer, verbose=True):
        """
        :param vectors: a theano tensor variable
        :param dim:
        :param uniform_range:
        :param normalize:
        :param dropout:
        :param verbose:
        :return:
        """
        self.vectors = vectors
        self.dim = dim
        self.normalize = normalize
        self.dropout = dropout
        self.verbose = verbose
        self.act = Activation(activation)
        # Composition Function Weight
        self.W = shared_rand_matrix((self.dim, 2 * self.dim), 'W', initializer=initializer)
        self.b = shared_rand_matrix((self.dim, ), 'b', 0)
        # Reconstruction Function Weight
        self.Wr = shared_rand_matrix((2 * self.dim, self.dim), 'Wr', initializer=initializer)
        self.br = shared_rand_matrix((self.dim * 2, ), 'br', 0)
        self.params = [self.W, self.b, self.Wr, self.br]

        self.seq = T.lmatrix()
        self.left_vector = T.vector()
        self.right_vector = T.vector()

        self.zero = theano.shared(np.array([0.0], dtype=theano.config.floatX))
        self.scan_result, _ = theano.scan(self.encode, sequences=[self.seq],
                                          outputs_info=[self.vectors, None], name="pos_rae_build")
        self.loss_rec = T.sum(self.scan_result[1])
        self.loss_l1 = sum([T.sum(T.abs_(param)) for param in self.params])
        self.loss_l2 = sum([T.sum(param ** 2) for param in self.params])
        # all history vector in scan
        self.history_output = self.scan_result[0]
        self.all_output = self.history_output[-1]
        # final output
        # self.output = self.all_output[-1]
        # Just for two compose
        self.two_compose_result, self.two_compose_rec = self.compose(self.left_vector, self.right_vector)
        self.compose_two = theano.function(
            inputs=[self.left_vector, self.right_vector],
            outputs=[self.two_compose_result, self.two_compose_rec]
        )
        # Compose Vectors: N vectors -> N-1 Vectors
        compose_vector = T.fmatrix()
        compose_len = T.iscalar()
        path = T.imatrix()
        hs, _ = theano.scan(fn=self.compose_step,
                            sequences=T.arange(compose_len - 1),
                            non_sequences=compose_vector,
                            name="compose_phrase")
        comp_vec = hs[0]
        comp_rec = hs[1]
        min_index = T.argmin(comp_rec)
        """compose_result, _ = theano.scan(fn=self.greed_step,
                                        sequences=[T.arange(compose_len - 1), T.arange(compose_len - 1)],
                                        non_sequences=[compose_vector, path])"""
        self._compose_vectors = theano.function([compose_vector, compose_len], [comp_vec[min_index], min_index])
        """self._compose_result = theano.function([compose_vector, compose_len, path], [path])"""

        if verbose:
            logger.debug('Architecture of RAE built finished, summarized as below: ')
            logger.debug('Hidden dimension: %d' % self.dim)
            logger.debug('Normalize:        %s' % self.normalize)
            logger.debug('Dropout Rate:     %s' % self.dropout)

    def compose(self, left_v, right_v):
        v = T.concatenate([left_v, right_v])
        z = self.act.activate(self.b + T.dot(self.W, v))
        if self.normalize:
            z = z / T.sqrt(T.sum(z ** 2))
        r = self.act.activate(self.br + T.dot(self.Wr, z))
        w_left_r, w_right_r = r[:self.dim], r[self.dim:]
        if self.normalize:
            w_left_r = w_left_r / T.sqrt(T.sum(w_left_r ** 2))
            w_right_r = w_right_r / T.sqrt(T.sum(w_right_r ** 2))
        loss_rec = T.sum((w_left_r - left_v) ** 2) + T.sum((w_right_r - right_v) ** 2)
        return z, loss_rec

    def encode(self, t, vecs):
        # vecs[t[0]] and vecs[t[0]] ==> vecs[t[2]]
        w_left, w_right = vecs[t[0]], vecs[t[1]]
        z, loss_rec = self.compose(w_left, w_right)
        return T.set_subtensor(vecs[t[2]], z), loss_rec

    def compose_step(self, iter_num, current_level):
        l_vec = current_level[iter_num]
        r_vec = current_level[iter_num + 1]
        return self.compose(l_vec, r_vec)

    def greed_step(self, vec_len, node_index, seq_index, vectors, path):
        hs, _ = theano.scan(fn=self.compose_step,
                            sequences=T.arange(vec_len - 1),
                            non_sequences=vectors,
                            name="compose_phrase")
        comp_vec = hs[0]
        comp_rec = hs[1]
        min_index = T.argmin(comp_rec)
        T.set_subtensor(vectors[min_index:-1], vectors[min_index + 1:])
        T.set_subtensor(vectors[min_index], comp_vec[min_index])
        T.set_subtensor(path[seq_index], T.concatenate([min_index, min_index + 1, node_index]))
        return vectors, min_index

    def scan_compose_vectors(self, vectors, vec_len):
        return self._compose_vectors(vectors, vec_len)


class PhraseRAE(RecursiveAutoEncoder):
    def __init__(self, dim, initializer=default_initializer, normalize=True, dropout=0, activation="tanh", verbose=True):
        self.pos_vectors = T.fmatrix()
        super(PhraseRAE, self).__init__(vectors=self.pos_vectors, dim=dim,
                                        initializer=initializer, normalize=normalize,
                                        dropout=dropout, verbose=verbose, activation=activation)
        # Consider Phrase Only One
        self.output, self.loss_rec = ifelse(T.eq(self.pos_vectors.shape[0], 1),
                                            (self.pos_vectors[0], 0.0 * self.loss_l2),  # True
                                            (self.all_output[-1], T.sum(self.scan_result[1])))  # False
        self.params = self.params

        self._compose_phrase = theano.function(
            inputs=[self.pos_vectors, self.seq],
            outputs=[self.output]
        )

    def update_param(self, grads, learn_rate=1.0):
        """
        Update param in Phrase RAE (Embedding and RAE)
        :param grads: [np.ndarray]. List of numpy.ndarray to update the model parameters.
        :param learn_rate: scalar. Learning rate.
        :return:
        """
        for param, grad in zip(self.params, grads):
            p = param.get_value()
            grad = np.asarray(grad)
            param.set_value(p - learn_rate * grad)

    def generate_greedy_path(self, vector):
        """
        Generate Path based on Greedy Strategy
        :param vector: np.ndarray   List of numpy.ndarray vector seq at each time
        :return:
        """
        times = (len(vector) - 1) / 2
        seq = list()
        remain = range(times + 1)
        remain_time = len(remain)
        for i in range(times):
            add_node = times + 1 + i
            compose_result, index = self.scan_compose_vectors(vector, remain_time)
            seq.append([remain[index], remain[index + 1], add_node])
            remain.pop(index)
            remain[index] = add_node
            vector[index:-1] = vector[index + 1:]
            vector[index] = compose_result
            remain_time -= 1
        return seq

    def generate_node_path(self, vector):
        if vector.shape[0] == 1:
            return vector, [[0, 0, 0]]
        # add_zeros = np.zeros((vector.shape[0] - 1, vector.shape[1]), dtype=theano.config.floatX)
        # vector = np.concatenate([vector, add_zeros])
        if vector.shape[0] - 1 == 2:
            return vector, [[0, 1, 2]]
        nodes = np.copy(vector)
        seq = self.generate_greedy_path(vector)
        return nodes, seq

    def compose_phrase(self, phrase):
        nodes, seq = self.generate_node_path(phrase)
        return self._compose_phrase(nodes, seq)


class NegativePhraseRAE(PhraseRAE):
    def __init__(self, dim, initializer=default_initializer, normalize=True, dropout=0, activation="tanh", verbose=True):
        super(NegativePhraseRAE, self).__init__(dim, initializer=initializer,
                                                normalize=normalize, dropout=dropout,
                                                activation=activation, verbose=verbose)
        self.neg_seq = T.lmatrix()
        self.neg_vectors = T.fmatrix()
        self.neg_scan_result, _ = theano.scan(self.encode, sequences=[self.neg_seq],
                                              outputs_info=[self.neg_vectors, None],
                                              name="neg_rae_build")
        # all Negative history vector in scan
        self.neg_history_output = self.neg_scan_result[0]
        self.neg_all_output = self.neg_history_output[-1]

        # Consider Negative Phrase Only One
        self.neg_output = ifelse(T.eq(self.neg_vectors.shape[0], 1),
                                 self.neg_vectors[0],  # True
                                 self.neg_all_output[-1])  # False
        self.neg_loss_rec = ifelse(T.eq(self.neg_vectors.shape[0], 1),
                                   0.0,  # True
                                   T.sum(self.neg_scan_result[1]))  # False


class BilingualPhraseRAE(object):
    def __init__(self, source_dim, target_dim, initializer=default_initializer, config=None, verbose=True):
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.alpha = config.alpha
        self.uniform_range = config.uniform_range
        self.normalize = config.normalize
        self.weight_rec = config.weight_rec
        self.weight_sem = config.weight_sem
        self.weight_l2 = config.weight_l2
        self.dropout = config.dropout
        self.verbose = verbose
        self.learning_rate = config.optimizer.param["lr"]
        self.source_encoder = NegativePhraseRAE(self.source_dim, initializer=initializer, normalize=self.normalize,
                                                dropout=self.dropout, verbose=self.verbose)
        self.target_encoder = NegativePhraseRAE(self.target_dim, initializer=initializer, normalize=self.normalize,
                                                dropout=self.dropout, verbose=self.verbose)
        self.source_pos = self.source_encoder.output
        self.source_neg = self.source_encoder.neg_output
        self.target_pos = self.target_encoder.output
        self.target_neg = self.target_encoder.neg_output
        # Define Bilingual Parameters
        self.Wsl = shared_rand_matrix(size=(self.target_dim, self.source_dim),
                                      name="Wsl", initializer=initializer)
        self.Wtl = shared_rand_matrix(size=(self.source_dim, self.target_dim),
                                      name="Wtl", initializer=initializer)
        self.bsl = shared_rand_matrix(size=(self.target_dim, ), name="bsl", initializer=initializer)
        self.btl = shared_rand_matrix(size=(self.source_dim, ), name="btl", initializer=initializer)
        self.param = [self.Wsl, self.Wtl, self.bsl, self.btl]
        self.loss_l2 = sum(T.sum(param ** 2) for param in [self.Wsl, self.Wtl]) * self.weight_sem

        def sem_distance(p1, w1, b1, p2):
            transform_p1 = T.tanh(T.dot(w1, p1) + b1)
            return T.sum((p2 - transform_p1) ** 2) / 2

        def sem_sim_distance(p1, w1, b1, p2):
            transform_p1 = T.tanh(T.dot(w1, p1) + b1)
            return cosine_similarity(transform_p1, p2)

        self.source_pos_sem = sem_distance(self.source_pos, self.Wsl, self.bsl, self.target_pos)
        self.target_pos_sem = sem_distance(self.target_pos, self.Wtl, self.btl, self.source_pos)
        self.source_neg_sem = sem_distance(self.source_pos, self.Wsl, self.bsl, self.target_neg)
        self.target_neg_sem = sem_distance(self.target_pos, self.Wtl, self.btl, self.source_neg)
        self.source_tar_sim = sem_sim_distance(self.source_pos, self.Wsl, self.bsl, self.target_pos)
        self.target_src_sim = sem_sim_distance(self.target_pos, self.Wtl, self.btl, self.source_pos)
        self.max_margin_source = T.maximum(0.0, self.source_pos_sem - self.source_neg_sem + 1.0)
        self.max_margin_target = T.maximum(0.0, self.target_pos_sem - self.target_neg_sem + 1.0)

        self.loss_sem = self.max_margin_source + self.max_margin_target
        self.loss_rec = self.source_encoder.loss_rec + self.target_encoder.loss_rec
        self.loss_l2 = self.loss_l2 + (self.source_encoder.loss_l2 + self.target_encoder.loss_l2) * self.weight_rec
        self.loss = self.alpha * self.loss_rec + \
                    (1 - self.alpha) * self.loss_sem + \
                    self.loss_l2
        self.params = self.source_encoder.params + self.target_encoder.params + self.param
        self.inputs = [self.source_encoder.pos_vectors, self.source_encoder.neg_vectors,
                       self.target_encoder.pos_vectors, self.target_encoder.neg_vectors]
        self.input_grad = T.grad(self.loss, self.inputs)
        grads = T.grad(self.loss, self.params)
        self.updates = OrderedDict()
        self.single_updates = OrderedDict()
        self.grad = {}
        for param, grad in zip(self.params, grads):
            g = theano.shared(np.asarray(np.zeros_like(param.get_value()), dtype=theano.config.floatX))
            self.grad[param] = g
            self.updates[g] = g + grad
            self.single_updates[param] = param - grad * self.learning_rate
        self._compute_result_grad = theano.function(
            inputs=[self.source_encoder.pos_vectors, self.source_encoder.seq,
                    self.source_encoder.neg_vectors, self.source_encoder.neg_seq,
                    self.target_encoder.pos_vectors, self.target_encoder.seq,
                    self.target_encoder.neg_vectors, self.target_encoder.neg_seq],
            outputs=[self.alpha * self.loss_rec,
                     (1 - self.alpha) * self.loss_sem,
                     self.loss_l2] + self.input_grad,
            updates=self.updates,
            allow_input_downcast=True
        )
        self._compute_result_grad_single = theano.function(
            inputs=[self.source_encoder.pos_vectors, self.source_encoder.seq,
                    self.source_encoder.neg_vectors, self.source_encoder.neg_seq,
                    self.target_encoder.pos_vectors, self.target_encoder.seq,
                    self.target_encoder.neg_vectors, self.target_encoder.neg_seq],
            outputs=[self.alpha * self.loss_rec,
                     (1 - self.alpha) * self.loss_sem,
                     self.loss_l2] + self.input_grad,
            updates=self.single_updates,
            allow_input_downcast=True
        )
        self.get_source_output = theano.function(
            inputs=[self.source_encoder.pos_vectors, self.source_encoder.seq],
            outputs=self.source_encoder.output,
            allow_input_downcast=True
        )
        self.get_target_output = theano.function(
            inputs=[self.target_encoder.pos_vectors, self.target_encoder.seq],
            outputs=self.target_encoder.output,
            allow_input_downcast=True
        )
        self.get_sem_distance = theano.function(
            inputs=[self.source_encoder.pos_vectors, self.source_encoder.seq,
                    self.target_encoder.pos_vectors, self.target_encoder.seq],
            outputs=[self.source_tar_sim, self.target_src_sim],
            allow_input_downcast=True
        )

    def compute_result_grad(self, src_pos_nodes, src_pos_seq, src_neg_nodes, src_neg_seq,
                            tar_pos_nodes, tar_pos_seq, tar_neg_nodes, tar_neg_seq):
        # Function for Profile
        return self._compute_result_grad(src_pos_nodes, src_pos_seq, src_neg_nodes, src_neg_seq,
                                         tar_pos_nodes, tar_pos_seq, tar_neg_nodes, tar_neg_seq)

    def compute_result_grad_single(self, src_pos_nodes, src_pos_seq, src_neg_nodes, src_neg_seq,
                                   tar_pos_nodes, tar_pos_seq, tar_neg_nodes, tar_neg_seq):
        # Function for Profile
        return self._compute_result_grad_single(src_pos_nodes, src_pos_seq, src_neg_nodes, src_neg_seq,
                                                tar_pos_nodes, tar_pos_seq, tar_neg_nodes, tar_neg_seq)

    @staticmethod
    def rand_neg(pos_phrase, dict_size):
        ran_time = np.random.randint(1, len(pos_phrase))
        neg_phrase = copy.copy(pos_phrase)
        for i in xrange(ran_time):
            index = np.random.randint(len(pos_phrase))
            neg_phrase[index] = [np.random.randint(dict_size)]
        return neg_phrase

    @staticmethod
    def generate_neg_instance(pos_phrase, dict_size):
        ran_time = np.random.randint(len(pos_phrase)) + 1
        neg_phrase = copy.copy(pos_phrase)
        for i in xrange(ran_time):
            index = np.random.randint(len(pos_phrase))
            neg_phrase[index] = np.random.randint(1, dict_size)
        return neg_phrase

    def generate_train_instance(self, pos_phrase, dict_size):
        # neg_instance = self.generate_neg_instance(pos_phrase, dict_size) + [0] * (2 * align_len - len(pos_phrase) - 1)
        # pos_instance = pos_phrase + [0] * (2 * align_len - len(pos_phrase) - 1)
        pos_instance = pos_phrase + [0] * (len(pos_phrase) - 1)
        neg_instance = self.generate_neg_instance(pos_phrase, dict_size) + [0] * (len(pos_phrase) - 1)
        return pos_instance, neg_instance

    def generate_train_array(self, pos_phrases, dict_size):
        # phrase_list = list()
        # for phrase in pos_phrases:
        #     phrase_list.append([p[0] for p in phrase])
        instance = [self.generate_train_instance(p, dict_size) for p in pos_phrases]
        pos_instance = [ins[0] for ins in instance]
        neg_instance = [ins[1] for ins in instance]
        return np.array(pos_instance, dtype=np.int32), np.array(neg_instance, dtype=np.int32)

    def train(self, src_tar_pair, config, source_embedding, target_embedding):
        n_epoch = config.n_epoch
        batch_size = config.batch_size
        size_src_word = source_embedding.size
        size_tar_word = target_embedding.size
        train_index = get_train_sequence(src_tar_pair, config.batch_size)
        num_batch = len(train_index) / config.batch_size
        if self.verbose:
            logger.debug("Train Details")
            logger.debug("Number of epochs: %d" % n_epoch)
            logger.debug("Size of Batches:  %d" % batch_size)
            logger.debug("Size phrase pair: %d" % len(src_tar_pair))
            logger.debug("Size Source word: %d" % size_src_word)
            logger.debug("Size Target word: %d" % size_tar_word)
            logger.debug("Start Training ...")
        print "Start Training ..."
        # optimizer = generate_optimizer(self.params, config.optimizer)
        source_embedding = source_embedding.W.get_value()
        target_embedding = target_embedding.W.get_value()
        for i in range(n_epoch):
            print "epoch ", i + 1
            loss = []
            for j in range(num_batch):
                if self.verbose:
                    print progress_bar_str(j, num_batch) + "\r",
                # train_batch_data = []
                if batch_size == 1:
                    src_pos, tar_pos = src_tar_pair[j]
                    src_pos, src_neg = self.generate_train_instance(src_pos, size_src_word)
                    tar_pos, tar_neg = self.generate_train_instance(tar_pos, size_tar_word)
                    # src_pos = [x[0] for x in src_pos]
                    # tar_pos = [x[0] for x in tar_pos]
                    # src_neg = [x[0] for x in src_neg]
                    # tar_neg = [x[0] for x in tar_neg]
                    src_pos_vec = source_embedding[src_pos]
                    src_neg_vec = source_embedding[src_neg]
                    tar_pos_vec = target_embedding[tar_pos]
                    tar_neg_vec = target_embedding[tar_neg]
                    src_pos_nodes, src_pos_seq = self.source_encoder.generate_node_path(src_pos_vec)
                    tar_pos_nodes, tar_pos_seq = self.target_encoder.generate_node_path(tar_pos_vec)
                    src_neg_nodes, src_neg_seq = self.source_encoder.generate_node_path(src_neg_vec)
                    tar_neg_nodes, tar_neg_seq = self.target_encoder.generate_node_path(tar_neg_vec)
                    result = self.compute_result_grad_single(src_pos_nodes, src_pos_seq,
                                                             src_neg_nodes, src_neg_seq,
                                                             tar_pos_nodes, tar_pos_seq,
                                                             tar_neg_nodes, tar_neg_seq)
                    rec_err, sem_err, l2_err = result[:3]
                    """print src_pos, tar_pos
                    print src_neg, tar_neg
                    print "%.2f\t%.2f\t%.2f\t" % (rec_err, sem_err, l2_err),
                    for r in result[3:7]:
                        print "%2.2f\t" % (r),
                    print"""
                    # print self.Wsl.get_value()
                    # print self.Wtl.get_value()
                    src_pos_grad, src_neg_grad, tar_pos_grad, tar_neg_grad = result[3:]
                    g = config.optimizer.param["lr"] * self.weight_l2
                    # source_embedding *= 1 - g
                    # target_embedding *= 1 - g
                    for index, value in zip(src_pos, src_pos_grad):
                        source_embedding[index] -= value * config.optimizer.param["lr"]
                    for index, value in zip(src_neg, src_neg_grad):
                        source_embedding[index] -= value * config.optimizer.param["lr"]
                    for index, value in zip(tar_pos, tar_pos_grad):
                        target_embedding[index] -= value * config.optimizer.param["lr"]
                    for index, value in zip(tar_neg, tar_neg_grad):
                        target_embedding[index] -= value * config.optimizer.param["lr"]
                    # if self.verbose:
                    #     print rec_err, sem_err, l2_err, src_pos, src_neg, tar_pos, tar_neg
                    loss.append([rec_err, sem_err, l2_err])
                    #  + self.weight_l2 * (sum(source_embedding ** 2) + sum(target_embedding ** 2))])
                """if batch_size > 1:
                    for k in range(batch_size):
                        index = j * batch_size + k
                        src_pos, tar_pos = src_tar_pair[index]
                        src_neg = self.rand_neg(src_pos, size_src_word)
                        tar_neg = self.rand_neg(tar_pos, size_tar_word)
                        src_pos = [x[0] for x in src_pos]
                        tar_pos = [x[0] for x in tar_pos]
                        src_neg = [x[0] for x in src_neg]
                        tar_neg = [x[0] for x in tar_neg]
                        src_pos_nodes, src_pos_seq = self.source_encoder.generate_node_path(src_pos)
                        tar_pos_nodes, tar_pos_seq = self.target_encoder.generate_node_path(tar_pos)
                        src_neg_nodes, src_neg_seq = self.source_encoder.generate_node_path(src_neg)
                        tar_neg_nodes, tar_neg_seq = self.target_encoder.generate_node_path(tar_neg)
                        train_batch_data.append((src_pos, src_neg, tar_pos, tar_neg,
                                                 src_pos_nodes, src_pos_seq,
                                                 tar_pos_nodes, tar_pos_seq,
                                                 src_neg_nodes, src_neg_seq,
                                                 tar_neg_nodes, tar_neg_seq,))
                    for k in range(batch_size):
                        src_pos, src_neg, tar_pos, tar_neg, \
                        src_pos_nodes, src_pos_seq, \
                        tar_pos_nodes, tar_pos_seq, \
                        src_neg_nodes, src_neg_seq, \
                        tar_neg_nodes, tar_neg_seq = train_batch_data[k]
                        result = self.compute_result_grad(src_pos_nodes, src_pos_seq,
                                                          src_neg_nodes, src_neg_seq,
                                                          tar_pos_nodes, tar_pos_seq,
                                                          tar_neg_nodes, tar_neg_seq)
                        rec_err, sem_err, l2_err = result[:3]
                        # if self.verbose:
                        #     print rec_err, sem_err, l2_err, src_pos, src_neg, tar_pos, tar_neg
                        loss.append([rec_err, sem_err, l2_err])
                    for _, grad in self.grad.iteritems():
                        grad.set_value(grad.get_value() / batch_size)
                    optimizer.iterate(self.grad)"""
            print
            logger.info("epoch %d rec_err %f sem_err %f" % (i + 1, np.mean(loss, axis=0)[0], np.mean(loss, axis=0)[1]))
            logger.info("epoch %d l2_err  %f sum_err %f" % (i + 1, np.mean(loss, axis=0)[2], sum(np.mean(loss, axis=0))))

    def predict(self, src_tar_pair, output_file):
        sem_pair = list()
        for src, tar in src_tar_pair:
            src_nodes, src_seq = self.source_encoder.generate_node_path(src)
            tar_nodes, tar_seq = self.target_encoder.generate_node_path(tar)
            src_sem, tar_sem = self.get_sem_distance(src_nodes, src_seq, tar_nodes, tar_seq)
            sem_pair.append((src_sem, tar_sem))
        f_w = open(output_file, 'w')
        for src, tar in sem_pair:
            f_w.write("%f ||| %f\n" % (src, tar))
        f_w.close()
