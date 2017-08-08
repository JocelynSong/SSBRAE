import abc
import logging
import sys

import numpy as np
import theano
import theano.tensor as T

sys.path.append('../../')
from src import default_initializer
from src.utils import shared_rand_matrix, align_batch_size
from src.Initializer import UniformInitializer
from src.embedding import WordEmbedding
from src.activations import Activation
from src.optimizer import AdaDeltaOptimizer

logger = logging.getLogger(__name__)

class EntityScorer(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self): pass

    @abc.abstractmethod
    def score(self, entity1, entity2, relation): pass


class DistanceModel(EntityScorer):
    def __init__(self, entity_embedding, relation_embedding):
        super(DistanceModel, self).__init__()


class SingleLayerModel(EntityScorer):
    def __init__(self, entity_dim, relation_num, k=50, activation='tanh',
                 initializer=default_initializer, prefix='', verbose=True):
        super(SingleLayerModel, self).__init__()
        self.k = k
        self.entity_dim = entity_dim
        self.relation_num = relation_num
        # (relation_num, k, entity_dim)
        self.W_1 = shared_rand_matrix((relation_num, self.k, self.entity_dim), prefix + 'W_1', initializer)
        # (relation_num, k, entity_dim)
        self.W_2 = shared_rand_matrix((relation_num, self.k, self.entity_dim), prefix + 'W_2', initializer)
        # (relation_num, k, )
        self.u = shared_rand_matrix((relation_num, self.k, ), prefix + 'u', initializer)
        self.act = Activation(activation)
        self.params = [self.W_1, self.W_2, self.u]

        self.l1_norm = T.sum(T.abs_(self.W_1)) + T.sum(T.abs_(self.W_2)) + T.sum(T.abs_(self.u))
        self.l2_norm = T.sum(self.W_1 ** 2) + T.sum(self.W_2 ** 2) + T.sum(self.u ** 2)

        if verbose:
            logger.debug('Architecture of Single Layer Model built finished, summarized as below:')
            logger.debug('Entity Dimension: %d' % self.entity_dim)
            logger.debug('K Dimension:      %d' % self.k)
            logger.debug('Relation Number:  %d' % self.relation_num)


    def score(self, e1, e2, r_index):
        """
        :param e1: (entity_dim, )
        :param e2: (entity_dim, )
        :param r_index: scalar
        :return: 
        """
        # (k, entity_dim) dot (entity_dim) + (k, entity_dim) dot (entity_dim) -> (k, )
        hidden = T.dot(self.W_1[r_index], e1) + T.dot(self.W_2[r_index], e2)
        # (k, ) -> (k, )
        act_hidden = self.act.activate(hidden)
        # (k, ) dot (k, ) -> 1
        return T.dot(self.u[r_index], act_hidden)

    def score_batch(self, e1, e2, r_index):
        """
        :param e1: (batch, entity_dim, )
        :param e2: (batch, entity_dim, )
        :param r_index: (batch, )
        :return: 
        """
        # (batch, k, entity_dim) dot (batch, entity_dim) + (batch, k, entity_dim) dot (batch, entity_dim)
        hidden = T.batched_dot(self.W_1[r_index], e1)
        hidden += T.batched_dot(self.W_2[r_index], e2)
        # (batch, k) -> (batch, k)
        act_hidden = self.act.activate(hidden)
        # (batch, k) dot (batch, k) -> (batch, )
        return T.sum(act_hidden * self.u[r_index], axis=1)

    def score_one_relation(self, e1, e2, r_index):
        """
        :param e1: (batch, entity_dim, )
        :param e2: (batch, entity_dim, )
        :param r_index: scalar
        :return: 
        """
        # (batch, entity_dim) dot (entity_dim, k) + (batch, entity_dim) dot (entity_dim, k) -> (batch, k)
        hidden = T.dot(e1, self.W_1[r_index].transpose()) + T.dot(e2, self.W_2[r_index].transpose())
        # (batch, k) -> (batch, k)
        act_hidden = self.act.activate(hidden)
        # (batch, k) dot (k, ) -> (batch, )
        return T.dot(act_hidden, self.u[r_index])


class ReasonTrainer(object):
    def __init__(self, entity_index, relation_index, entity_dim=100, k=100,
                 initializer=default_initializer, regularization_weight=0.0001):
        self.relation_num = len(relation_index)
        self.scorer = SingleLayerModel(entity_dim=entity_dim, relation_num=self.relation_num,
                                       k=k, initializer=UniformInitializer(scale=1 / np.sqrt(entity_dim * 2)))
        self.entity_embedding = WordEmbedding(entity_index, dim=entity_dim, initializer=initializer)
        self.regularization_weight = regularization_weight
        self.e1_index = T.lscalar()
        self.e2_index = T.lscalar()
        self.ec_index = T.lscalar()
        self.relation_index = T.lscalar()
        self.pos_score = self.scorer.score(self.entity_embedding[self.e1_index],
                                           self.entity_embedding[self.e2_index],
                                           self.relation_index)
        self.neg_score = self.scorer.score(self.entity_embedding[self.e1_index],
                                           self.entity_embedding[self.ec_index],
                                           self.relation_index)
        self.loss_max_margin =  T.maximum(0.0, self.neg_score - self.pos_score + 1.0)

        self.e1_index_batch = T.lvector()
        self.e2_index_batch = T.lvector()
        self.ec_index_batch = T.lvector()
        self.relation_index_batch = T.lvector()
        self.pos_score_batch = self.scorer.score_batch(self.entity_embedding[self.e1_index_batch],
                                                       self.entity_embedding[self.e2_index_batch],
                                                       self.relation_index_batch)
        self.neg_score_batch = self.scorer.score_batch(self.entity_embedding[self.e1_index_batch],
                                                       self.entity_embedding[self.ec_index_batch],
                                                       self.relation_index_batch)
        self.loss_max_margin_batch =  T.sum(T.maximum(0.0, self.neg_score_batch - self.pos_score_batch + 1.0))

        self.pos_score_relation = self.scorer.score_one_relation(self.entity_embedding[self.e1_index_batch],
                                                              self.entity_embedding[self.e2_index_batch],
                                                              self.relation_index)
        self.neg_score_relation = self.scorer.score_one_relation(self.entity_embedding[self.e1_index_batch],
                                                              self.entity_embedding[self.ec_index_batch],
                                                              self.relation_index)
        self.loss_max_margin_relation =  T.sum(T.maximum(0.0, self.neg_score_relation - self.pos_score_relation + 1.0))

        self.params = self.entity_embedding.params + self.scorer.params
        self.l2_norm = self.entity_embedding.l2_norm + self.scorer.l2_norm
        self.l2_loss = self.regularization_weight * self.l2_norm / 2
        sgd_optimizer = AdaDeltaOptimizer(lr=0.95, norm_lim=-1)

        self.loss = self.loss_max_margin + self.l2_loss
        updates = sgd_optimizer.get_update(self.loss, self.params)

        self.loss_batch = self.loss_max_margin_batch + self.l2_loss
        updates_batch = sgd_optimizer.get_update(self.loss_batch, self.params)

        grad_margin_relation = T.grad(self.loss_max_margin_relation, self.params)
        grad_l2 = T.grad(self.l2_loss, self.params)

        self.train_one_instance = theano.function(inputs=[self.e1_index, self.e2_index,
                                                          self.ec_index, self.relation_index],
                                                  outputs=[self.loss, self.loss_max_margin, self.l2_loss],
                                                  updates=updates)

        self.score_one_instance = theano.function(inputs=[self.e1_index, self.e2_index, self.relation_index],
                                                  outputs=[self.pos_score])

        self.train_batch_instance = theano.function(inputs=[self.e1_index_batch, self.e2_index_batch,
                                                            self.ec_index_batch, self.relation_index_batch],
                                                    outputs=[self.loss_batch, self.loss_max_margin_batch, self.l2_loss],
                                                    updates=updates_batch)

        self.score_batch_instance = theano.function(inputs=[self.e1_index_batch, self.e2_index_batch,
                                                            self.relation_index_batch],
                                                    outputs=self.pos_score_batch)

        self.grad_relation_margin = theano.function(inputs=[self.e1_index_batch, self.e2_index_batch,
                                                            self.ec_index_batch, self.relation_index],
                                                    outputs=[self.loss_max_margin_relation] + grad_margin_relation,
                                                    )

        self.forward_relation_margin = theano.function(inputs=[self.e1_index_batch, self.e2_index_batch,
                                                               self.ec_index_batch, self.relation_index],
                                                       outputs=[self.loss_max_margin_relation],
                                                       )

        self.grad_l2 = theano.function(inputs=[], outputs=[self.l2_loss] + grad_l2,)

        self.forward_l2 = theano.function(inputs=[], outputs=[self.l2_loss],)

        self.score_relation_instance = theano.function(inputs=[self.e1_index_batch, self.e2_index_batch,
                                                               self.relation_index],
                                                       outputs=self.pos_score_relation)

    def score(self, e1, e2, r):
        if type(e1) == list or type(e2) == list or type(r) == list:
            assert len(e1) == len(e2) and len(e1) == len(r)
            return self.score_batch_instance(e1, e2, r)
        elif type(e1) == np.ndarray or type(e2) == np.ndarray or type(r) == np.ndarray:
            # assert len(e1) == len(e2) and len(e1) == len(r)
            return self.score_batch_instance(e1, e2, r)
        else:
            return self.score_one_instance(e1, e2, r)

    def find_threshold(self, pos_scores, neg_scores):
        pos_num = len(pos_scores)
        neg_sum = len(neg_scores)
        best_acc = 0
        best_threshold = 0
        merge = np.concatenate((pos_scores, neg_scores))
        merge_scores_sort = sorted(merge)
        for i in xrange(len(merge_scores_sort)):
            threshold = merge_scores_sort[i]
            pos_right = np.sum(pos_scores >= threshold)
            neg_right = np.sum(neg_scores < threshold)
            acc = (pos_right + neg_right) / float(pos_num +  neg_sum)
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold
        return best_threshold

    def accuracy(self, dev_data, test_data):
        # Num of relations, 4 (TP, FN, FP, TN)
        dev_predict = np.zeros((len(dev_data), 4))
        test_predict = np.zeros((len(dev_data), 4))
        for r_id in dev_data.keys():
            dev_pos, dev_neg = dev_data[r_id]["pos"], dev_data[r_id]["neg"]
            test_pos, test_neg = test_data[r_id]["pos"], test_data[r_id]["neg"]
            dev_pos_scores = self.score(dev_pos[:, 0], dev_pos[:, 1], dev_pos[:, 2])
            dev_neg_scores = self.score(dev_neg[:, 0], dev_neg[:, 1], dev_neg[:, 2])
            test_pos_scores = self.score(test_pos[:, 0], test_pos[:, 1], test_pos[:, 2])
            test_neg_scores = self.score(test_neg[:, 0], test_neg[:, 1], test_neg[:, 2])
            threshold = self.find_threshold(dev_pos_scores, dev_neg_scores)
            # print threshold,
            test_predict[r_id][0] = sum(test_pos_scores >= threshold)
            test_predict[r_id][1] = sum(test_pos_scores < threshold)
            test_predict[r_id][2] = sum(test_neg_scores >= threshold)
            test_predict[r_id][3] = sum(test_neg_scores < threshold)
            dev_predict[r_id][0] = sum(dev_pos_scores >= threshold)
            dev_predict[r_id][1] = sum(dev_pos_scores < threshold)
            dev_predict[r_id][2] = sum(dev_neg_scores >= threshold)
            dev_predict[r_id][3] = sum(dev_neg_scores < threshold)
        # print
        dev_acc = (np.sum(dev_predict[:, 0]) + np.sum(dev_predict[:, 3])) / np.sum(dev_predict)
        test_acc = (np.sum(test_predict[:, 0]) + np.sum(test_predict[:, 3])) / np.sum(test_predict)
        return dev_acc, test_acc

    def train_batch(self, triples, dev, test, batch_size=500, max_iter=500, C=5):
        neg_entity_list = dict()
        for _, e2, r in triples:
            if r not in neg_entity_list:
                neg_entity_list[r] = list()
            neg_entity_list[r].append(e2)
        train_data = np.zeros((len(triples), 3 + C), dtype=np.int64)
        for triple_index in xrange(len(triples)):
            e1_index, e2_index, r_index = triples[triple_index]
            train_data[triple_index][0] = e1_index
            train_data[triple_index][1] = e2_index
            train_data[triple_index][2] = r_index
            for j in xrange(C):
                train_data[triple_index][3 + j] = np.random.choice(neg_entity_list[r_index])
        train_indexs = align_batch_size(range(train_data.shape[0]), batch_size=batch_size)
        num_batch = len(train_indexs) / batch_size
        for i in xrange(max_iter):
            for triple_index in xrange(len(triples)):
                _, _, r_index = triples[triple_index]
                train_data[triple_index][2] = np.random.choice(neg_entity_list[r_index])
            np.random.permutation(train_indexs)
            loss = 0
            max_margin_loss = 0
            l2_loss = 0
            for j in xrange(num_batch):
                losses = self.train_batch_instance(train_data[train_indexs[j * batch_size: (j + 1) * batch_size], 0],
                                                   train_data[train_indexs[j * batch_size: (j + 1) * batch_size], 1],
                                                   train_data[train_indexs[j * batch_size: (j + 1) * batch_size], 2],
                                                   train_data[train_indexs[j * batch_size: (j + 1) * batch_size], 3])
                loss += losses[0]
                max_margin_loss += losses[1]
                l2_loss += losses[2]
            dev_acc, test_acc = self.accuracy(dev, test)
            logger.info("Iter %d: train loss: %s, margin loss: %s, l2 loss: %s" % (i + 1, loss, max_margin_loss, l2_loss))
            logger.info("Iter %d: dev acc: %s, test acc: %s" % (i + 1, dev_acc, test_acc))

    def train_relation(self, triples, dev, test, max_iter=50, C=1, batch_size=2000):
        neg_entity_list = dict()
        for _, e2, r in triples:
            if r not in neg_entity_list:
                neg_entity_list[r] = list()
            neg_entity_list[r].append(e2)
        '''train_data = np.zeros((len(triples), 3 + C), dtype=np.int64)
        logger.info("Start Generate Negative Instance ...")
        for triple_index in xrange(len(triples)):
            e1_index, e2_index, r_index = triples[triple_index]
            train_data[triple_index][0] = e1_index
            train_data[triple_index][1] = e2_index
            train_data[triple_index][2] = r_index
            for j in xrange(C):
                train_data[triple_index][3 + j] = np.random.choice(neg_entity_list[r_index])'''
        import cPickle
        out = open("data.txt", 'rb')
        data = cPickle.load(out)
        out.close()
        logger.info("Finish Generate Negative Instance")
        params_size = list()
        params_shape = list()
        for param in self.params:
            shape = param.get_value().shape
            params_size.append(np.prod(shape))
            params_shape.append(shape)
        iter_index = [0]

        train_data = None
        dev_acc_list = list()
        test_acc_list = list()

        def minimize_me(vars):
            # unpack param
            vars_index = 0
            for param, size, shape in zip(self.params, params_size, params_shape):
                param.set_value(vars[vars_index: vars_index + size].reshape(shape))
                vars_index += size
            # get loss and gradients from theano function
            grad = np.zeros(np.sum(params_size))
            loss, max_margin_loss, l2_loss = 0, 0, 0
            for relation_index in xrange(self.relation_num):
                train_relation_data = train_data[train_data[:, 2] == relation_index, :]
                for c_index in xrange(C):
                    losses = self.grad_relation_margin(train_relation_data[:, 0],
                                                       train_relation_data[:, 1],
                                                       train_relation_data[:, 3 + c_index],
                                                       relation_index,
                                                       )
                    max_margin_loss += losses[0]
                    dloss = np.concatenate([param.ravel() for param in losses[1: ]])
                    grad += dloss
            grad = grad / train_data.shape[0]
            max_margin_loss = max_margin_loss / train_data.shape[0]
            losses = self.grad_l2()
            l2_loss = losses[0]
            dloss = np.concatenate([param.ravel() for param in losses[1: ]])
            grad += dloss
            loss = max_margin_loss + l2_loss
            # fmin_l_bfgs_b needs double precision...
            return loss.astype('float64'), grad.astype('float64')

        def test_me(x):
            iter_index[0] += 1
            loss, max_margin_loss, l2_loss = 0, 0, 0
            for relation_index in xrange(self.relation_num):
                train_relation_data = train_data[train_data[:, 2] == relation_index, :]
                for c_index in xrange(C):
                    losses = self.forward_relation_margin(train_relation_data[:, 0],
                                                          train_relation_data[:, 1],
                                                          train_relation_data[:, 3 + c_index],
                                                          relation_index,
                                                          )
                    max_margin_loss += losses[0]
            max_margin_loss = max_margin_loss / train_data.shape[0]
            losses = self.forward_l2()
            l2_loss = losses[0]
            loss = max_margin_loss + l2_loss
            logger.info("Iter %d: train loss: %s, margin loss: %s, l2 loss: %s" % (iter_index[0], loss,
                                                                                   max_margin_loss, l2_loss))
            dev_acc, test_acc = self.accuracy(dev, test)
            dev_acc_list.append(dev_acc)
            test_acc_list.append(test_acc)
            logger.info("Iter %d: dev acc: %s, test acc: %s" % (iter_index[0], dev_acc, test_acc))

        print "Start Minimize"
        DEFAULT_LBFGS_PARAMS = dict(iprint=0, factr=10, maxfun=1e4, maxiter=5)
        from scipy.optimize import fmin_l_bfgs_b
        train_indexs = align_batch_size(range(data.shape[0]), batch_size=batch_size)
        for i in xrange(max_iter):
            np.random.permutation(train_indexs)
            train_index = train_indexs[:batch_size]
            vars = np.concatenate([param.get_value().ravel() for param in self.params])
            train_data = data[train_index]
            best, bestval, info = fmin_l_bfgs_b(minimize_me, vars, **DEFAULT_LBFGS_PARAMS)
            vars_index = 0
            for param, size, shape in zip(self.params, params_size, params_shape):
                param.set_value(best[vars_index: vars_index + size].reshape(shape))
                vars_index += size
            test_me(None)
        dev_acc_list, test_acc_list = np.array(dev_acc_list), np.array(test_acc_list)
        max_index = np.argmax(dev_acc_list)
        logger.info("Max Dev Acc Iter %s: dev acc: %s, test acc: %s" % (max_index + 1, dev_acc_list[max_index],
                                                                        test_acc_list[max_index]))

