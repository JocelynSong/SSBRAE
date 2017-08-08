import logging

import numpy as np
import theano
import theano.tensor as T

from src import default_initializer, BIG_INT
from src.activations import Activation
from src.classifier import SoftmaxClassifier
from src.optimizer import AdaGradOptimizer
from src.utils import shared_rand_matrix, align_batch_size, shared_zero_matrix, ndarray_slice

__author__ = 'roger'
logger = logging.getLogger(__name__)


def get_pooling(hs, pooling_method):
    if pooling_method == 'max':
        T.max(hs, axis=0)
    elif pooling_method == 'averaging':
        return T.mean(hs, axis=0)
    elif pooling_method == 'sum':
        return T.sum(hs, axis=0)
    elif pooling_method == 'final':
        return hs[-1]
    else:
        raise NotImplementedError('Not implemented pooling method: {}'.format(pooling_method))


def get_pooling_batch(hs, mask, pooling_method):
    """
    :param hs:   (batch, len, dim)
    :param mask: (batch, len)
    :param pooling_method:
    :return:
    """
    print mask.ndim
    if pooling_method == 'max':
        add_v = ((1 - mask) * -BIG_INT)[:, :, None]
        return T.max(hs + add_v, axis=1)
    elif pooling_method == 'averaging':
        return T.sum(hs * mask[:, :, None], axis=1) / T.sum(mask, axis=1)[:, None]
    elif pooling_method == 'sum':
        return T.sum(hs * mask[:, :, None], axis=1)
    elif pooling_method == 'final':
        return hs[:, -1, :]
    else:
        raise NotImplementedError('Not implemented pooling method: {}'.format(pooling_method))


one_int32 = T.constant(1, dtype=np.int32)
one_float32 = T.constant(1, dtype=theano.config.floatX)


class RecurrentEncoder(object):
    def __init__(self, in_dim, hidden_dim, pooling, activation, prefix="",
                 initializer=default_initializer, dropout=0, verbose=True):
        if verbose:
            logger.debug('Building {}...'.format(self.__class__.__name__))
        self.in_dim = in_dim
        self.out_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.pooling = pooling
        self.dropout = dropout
        self.act = Activation(activation)
        # Composition Function Weight
        # Feed-Forward Matrix (hidden, in)
        self.W = shared_rand_matrix((self.hidden_dim, self.in_dim), prefix + 'W_forward', initializer)
        # Bias Term (hidden)
        self.b = shared_zero_matrix((self.hidden_dim,), prefix + 'b_forward')
        # Recurrent Matrix (hidden, hidden)
        self.U = shared_rand_matrix((self.hidden_dim, self.hidden_dim), prefix + 'U_forward', initializer)

        self.params = [self.W, self.U, self.b]
        self.norm_params = [self.W, self.U]

        # L1, L2 Norm
        self.l1_norm = T.sum(T.abs_(self.W)) + T.sum(T.abs_(self.U))
        self.l2_norm = T.sum(self.W ** 2) + T.sum(self.U ** 2)

        if verbose:
            logger.debug('Architecture of {} built finished'.format(self.__class__.__name__))
            logger.debug('Input dimension:  %d' % self.in_dim)
            logger.debug('Hidden dimension: %d' % self.hidden_dim)
            logger.debug('Pooling methods:  %s' % self.pooling)
            logger.debug('Activation Func:  %s' % self.act.method)
            logger.debug('Dropout Rate:     %f' % self.dropout)

    def _step(self, x_t, h_t_1, W, U, b):
        """
        step function of forward
        :param x_t:   (in, )
        :param h_t_1: (hidden, )
        :param W:     (hidden, in)
        :param U:     (hidden, hidden)
        :param b:     (hidden, )
        :return:      (hidden)
        """
        # (hidden, in) (in, ) + (hidden, hidden) (hidden, ) + (hidden, ) -> hidden
        h_t = self.act.activate(T.dot(W, x_t) + T.dot(U, h_t_1) + b)
        return h_t

    def _step_batch(self, x_t, mask, h_t_1, W, U, b):
        """
        step function of forward in batch version
        :param x_t:   (batch, in)
        :param mask:  (batch, )
        :param h_t_1: (batch, hidden)
        :param W:     (hidden, in)
        :param U:     (hidden, hidden)
        :param b:     (hidden)
        :return:      (batch, hidden)
        """
        # (batch, in) (in, hidden) -> (batch, hidden)
        h_t = self.act.activate(T.dot(x_t, W.T) + T.dot(h_t_1, U.T) + b)
        # (batch, hidden) * (batch, None) + (batch, hidden) * (batch, None) -> (batch, hidden)
        return h_t * mask[:, None] + h_t_1 * (1 - mask[:, None])

    def forward_scan(self, x):
        h0 = shared_zero_matrix((self.hidden_dim,), 'h0_forward')
        hs, _ = theano.scan(fn=self._step,
                            sequences=x,
                            outputs_info=[h0],
                            non_sequences=[self.W, self.U, self.b],
                            )
        return hs

    def forward_scan_batch(self, x, mask, batch_size):
        """
        :param x: (batch, max_len, dim)
        :param mask:  (batch, max_len)
        :param batch_size:
        """
        h0 = shared_zero_matrix((batch_size, self.hidden_dim), 'h0_forward')
        hs, _ = theano.scan(fn=self._step_batch,
                            sequences=[T.transpose(x, (1, 0, 2)),  # (batch, max_len, dim) -> (max_len, batch, dim)
                                       T.transpose(mask, (1, 0))],     # (batch, max_len) -> (max_len, batch)
                            outputs_info=[h0],
                            non_sequences=[self.W, self.U, self.b],
                            )
        # (max_len, batch, dim) -> (batch, max_len, dim)
        return T.transpose(hs, (1, 0, 2))

    def forward_sequence(self, x):
        return self.forward_scan(x)

    def forward_sequence_batch(self, x, mask, batch_size):
        """
        :param x: (batch, max_len, dim)
        :param mask:  (batch, max_len)
        :param batch_size:
        """
        return self.forward_scan_batch(x, mask, batch_size)

    def forward(self, x):
        """
        :param x: (len, dim)
        """
        # Use Pooling to reduce into a fixed-length representation
        return get_pooling(self.forward_sequence(x), self.pooling)

    def forward_batch(self, x, mask, batch_size):
        """
        :param x: (batch, max_len, dim)
        :param mask:  (batch, max_len)
        :param batch_size:
        """
        # Use Pooling to reduce into a fixed-length representation
        # (max_len, batch, dim) -> (batch, max_len, dim) -> (batch, dim)
        hidden = self.forward_sequence_batch(x, mask, batch_size)
        return get_pooling_batch(hidden, mask, self.pooling)


class MultiLayerRecurrentEncoder(object):
    def __init__(self, in_dim, hidden_dims, pooling, activation, prefix="",
                 initializer=default_initializer, dropout=0, verbose=True):
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = hidden_dims[-1]
        layer_num = len(self.hidden_dims)
        self.activations = [activation] * layer_num if type(activation) is not [list, tuple] else activation
        self.poolings = [pooling] * layer_num if type(pooling) is not [list, tuple] else pooling
        self.initializers = [initializer] * layer_num if type(initializer) is not [list, tuple] else initializer
        self.dropouts = [dropout] * layer_num if type(dropout) is not [list, tuple] else dropout
        self.layers = [RecurrentEncoder(d_in, d_h, pooling=pool, activation=act, prefix=prefix + "layer%d_" % i,
                                        initializer=init, dropout=drop, verbose=verbose)
                       for d_in, d_h, pool, act, i, init, drop
                       in zip([in_dim] + hidden_dims, hidden_dims, self.poolings, self.activations, range(layer_num),
                              self.initializers, self.dropouts)]
        self.params = []
        self.norm_params = []
        for layer in self.layers:
            self.params += layer.params
            self.norm_params += layer.norm_params
        self.l1_norm = T.sum([T.sum(T.abs_(param)) for param in self.norm_params])
        self.l2_norm = T.sum([T.sum(param ** 2) for param in self.norm_params])
        if verbose:
            logger.debug('Architecture of {} built finished'.format(self.__class__.__name__))
            logger.debug('Layer Num:  %d' % layer_num)

    def forward_sequence(self, x):
        hidden = x
        for layer in self.layers:
            hidden = layer.forward_sequence(hidden)
        return hidden

    def forward_sequence_batch(self, x, mask, batch_size):
        hidden = x
        for layer in self.layers:
            hidden = layer.forward_sequence_batch(hidden, mask, batch_size)
        return hidden

    def forward(self, x):
        """
        :param x: (len, dim)
        """
        # Use Pooling to reduce into a fixed-length representation
        return get_pooling(self.forward_sequence(x), self.poolings[-1])

    def forward_batch(self, x, mask, batch_size):
        """
        :param x: (batch, max_len, dim)
        :param mask:  (batch, max_len)
        :param batch_size:
        """
        # Use Pooling to reduce into a fixed-length representation
        # (max_len, batch, dim) -> (batch, max_len, dim) -> (batch, dim)
        hidden = self.forward_sequence_batch(x, mask, batch_size)
        return get_pooling_batch(hidden, mask, self.poolings[-1])


class LSTMEncoder(object):
    def __init__(self, in_dim, hidden_dim, pooling, activation, gates=("sigmoid", "sigmoid", "sigmoid"), prefix="",
                 initializer=default_initializer, dropout=0, verbose=True):
        if verbose:
            logger.debug('Building {}...'.format(self.__class__.__name__))
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = hidden_dim
        self.pooling = pooling
        self.act = Activation(activation)
        self.in_gate, self.forget_gate, self.out_gate = Activation(gates[0]), Activation(gates[1]), Activation(gates[2])
        self.dropout = dropout

        # W [in, forget, output, recurrent] (4 * hidden, in)
        self.W = shared_rand_matrix((self.hidden_dim * 4, self.in_dim), prefix + 'W', initializer)
        # U [in, forget, output, recurrent] (4 * hidden, hidden)
        self.U = shared_rand_matrix((self.hidden_dim * 4, self.hidden_dim), prefix + 'U', initializer)
        # b [in, forget, output, recurrent] (4 * hidden,)
        self.b = shared_zero_matrix((self.hidden_dim * 4,), prefix + 'b')

        self.params = [self.W, self.U, self.b]
        self.l1_norm = T.sum(T.abs_(self.W)) + T.sum(T.abs_(self.U))
        self.l2_norm = T.sum(self.W ** 2) + T.sum(self.U ** 2)

        if verbose:
            logger.debug('Architecture of {} built finished'.format(self.__class__.__name__))
            logger.debug('Input dimension:  %d' % self.in_dim)
            logger.debug('Hidden dimension: %d' % self.hidden_dim)
            logger.debug('Pooling methods:  %s' % self.pooling)
            logger.debug('Activation Func:  %s' % self.act.method)
            logger.debug('Input Gate:       %s' % self.in_gate.method)
            logger.debug('Forget Gate:      %s' % self.forget_gate.method)
            logger.debug('Output Gate:      %s' % self.out_gate.method)
            logger.debug('Activation Func:  %s' % self.act.method)
            logger.debug('Dropout Rate:     %f' % self.dropout)

    def _step(self, x_t, h_t_1, c_t_1, W, U, b):
        pre_calc = T.dot(x_t, W) + T.dot(h_t_1, U) + b
        i_t = self.in_gate.activate(ndarray_slice(pre_calc, 0, self.hidden_dim))
        f_t = self.forget_gate.activate(ndarray_slice(pre_calc, 1, self.hidden_dim))
        o_t = self.out_gate.activate(ndarray_slice(pre_calc, 2, self.hidden_dim))
        g_t = self.act.activate(ndarray_slice(pre_calc, 3, self.hidden_dim))
        c_t = f_t * c_t_1 + i_t * g_t
        h_t = o_t * self.act.activate(c_t)
        return h_t, c_t

    def _step_batch(self, x_t, m_t, h_t_1, c_t_1, W, U, b):
        # (batch, in) (in, hidden * 4) + (hidden, in) (in, hidden * 4) +
        pre_calc = T.dot(x_t, W.T) + T.dot(h_t_1, U.T) + b
        i_t = self.in_gate.activate(ndarray_slice(pre_calc, 0, self.hidden_dim))
        f_t = self.forget_gate.activate(ndarray_slice(pre_calc, 1, self.hidden_dim))
        o_t = self.out_gate.activate(ndarray_slice(pre_calc, 2, self.hidden_dim))
        g_t = self.act.activate(ndarray_slice(pre_calc, 3, self.hidden_dim))
        c_t = f_t * c_t_1 + i_t * g_t
        h_t = o_t * self.act.activate(c_t)
        c_t = m_t[:, None] * c_t + (1. - m_t)[:, None] * c_t_1
        h_t = m_t[:, None] * h_t + (1. - m_t)[:, None] * h_t_1
        return h_t, c_t

    def forward_scan(self, x):
        h0 = shared_zero_matrix((self.hidden_dim,), 'h0_forward')
        c0 = shared_zero_matrix((self.hidden_dim,), 'c0_forward')
        hs, _ = theano.scan(fn=self._step,
                            sequences=x,
                            outputs_info=[h0, c0],
                            non_sequences=[self.W, self.U, self.b],
                            )
        return hs[0]

    def forward_scan_batch(self, x, mask, batch_size):
        h0 = shared_zero_matrix((batch_size, self.hidden_dim,), 'h0_forward')
        c0 = shared_zero_matrix((batch_size, self.hidden_dim,), 'c0_forward')
        hs, _ = theano.scan(fn=self._step_batch,
                            sequences=[T.transpose(x, (1, 0, 2)),
                                       T.transpose(mask, (1, 0))],
                            outputs_info=[h0, c0],
                            non_sequences=[self.W, self.U, self.b],
                            )
        return T.transpose(hs[0], (1, 0, 2))

    def forward_sequence(self, x):
        return self.forward_scan(x)

    def forward_sequence_batch(self, x, mask, batch_size):
        return self.forward_scan_batch(x, mask, batch_size)

    def forward(self, x):
        return get_pooling(self.forward_sequence(x), self.pooling)

    def forward_batch(self, x, mask, batch_size):
        return get_pooling_batch(self.forward_sequence_batch(x, mask, batch_size), mask, self.pooling)


class GRUEncoder(object):
    pass


class BiGRUEncoder(object):
    pass


class SGUEncoder(object):
    pass


class DSGUEncoder(object):
    pass


class BiRecurrentEncoder(RecurrentEncoder):
    def __init__(self, in_dim, hidden_dim, pooling, activation, prefix="",
                 initializer=default_initializer, dropout=0, verbose=True):
        super(BiRecurrentEncoder, self).__init__(in_dim, hidden_dim, pooling, activation, prefix,
                                                 initializer, dropout, verbose)
        if verbose:
            logger.debug('Building {}...'.format(self.__class__.__name__))
        self.out_dim = hidden_dim * 2
        # Forward Direction - Backward Direction
        # Feed-Forward Matrix (hidden, in)
        self.W_forward = self.W
        self.W_forward.name = prefix + "W_forward"
        self.W_backward = shared_rand_matrix((self.hidden_dim, self.in_dim), prefix + 'W_backward', initializer)
        # Bias Term (hidden,)
        self.b_forward = self.b
        self.b_forward.name = prefix + "b_forward"
        self.b_backward = shared_zero_matrix((self.hidden_dim,), prefix + 'b_backward')
        # Recurrent Matrix (hidden, hidden)
        self.U_forward = self.U
        self.U_forward.name = prefix + "U_forward"
        self.U_backward = shared_rand_matrix((self.hidden_dim, self.hidden_dim), prefix + 'U_backward', initializer)

        self.params = [self.W_forward, self.W_backward, self.U_forward, self.U_backward,
                       self.b_forward, self.b_backward]
        self.norm_params = [self.W_forward, self.W_backward, self.U_forward, self.U_backward]
        # L1, L2 Norm
        self.l1_norm = T.sum([T.sum(T.abs_(param)) for param in self.norm_params])
        self.l2_norm = T.sum([T.sum(param ** 2) for param in self.norm_params])

        if verbose:
            logger.debug('Architecture of {} built finished'.format(self.__class__.__name__))
            logger.debug('Input dimension:  %d' % self.in_dim)
            logger.debug('Hidden dimension: %d' % self.hidden_dim)
            logger.debug('Pooling methods:  %s' % self.pooling)
            logger.debug('Activation Func:  %s' % self.act.method)
            logger.debug('Dropout Rate:     %f' % self.dropout)

    def backward_scan(self, x):
        h0_backward = shared_zero_matrix(self.hidden_dim, 'h0_backward')
        h_backwards, _ = theano.scan(fn=self._step,
                                     sequences=x,
                                     outputs_info=[h0_backward],
                                     non_sequences=[self.W_backward, self.U_backward, self.b_backward],
                                     go_backwards=True,
                                     )
        return h_backwards[::-1]

    def backward_scan_batch(self, x, mask, batch_size):
        h0_backward = shared_zero_matrix((batch_size, self.hidden_dim), 'h0_backward')
        h_backwards, _ = theano.scan(fn=self._step_batch,
                                     sequences=[T.transpose(x, (1, 0, 2)),
                                                T.transpose(mask, (1, 0))],
                                     outputs_info=[h0_backward],
                                     non_sequences=[self.W_backward, self.U_backward, self.b_backward],
                                     go_backwards=True,
                                     )
        return T.transpose(h_backwards, (1, 0, 2))[:, :-1]

    def forward_sequence(self, x):
        return T.concatenate([self.forward_scan(x),
                              self.backward_scan(x),
                              ])

    def forward_sequence_batch(self, x, mask, batch_size):
        return T.concatenate([self.forward_scan_batch(x, mask, batch_size),
                              self.backward_scan_batch(x, mask, batch_size),
                              ])


class BiLSTMEncoder(LSTMEncoder):
    def __init__(self, in_dim, hidden_dim, pooling, activation, gates=("sigmoid", "sigmoid", "sigmoid"), prefix="",
                 initializer=default_initializer, dropout=0, verbose=True):
        if verbose:
            logger.debug('Building {}...'.format(self.__class__.__name__))
        super(BiLSTMEncoder, self).__init__(in_dim, hidden_dim, pooling, activation, gates, prefix,
                                            initializer, dropout, verbose)
        self.out_dim = hidden_dim * 2
        # Composition Function Weight -- Gates
        # W [in, forget, output, recurrent]
        self.W_forward, self.W_forward.name = self.W, prefix + "W_forward"
        self.W_backward = shared_rand_matrix((self.hidden_dim * 4, self.in_dim), prefix + 'W_backward', initializer)
        # U [in, forget, output, recurrent]

        self.U_forward, self.U_forward.name = self.U, prefix + "U_forward"
        self.U_backward = shared_rand_matrix((self.hidden_dim * 4, self.hidden_dim), prefix + 'U_backward', initializer)
        # b [in, forget, output, recurrent]
        self.b_forward, self.b_forward.name = self.b, prefix + "b_forward"
        self.b_backward = shared_zero_matrix((self.hidden_dim * 4,), prefix + 'b_backward')

        self.params = [self.W_forward, self.U_forward, self.b_forward,
                       self.W_backward, self.U_backward, self.b_backward]
        self.norm_params = [self.W_forward, self.U_forward, self.W_backward, self.U_backward]
        self.l1_norm = T.sum([T.sum(T.abs_(param)) for param in self.norm_params])
        self.l2_norm = T.sum([T.sum(param ** 2) for param in self.norm_params])

        if verbose:
            logger.debug('Architecture of {} built finished'.format(self.__class__.__name__))
            logger.debug('Input dimension:  %d' % self.in_dim)
            logger.debug('Hidden dimension: %d' % self.hidden_dim)
            logger.debug('Pooling methods:  %s' % self.pooling)
            logger.debug('Activation Func:  %s' % self.act.method)
            logger.debug('Input Gate:       %s' % self.in_gate.method)
            logger.debug('Forget Gate:      %s' % self.forget_gate.method)
            logger.debug('Output Gate:      %s' % self.out_gate.method)
            logger.debug('Activation Func:  %s' % self.act.method)
            logger.debug('Dropout Rate:     %f' % self.dropout)

    def backward_scan(self, x):
        h0_backward = shared_zero_matrix(self.hidden_dim, 'h0_backward')
        c0_backward = shared_zero_matrix(self.hidden_dim, 'c0_backward')
        h_backwards, _ = theano.scan(fn=self._step,
                                     sequences=x,
                                     outputs_info=[h0_backward, c0_backward],
                                     non_sequences=[self.W_backward, self.U_backward, self.b_backward],
                                     go_backwards=True,
                                     )
        return h_backwards[::-1]

    def backward_scan_batch(self, x, mask, batch_size):
        h0_backward = shared_zero_matrix((batch_size, self.hidden_dim), 'h0_backward')
        c0_backward = shared_zero_matrix((batch_size, self.hidden_dim), 'c0_backward')
        h_backwards, _ = theano.scan(fn=self._step_batch,
                                     sequences=[T.transpose(x, (1, 0, 2)),
                                                T.transpose(mask, (1, 0))],
                                     outputs_info=[h0_backward, c0_backward],
                                     non_sequences=[self.W_backward, self.U_backward, self.b_backward],
                                     go_backwards=True,
                                     )
        return T.transpose(h_backwards[0], (1, 0, 2))[:, ::-1]

    def forward_sequence(self, x):
        return T.concatenate([self.forward_scan(x),
                              self.backward_scan(x),
                              ])

    def forward_sequence_batch(self, x, mask, batch_size):
        return T.concatenate([self.forward_scan_batch(x, mask, batch_size),
                              self.backward_scan_batch(x, mask, batch_size),
                              ],
                             axis=2)


class RecurrentClassifier(object):
    def __init__(self, lookup_table, recurrent_encoder, in_dim, hidden_dim, num_label, pooling, activation,
                 batch_size=64, initializer=default_initializer, dropout=0, verbose=True):
        self.batch_size = batch_size
        word_index = T.imatrix()  # (batch, max_len)
        gold_truth = T.ivector()  # (batch, 1)
        rnn_encoder = recurrent_encoder(in_dim=in_dim, hidden_dim=hidden_dim, pooling=pooling, activation=activation,
                                        initializer=initializer, dropout=dropout, verbose=verbose)
        mask = (word_index > 0) * one_float32
        word_embedding = lookup_table.W[word_index]
        rnn_output = rnn_encoder.forward_batch(word_embedding, mask, batch_size)
        classifier = SoftmaxClassifier(num_in=rnn_encoder.out_dim, num_out=num_label, initializer=initializer)
        classifier_output = classifier.forward(rnn_output)
        loss = classifier.loss(rnn_output, gold_truth)
        params = lookup_table.params + classifier.params + rnn_encoder.params
        sgd_optimizer = AdaGradOptimizer(lr=0.95, norm_lim=16)
        except_norm_list = [param.name for param in lookup_table.params]
        updates = sgd_optimizer.get_update(loss, params, except_norm_list)

        self.train_x = shared_zero_matrix((batch_size, 1), dtype=np.int32)
        self.train_y = shared_zero_matrix(1, dtype=np.int32)
        self.dev_x = shared_zero_matrix((batch_size, 1), dtype=np.int32)
        self.test_x = shared_zero_matrix((batch_size, 1), dtype=np.int32)

        index = T.ivector()
        self.train_batch = theano.function(inputs=[index],
                                           outputs=[classifier_output, loss],
                                           updates=updates,
                                           givens={word_index: self.train_x[index],
                                                   gold_truth: self.train_y[index]}
                                           )
        self.get_norm = theano.function(inputs=[],
                                        outputs=[lookup_table.l2_norm, classifier.l2_norm])
        self.pred_train_batch = theano.function(inputs=[index],
                                                outputs=classifier_output,
                                                givens={word_index: self.train_x[index]}
                                                )
        self.pred_dev_batch = theano.function(inputs=[index],
                                              outputs=classifier_output,
                                              givens={word_index: self.dev_x[index]}
                                              )
        self.pred_test_batch = theano.function(inputs=[index],
                                               outputs=classifier_output,
                                               givens={word_index: self.test_x[index]}
                                               )

    def set_gpu_data(self, train, dev, test):
        self.train_x.set_value(train[0])
        self.train_y.set_value(train[1])
        self.dev_x.set_value(dev[0])
        self.test_x.set_value(test[0])

    def predict(self, x, predict_indexs, predict_function):
        num_batch = len(predict_indexs) / self.batch_size
        predict = list()
        for i in xrange(num_batch):
            indexs = predict_indexs[i * self.batch_size: (i + 1) * self.batch_size]
            predict.append(predict_function(indexs))
        return np.argmax(np.concatenate(predict), axis=1)[:len(x)]

    def train(self, train, dev, test):
        train_x, train_y = train
        dev_x, dev_y = dev
        test_x, test_y = test
        self.set_gpu_data(train, dev, test)
        train_index = align_batch_size(range(len(train_x)), self.batch_size)
        dev_index = align_batch_size(range(len(dev_x)), self.batch_size)
        test_index = align_batch_size(range(len(test_x)), self.batch_size)
        num_batch = len(train_index) / self.batch_size
        batch_list = range(num_batch)
        from sklearn.metrics import accuracy_score
        logger.info("epoch_num train_loss train_acc dev_acc test_acc")
        dev_result = list()
        test_result = list()
        for j in xrange(25):
            loss_list = list()
            batch_list = np.random.permutation(batch_list)
            for i in batch_list:
                indexs = train_index[i * self.batch_size: (i + 1) * self.batch_size]
                output, loss = self.train_batch(indexs)
                loss_list.append(loss)
            train_pred = self.predict(train_x, train_index, self.pred_train_batch)
            dev_pred = self.predict(dev_x, dev_index, self.pred_dev_batch)
            test_pred = self.predict(test_x, test_index, self.pred_test_batch)
            train_acc = accuracy_score(train_y, train_pred)
            dev_acc = accuracy_score(dev_y, dev_pred)
            test_acc = accuracy_score(test_y, test_pred)
            dev_result.append(dev_acc)
            test_result.append(test_acc)
            logger.info("epoch %d, loss: %f, train: %f, dev: %f, test: %f" % (j, float(np.mean(loss_list)),
                                                                              train_acc, dev_acc, test_acc))
        best_dev_index = np.argmax(dev_result)
        print "Best Dev:", dev_result[best_dev_index], "Test:", test_result[best_dev_index]


class RecurrentNormEncoder(object):
    def __init__(self, in_dim, hidden_dim, pooling, activation, prefix="",
                 initializer=default_initializer, dropout=0, verbose=True):
        if verbose:
            logger.debug('Building {}...'.format(self.__class__.__name__))
        self.in_dim = in_dim
        self.out_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.pooling = pooling
        self.dropout = dropout
        self.act = Activation(activation)
        # Composition Function Weight
        # Feed-Forward Matrix (hidden, in)
        self.W = shared_rand_matrix((8, 8), prefix + 'W_forward', initializer)
        # Bias Term (hidden)
        self.b = shared_zero_matrix((8, 8), prefix + 'b_forward')
        # Recurrent Matrix (hidden, hidden)
        self.U = shared_rand_matrix((8, 8), prefix + 'U_forward', initializer)

        self.params = [self.W, self.U, self.b]
        self.norm_params = [self.W, self.U]

        # L1, L2 Norm
        self.l1_norm = T.sum(T.abs_(self.W)) + T.sum(T.abs_(self.U))
        self.l2_norm = T.sum(self.W ** 2 + self.U ** 2)

        if verbose:
            logger.debug('Architecture of {} built finished'.format(self.__class__.__name__))
            logger.debug('Input dimension:  %d' % self.in_dim)
            logger.debug('Hidden dimension: %d' % self.hidden_dim)
            logger.debug('Pooling methods:  %s' % self.pooling)
            logger.debug('Activation Func:  %s' % self.act.method)
            logger.debug('Dropout Rate:     %f' % self.dropout)

    def _step(self, x_t, h_t_1, W, U, b):
        """
        step function of forward
        :param x_t:   (in, )
        :param h_t_1: (hidden, )
        :param W:     (hidden, in)
        :param U:     (hidden, hidden)
        :param b:     (hidden, )
        :return:      (hidden)
        """
        # (hidden, in) (in, ) + (hidden, hidden) (hidden, ) + (hidden, ) -> hidden
        h_t = self.act.activate(T.dot(W, x_t) + T.dot(U, h_t_1) + b)
        return h_t

    def _step_batch(self, x_t, mask, h_t_1, W, U, b):
        """
        step function of forward in batch version
        :param x_t:   (batch, in)
        :param mask:  (batch, )
        :param h_t_1: (batch, hidden)
        :param W:     (hidden, in)
        :param U:     (hidden, hidden)
        :param b:     (hidden)
        :return:      (batch, hidden)
        """
        # (batch, in) (in, hidden) -> (batch, hidden)
        h_t_1 = T.reshape(h_t_1, (h_t_1.shape[0], 8, 8))
        x_t = T.reshape(x_t, (x_t.shape[0], 8, 8))
        x_t = x_t / x_t.norm(2, axis=1)[:, None, :]
        h_t = self.act.activate(T.dot(x_t, W.T) + T.dot(h_t_1, U.T) + b)
        h_t = h_t / h_t.norm(2, axis=1)[:, None, :]
        h_t_1 = T.reshape(h_t_1, (h_t_1.shape[0], 64))
        h_t = T.reshape(h_t, (h_t.shape[0], 64))
        # (batch, hidden) * (batch, None) + (batch, hidden) * (batch, None) -> (batch, hidden)
        return h_t * mask[:, None] + h_t_1 * (1 - mask[:, None])

    def forward_sequence(self, x):
        h0 = shared_zero_matrix((self.hidden_dim,), 'h0')
        hs, _ = theano.scan(fn=self._step,
                            sequences=x,
                            outputs_info=[h0],
                            non_sequences=[self.W, self.U, self.b],
                            )
        return hs

    def forward_sequence_batch(self, x, mask, batch_size):
        """
        :param x: (batch, max_len, dim)
        :param mask:  (batch, max_len)
        :param batch_size:
        """
        h0 = shared_zero_matrix((batch_size, self.hidden_dim), 'h0')
        hs, _ = theano.scan(fn=self._step_batch,
                            sequences=[T.transpose(x, (1, 0, 2)),  # (batch, max_len, dim) -> (max_len, batch, dim)
                                       T.transpose(mask, (1, 0))],     # (batch, max_len) -> (max_len, batch)
                            outputs_info=[h0],
                            non_sequences=[self.W, self.U, self.b],
                            )
        # (max_len, batch, dim) -> (batch, max_len, dim)
        return T.transpose(hs, (1, 0, 2))

    def forward(self, x):
        """
        :param x: (len, dim)
        """
        # Use Pooling to reduce into a fixed-length representation
        return get_pooling(self.forward_sequence(x), self.pooling)

    def forward_batch(self, x, mask, batch_size):
        """
        :param x: (batch, max_len, dim)
        :param mask:  (batch, max_len)
        :param batch_size:
        """
        # Use Pooling to reduce into a fixed-length representation
        # (max_len, batch, dim) -> (batch, max_len, dim) -> (batch, dim)
        hidden = self.forward_sequence_batch(x, mask, batch_size)
        return get_pooling_batch(hidden, mask, self.pooling)
