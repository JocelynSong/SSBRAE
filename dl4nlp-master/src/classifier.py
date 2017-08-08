import theano.tensor as T
import numpy as np
from src import default_initializer
from src.utils import shared_rand_matrix, shared_zero_matrix

__author__ = 'roger'


class LogisticClassifier(object):
    def __init__(self, num_in, initializer=default_initializer):
        self.W = shared_rand_matrix(shape=(num_in, 1), name="logistic_W", initializer=initializer)
        self.b = shared_zero_matrix(np.asarray([0]), name='logistic_b')
        self.params = [self.W, self.b]

        self.l1_norm = T.sum(T.abs_(self.W))
        self.l2_norm = T.sum(self.W ** 2)

    def forward(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def forward_batch(self, input):
        self.forward(input)

    def loss(self, input, truth):
        """
        negative log likelihood loss function
        :param input:
        :param truth: n_examples * label (0 or 1)
        :return:
        """
        predict = self.forward(input)
        return - T.mean(truth * T.log(predict) + (1 - truth) * T.log(1 - predict))


class SoftmaxClassifier(object):
    def __init__(self, num_in, num_out, initializer=default_initializer):
        self.num_in = num_in
        self.num_out = num_out

        self.W = shared_rand_matrix(shape=(num_in, num_out), name="softmax_W", initializer=initializer)
        self.b = shared_zero_matrix((num_out, ), 'softmax_b')
        self.params = [self.W, self.b]
        self.l1_norm = T.sum(T.abs_(self.W))
        self.l2_norm = T.sum(self.W ** 2)

    def forward(self, input):
        return T.nnet.softmax(T.dot(input, self.W) + self.b)

    def forward_batch(self, input):
        return self.forward(input)

    def loss(self, input, truth):
        """
        negative log likelihood loss function
        :param input
        :param truth: n_examples * label
        :return:
        """
        return -T.mean(T.log(self.forward(input))[T.arange(truth.shape[0]), truth])
