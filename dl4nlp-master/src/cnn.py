from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T

from classifier import SoftmaxClassifier
from optimizer import AdaGrad
from src import default_initializer
from src.utils import progress_bar_str, get_train_sequence

__author__ = 'roger'


class ShallowCNNClassifier(object):
    def __init__(self, embedding, n_out, initializer=default_initializer, weight_l2=0.01, dropout=0, verbose=True):
        self.embedding = embedding
        self.verbose = verbose
        self.n_out = n_out
        self.weight_l2 = weight_l2
        self.dropout = dropout
        # Define Symbol Operation
        self.index = T.imatrix()
        self.truth = T.iscalar()
        # vectors, _ = theano.scan(lambda x: self.embedding[x[0]], sequences=self.index)
        vectors = self.embedding[self.index[:, 0]]
        # Define Layer
        self.average_hidden = T.mean(vectors, axis=0)
        self.max_hidden = T.max(vectors, axis=0)
        self.min_hidden = T.min(vectors, axis=0)
        self.hidden = T.concatenate([self.average_hidden, self.max_hidden, self.min_hidden])
        self.hidden_norm = self.hidden / T.sqrt(T.sum(self.hidden ** 2))
        self.classifier = SoftmaxClassifier(self.hidden_norm, self.embedding.dim * 3,
                                            n_out, initializer=initializer)
        self.params = self.embedding.params + self.classifier.params
        self.neg_log_likelihood = -T.log(self.classifier.output)[0, self.truth]
        self.loss_l2 = self.classifier.loss_l2 * self.weight_l2
        self.loss = self.neg_log_likelihood + self.loss_l2
        self.grad = T.grad(self.loss, self.params)

        grads = T.grad(self.loss, self.params)
        self.updates = OrderedDict()
        self.grad = {}
        for param, grad in zip(self.params, grads):
            g = theano.shared(np.asarray(np.zeros_like(param.get_value()), dtype=theano.config.floatX))
            self.grad[param] = g
            self.updates[g] = g + grad

        self.compute_result_grad = theano.function(
            inputs=[self.index, self.truth],
            outputs=[self.loss, self.classifier.output],
            updates=self.updates,
            allow_input_downcast=True
        )
        self.cost = theano.function(
            inputs=[self.index, self.truth],
            outputs=[self.neg_log_likelihood, self.loss_l2],
            allow_input_downcast=True
        )
        self.output = theano.function(
            inputs=[self.index],
            outputs=self.classifier.output
        )
        self.pred = theano.function(
            inputs=[self.index],
            outputs=self.classifier.pred
        )

    def update_param(self, grads, learn_rate=1.0):
        """
        Update param in Shallow CNN
        :param grads: [np.ndarray]. List of numpy.ndarray to update the model parameters.
        :param learn_rate: scalar. Learning rate.
        :return:
        """
        for param, grad in zip(self.params, grads):
            p = param.get_value(borrow=True)
            param.set_value(p - learn_rate * grad, borrow=True)

    def predict(self, x):
        return self.pred(x)

    def test(self, test_x, test_y):
        preds = []
        for j in range(len(test_x)):
            if self.verbose:
                print progress_bar_str(j + 1, len(test_x)) + "\r",
            pred = self.predict(test_x[j])
            preds.append(pred)
        acc = float(sum([1 if p == t else 0 for p, t in zip(preds, test_y)])) / len(test_x)
        if self.verbose:
            print
        return acc, preds

    def fit(self, train, dev, test):
        batch_size = 50
        num_epoch = 25
        trainX, trainY = train
        devX, devY = dev
        testX, testY = test
        train_index = get_train_sequence(trainX, batch_size)
        num_batch = len(train_index) / batch_size

        print "Start Training ..."
        for i in range(num_epoch):
            print "epoch ", i + 1
            loss = []
            pred = []
            optimizer = AdaGrad(self.params, lr=0.95)
            for j in range(num_batch):
                if self.verbose:
                    print progress_bar_str(j, num_batch) + "\r",
                for k in range(batch_size):
                    index = j * batch_size + k
                    trainx = trainX[train_index[index]]
                    trainy = trainY[train_index[index]]
                    result = self.compute_result_grad(trainx, trainy)
                    loss.append(result[0])
                    pred.append(result[1])
                for _, grad in self.grad.iteritems():
                    grad.set_value(grad.get_value() / batch_size)
                optimizer.iterate(self.grad)
            train_acc, train_pred = self.test(trainX, trainY)
            dev_acc, dev_pred = self.test(devX, devY)
            test_acc, test_pred = self.test(testX, testY)
            print "Loss", np.mean(loss, axis=0)
            print "Train ACC %f, Dev ACC %f, Test ACC %f" % (train_acc, dev_acc, test_acc)
