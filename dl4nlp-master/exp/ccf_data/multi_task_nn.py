import logging
import sys

import numpy as np
import theano
import theano.tensor as T

sys.path.append('../../')
from src import default_initializer, BIG_INT
from src.layers import MultiHiddenLayer, HighwayLayer
from src.dropout import set_dropout_on
from src.utils import shared_zero_matrix, align_batch_size
from src.optimizer import AdaGradOptimizer
from src.classifier import SoftmaxClassifier
from src.metrics import accuracy_score
from src.recurrent import get_pooling_batch


__author__ = "roger"
logger = logging.getLogger(__name__)


class MultiTaskClassifier(object):
    """
    Task Number is decided by labels
    """
    def __init__(self, lookup_table, in_dim, hidden_dims, labels_nums, activation, highway=False,
                 batch_size=64, initializer=default_initializer, optimizer=None, dropout=0, verbose=True):
        self.batch_size = batch_size
        self.num_task = len(labels_nums)
        word_index = T.imatrix()  # (batch, max_len)
        gold_truth = T.ivector()  # (batch, 1)
        mask = (word_index > 0) * T.constant(1, dtype=theano.config.floatX)
        word_embedding = lookup_table.W[word_index]
        # max sum averaging
        hidden = get_pooling_batch(word_embedding, mask, "max")
        if len(hidden_dims) == 0 or hidden_dims[0] == 0:
            nn_output = hidden
            nn_output_dim = in_dim
        elif highway:
            encoder = HighwayLayer(in_dim=in_dim, activation=activation, initializer=initializer,
                                   dropout=dropout, verbose=verbose)
            nn_output = encoder.forward_batch(hidden)
            nn_output_dim = encoder.out_dim
        else:
            encoder = MultiHiddenLayer(in_dim=in_dim, hidden_dims=hidden_dims, activation=activation,
                                       initializer=initializer, dropout=dropout, verbose=verbose)
            nn_output = encoder.forward_batch(hidden)
            nn_output_dim = encoder.out_dim
        if optimizer is None:
            sgd_optimizer = AdaGradOptimizer(lr=0.95, norm_lim=16)
        else:
            sgd_optimizer = optimizer
        self.train_x = shared_zero_matrix((batch_size, 1), dtype=np.int32)
        self.train_y = shared_zero_matrix((1, 1), dtype=np.int32)
        self.dev_x = shared_zero_matrix((batch_size, 1), dtype=np.int32)
        self.test_x = shared_zero_matrix((batch_size, 1), dtype=np.int32)
        self.train_batch_list = list()
        self.pred_train_batch_list = list()
        self.pred_dev_batch_list = list()
        self.pred_test_batch_list = list()
        index = T.ivector()
        classifier_list = list()
        classifier_output_list = list()
        classifier_loss_list = list()
        classifier_param_list = list()
        classifier_updates_list = list()
        for i in xrange(len(labels_nums)):
            classifier = SoftmaxClassifier(num_in=nn_output_dim, num_out=labels_nums[i], initializer=initializer)
            classifier_list.append(classifier)
            classifier_output_list.append(classifier_list[i].forward(nn_output))
            classifier_loss_list.append(classifier_list[i].loss(nn_output, gold_truth))
            if len(hidden_dims) == 0 or hidden_dims[0] == 0:
                classifier_param_list.append(lookup_table.params + classifier.params)
            else:
                classifier_param_list.append(lookup_table.params + classifier.params + encoder.params)
            except_norm_list = [param.name for param in lookup_table.params]
            classifier_updates_list.append(sgd_optimizer.get_update(classifier_loss_list[i],
                                                                    classifier_param_list[i],
                                                                    except_norm_list))
            train_batch = theano.function(inputs=[index],
                                          outputs=[classifier_output_list[i], classifier_loss_list[i]],
                                          updates=classifier_updates_list[i],
                                          givens={word_index: self.train_x[index],
                                                  gold_truth: self.train_y[index, i]}
                                          )
            self.train_batch_list.append(train_batch)
            pred_train_batch = theano.function(inputs=[index],
                                               outputs=classifier_output_list[i],
                                               givens={word_index: self.train_x[index]}
                                               )
            self.pred_train_batch_list.append(pred_train_batch)
            pred_dev_batch = theano.function(inputs=[index],
                                             outputs=classifier_output_list[i],
                                             givens={word_index: self.dev_x[index]}
                                             )
            self.pred_dev_batch_list.append(pred_dev_batch)
            pred_test_batch = theano.function(inputs=[index],
                                              outputs=classifier_output_list[i],
                                              givens={word_index: self.test_x[index]}
                                              )
            self.pred_test_batch_list.append(pred_test_batch)

    def set_gpu_data(self, train, dev, test=None):
        self.train_x.set_value(train[0])
        self.train_y.set_value(train[1])
        self.dev_x.set_value(dev[0])
        if test is not None:
            self.test_x.set_value(test[0])

    def predict_model_data(self, predict_indexs, predict_function):
        num_batch = len(predict_indexs) / self.batch_size
        predict = list()
        for i in xrange(num_batch):
            indexs = predict_indexs[i * self.batch_size: (i + 1) * self.batch_size]
            predict.append(predict_function(indexs))
        return np.argmax(np.concatenate(predict), axis=1)

    def predict(self, x):
        self.test_x.set_value(x)
        predict_indexs = align_batch_size(range(len(x)), self.batch_size)
        num_batch = len(predict_indexs) / self.batch_size
        predict = [list() for i in xrange(self.num_task)]
        for i in xrange(num_batch):
            indexs = predict_indexs[i * self.batch_size: (i + 1) * self.batch_size]
            for task_index in xrange(self.num_task):
                predict[task_index].append(self.pred_test_batch_list[task_index](indexs))
        predict = [np.concatenate(predict[task_index]) for task_index in xrange(self.num_task)]
        self.test_x.set_value(None)
        return predict

    @staticmethod
    def get_train_valid_index(train_labels, invalid_label=-1):
        num_instance, num_label = train_labels.shape
        train_valid_index_list = list()
        for i_label in xrange(num_label):
            index_list = list()
            for index in xrange(num_instance):
                if train_labels[index][i_label] != invalid_label:
                    index_list.append(index)
            train_valid_index_list.append(index_list)
        return train_valid_index_list

    def train(self, train, dev, test=None, iter_num=25):
        train_x, train_y = train
        dev_x, dev_y = dev
        test_x, test_y = None, None
        task_num = train_y.shape[1]

        # Each task have different valid instances
        # valid_train_indexs element: Distinct Valid instances
        # train_indexs       element: actual train indexs
        valid_train_indexs = self.get_train_valid_index(train_y)
        train_indexs = self.get_train_valid_index(train_y)
        train_indexs = [align_batch_size(indexs, self.batch_size) for indexs in train_indexs]
        valid_dev_indexs = self.get_train_valid_index(dev_y)
        dev_indexs = self.get_train_valid_index(dev_y)
        dev_indexs = [align_batch_size(indexs, self.batch_size) for indexs in dev_indexs]

        # Consider test data case
        if test is not None:
            test_x, test_y = test
            valid_test_indexs = self.get_train_valid_index(test_y)
            test_indexs = self.get_train_valid_index(test_y)
            test_indexs = [align_batch_size(indexs, self.batch_size) for indexs in test_indexs]
            self.set_gpu_data(train, dev, test)
        else:
            self.set_gpu_data(train, dev)

        # train_list each element: (train_batch_index, task_index)
        train_batch_task_list = list()
        batch_task_count = [0, 0, 0]
        for task_index in xrange(task_num):
            for task_batch_index in xrange(len(train_indexs[task_index]) / self.batch_size):
                batch_task_count[task_index] += 1
                train_batch_task_list.append((task_batch_index, task_index))
        task_info = "\n".join(["Task %d: Batch %d" % (task_index, batch_task_count[task_index])
                               for task_index in xrange(task_num)])
        logger.info(task_info)
        logger.info("epoch_num train_loss train_acc dev_acc test_acc")
        over_task_dev_acc = [list() for i in xrange(task_num)]
        predict_result = list()
        for j in xrange(iter_num):
            losses_list = list()
            for task_index in xrange(task_num):
                losses_list.append(list())
            np.random.shuffle(train_batch_task_list)
            set_dropout_on(True)
            for batch_index, task_index in train_batch_task_list:
                indexs = train_indexs[task_index][batch_index * self.batch_size: (batch_index + 1) * self.batch_size]
                output, loss = self.train_batch_list[task_index](indexs)
                losses_list[task_index].append(loss)
            logger.info("epoch %d" % j)
            set_dropout_on(False)
            for task_index in xrange(task_num):
                train_true = train_y[valid_train_indexs[task_index], task_index]
                dev_true = dev_y[valid_dev_indexs[task_index], task_index]
                # Align Pred ( means duplicate pred in it
                train_pred = self.predict_model_data(train_indexs[task_index], self.pred_train_batch_list[task_index])
                dev_pred = self.predict_model_data(dev_indexs[task_index], self.pred_dev_batch_list[task_index])
                train_pred = train_pred[:len(train_true)]
                dev_pred = dev_pred[:len(dev_true)]
                train_acc = accuracy_score(train_true, train_pred)
                dev_acc = accuracy_score(dev_true, dev_pred)
                over_task_dev_acc[task_index].append(dev_acc)
                output_string = "Task %d: Error: %s Acc: %f %f " % (task_index, np.mean(losses_list[task_index]),
                                                                    train_acc, dev_acc)
                if test is not None:
                    test_true = test_y[valid_test_indexs[task_index], task_index]
                    test_pred = self.predict_model_data(test_indexs[task_index], self.pred_test_batch_list[task_index])
                    test_pred = test_pred[:len(test_true)]
                    test_acc = accuracy_score(test_true, test_pred)
                    output_string += "%d " % test_acc
                logger.info(output_string)
            if to_predict is not None:
                predict_result.append(self.predict(to_predict))
        max_indexs = [np.argmax(over_task_dev_acc[task_index]) for task_index in xrange(task_num)]
        max_dev_score = np.mean([over_task_dev_acc[task_index][max_index]
                                 for max_index, task_index in zip(max_indexs, xrange(task_num))])
        if to_predict is not None:
            max_predict_result = [predict_result[max_indexs[task_index]][task_index]
                                  for task_index in xrange(task_num)]
        return max_dev_score


def get_pooling_batch_word(hs, mask, pooling_method):
    """
    :param hs:   (batch, len, dim)
    :param mask: (batch, len)
    :param pooling_method:
    :return:
    """
    if pooling_method == 'max':
        add_v = ((1 - mask) * -BIG_INT)[:, :, :, None]
        return T.max(hs + add_v, axis=2)
    elif pooling_method == 'averaging':
        return T.sum(hs * mask[:, :, None], axis=2) / T.sum(mask, axis=2)[:, None]
    elif pooling_method == 'sum':
        return T.sum(hs * mask[:, :, None], axis=2)
    elif pooling_method == 'final':
        return hs[:, :, -1, :]
    else:
        raise NotImplementedError('Not implemented pooling method: {}'.format(pooling_method))


class MultiTaskHierarchicalClassifier(object):
    """
    Task Number is decided by labels
    """
    def __init__(self, lookup_table, in_dim, hidden_dims, labels_nums, activation, highway=False,
                 batch_size=64, initializer=default_initializer, optimizer=None, dropout=0, verbose=True):
        self.batch_size = batch_size
        self.num_task = len(labels_nums)
        word_index = T.itensor3()  # (batch, max_len)
        gold_truth = T.ivector()  # (batch, 1)

        mask_query = (word_index > 0) * T.constant(1, dtype=theano.config.floatX)
        mask_user = (T.sum(word_index, axis=2) > 0) * T.constant(1, dtype=theano.config.floatX)
        word_embedding = lookup_table.W[word_index]
        # max sum averaging
        hidden = get_pooling_batch_word(word_embedding, mask_query, "averaging")
        hidden = get_pooling_batch(hidden, mask_user, "averaging")
        # hidden = T.mean(hidden, axis=1)
        if len(hidden_dims) == 0 or hidden_dims[0] == 0:
            nn_output = hidden
            nn_output_dim = in_dim
        elif highway:
            encoder = HighwayLayer(in_dim=in_dim, activation=activation, initializer=initializer,
                                   dropout=dropout, verbose=verbose)
            nn_output = encoder.forward_batch(hidden)
            nn_output_dim = encoder.out_dim
        else:
            encoder = MultiHiddenLayer(in_dim=in_dim, hidden_dims=hidden_dims, activation=activation,
                                       initializer=initializer, dropout=dropout, verbose=verbose)
            nn_output = encoder.forward_batch(hidden)
            nn_output_dim = encoder.out_dim
        if optimizer is None:
            sgd_optimizer = AdaGradOptimizer(lr=0.95, norm_lim=16)
        else:
            sgd_optimizer = optimizer
        self.train_x = shared_zero_matrix((batch_size, 1, 1), dtype=np.int32)
        self.train_y = shared_zero_matrix((1, 1), dtype=np.int32)
        self.dev_x = shared_zero_matrix((batch_size, 1, 1), dtype=np.int32)
        self.test_x = shared_zero_matrix((batch_size, 1, 1), dtype=np.int32)
        self.train_batch_list = list()
        self.pred_train_batch_list = list()
        self.pred_dev_batch_list = list()
        self.pred_test_batch_list = list()
        self.get_y_list = list()
        index = T.ivector()
        classifier_list = list()
        classifier_output_list = list()
        classifier_loss_list = list()
        classifier_param_list = list()
        classifier_updates_list = list()
        for i in xrange(len(labels_nums)):
            classifier = SoftmaxClassifier(num_in=nn_output_dim, num_out=labels_nums[i], initializer=initializer)
            classifier_list.append(classifier)
            classifier_output_list.append(classifier_list[i].forward(nn_output))
            classifier_loss_list.append(classifier_list[i].loss(nn_output, gold_truth))
            if len(hidden_dims) == 0 or hidden_dims[0] == 0:
                classifier_param_list.append(lookup_table.params + classifier.params)
            else:
                classifier_param_list.append(lookup_table.params + classifier.params + encoder.params)
            except_norm_list = [param.name for param in lookup_table.params]
            classifier_updates_list.append(sgd_optimizer.get_update(classifier_loss_list[i],
                                                                    classifier_param_list[i],
                                                                    except_norm_list))
            train_batch = theano.function(inputs=[index],
                                          outputs=[classifier_output_list[i], classifier_loss_list[i]],
                                          updates=classifier_updates_list[i],
                                          givens={word_index: self.train_x[index],
                                                  gold_truth: self.train_y[index, i]}
                                          )
            self.train_batch_list.append(train_batch)
            pred_train_batch = theano.function(inputs=[index],
                                               outputs=classifier_output_list[i],
                                               givens={word_index: self.train_x[index]}
                                               )
            self.pred_train_batch_list.append(pred_train_batch)
            pred_dev_batch = theano.function(inputs=[index],
                                             outputs=classifier_output_list[i],
                                             givens={word_index: self.dev_x[index]}
                                             )
            self.pred_dev_batch_list.append(pred_dev_batch)
            pred_test_batch = theano.function(inputs=[index],
                                              outputs=classifier_output_list[i],
                                              givens={word_index: self.test_x[index]}
                                              )
            self.pred_test_batch_list.append(pred_test_batch)
            self.get_y_list.append(theano.function(inputs=[index], outputs=self.train_y[index, i]))

    def set_gpu_data(self, train, dev, test=None):
        self.train_x.set_value(train[0])
        self.train_y.set_value(train[1])
        self.dev_x.set_value(dev[0])
        self.test_x.set_value(test[0])

    def predict_model_data(self, predict_indexs, predict_function):
        num_batch = len(predict_indexs) / self.batch_size
        predict = list()
        for i in xrange(num_batch):
            indexs = predict_indexs[i * self.batch_size: (i + 1) * self.batch_size]
            predict.append(predict_function(indexs))
        return np.argmax(np.concatenate(predict), axis=1)

    def predict(self, x):
        self.test_x.set_value(x)
        predict_indexs = align_batch_size(range(len(x)), self.batch_size)
        num_batch = len(predict_indexs) / self.batch_size
        predict = [list() for i in xrange(self.num_task)]
        for i in xrange(num_batch):
            indexs = predict_indexs[i * self.batch_size: (i + 1) * self.batch_size]
            for task_index in xrange(self.num_task):
                predict[task_index].append(self.pred_test_batch_list[task_index](indexs))
        predict = [np.concatenate(predict[task_index]) for task_index in xrange(self.num_task)]
        self.test_x.set_value(None)
        return predict

    @staticmethod
    def get_train_valid_index(train_labels, invalid_label=-1):
        num_instance, num_label = train_labels.shape
        train_valid_index_list = list()
        for i_label in xrange(num_label):
            index_list = list()
            for index in xrange(num_instance):
                if train_labels[index][i_label] != invalid_label:
                    index_list.append(index)
            train_valid_index_list.append(index_list)
        return train_valid_index_list

    def train(self, train, test, dev=None, iter_num=25):

        task_num = train[1].shape[1]

        if dev is None:
            train_part = int(round(train[0].shape[0] * 0.9))
            train_x, dev_x = train[0][:train_part], train[0][train_part:]
            train_y, dev_y = train[1][:train_part], train[1][train_part:]
        else:
            train_x, train_y = train
            dev_x, dev_y = dev
        test_x, test_y = test

        # Each task have different valid instances
        # valid_train_indexs element: Distinct Valid instances
        # train_indexs       element: actual train indexs
        valid_train_indexs = self.get_train_valid_index(train_y)
        train_indexs = self.get_train_valid_index(train_y)
        train_indexs = [align_batch_size(indexs, self.batch_size) for indexs in train_indexs]

        valid_dev_indexs = self.get_train_valid_index(dev_y)
        dev_indexs = self.get_train_valid_index(dev_y)
        dev_indexs = [align_batch_size(indexs, self.batch_size) for indexs in dev_indexs]

        valid_test_indexs = self.get_train_valid_index(test_y)
        test_indexs = self.get_train_valid_index(test_y)
        test_indexs = [align_batch_size(indexs, self.batch_size) for indexs in test_indexs]

        self.set_gpu_data([train_x, train_y], [dev_x], [test_x])

        # train_list each element: (train_batch_index, task_index)
        train_batch_task_list = list()
        batch_task_count = [0, 0, 0]
        for task_index in xrange(task_num):
            for task_batch_index in xrange(len(train_indexs[task_index]) / self.batch_size):
                batch_task_count[task_index] += 1
                train_batch_task_list.append((task_batch_index, task_index))
        task_info = "\n".join(["Task %d: Batch %d" % (task_index, batch_task_count[task_index])
                               for task_index in xrange(task_num)])
        logger.info(task_info)
        logger.info("epoch_num train_loss train_acc dev_acc test_acc")
        over_task_dev_acc = [list() for i in xrange(task_num)]
        over_task_test_acc = [list() for i in xrange(task_num)]
        for j in xrange(iter_num):
            losses_list = list()
            for task_index in xrange(task_num):
                losses_list.append(list())
            np.random.shuffle(train_batch_task_list)
            set_dropout_on(True)
            for batch_index, task_index in train_batch_task_list:
                indexs = train_indexs[task_index][batch_index * self.batch_size: (batch_index + 1) * self.batch_size]
                output, loss = self.train_batch_list[task_index](indexs)
                losses_list[task_index].append(loss)
            logger.info("epoch %d" % j)
            set_dropout_on(False)
            for task_index in xrange(task_num):
                train_true = train_y[valid_train_indexs[task_index], task_index]
                dev_true = dev_y[valid_dev_indexs[task_index], task_index]
                test_true = test_y[valid_test_indexs[task_index], task_index]
                # Align Pred ( means duplicate pred in it
                train_pred = self.predict_model_data(train_indexs[task_index], self.pred_train_batch_list[task_index])
                dev_pred = self.predict_model_data(dev_indexs[task_index], self.pred_dev_batch_list[task_index])
                test_pred = self.predict_model_data(test_indexs[task_index], self.pred_test_batch_list[task_index])

                train_acc = accuracy_score(train_true, train_pred[:len(train_true)])
                dev_acc = accuracy_score(dev_true, dev_pred[:len(dev_true)])
                test_acc = accuracy_score(test_true, test_pred[:len(test_true)])

                over_task_dev_acc[task_index].append(dev_acc)
                over_task_test_acc[task_index].append(test_acc)

                output_string = "Task %d: Error: %s Acc: %f %f %f" % (task_index, np.mean(losses_list[task_index]),
                                                                      train_acc, dev_acc, test_acc)
                logger.info(output_string)
        max_indexs = [np.argmax(over_task_dev_acc[task_index]) for task_index in xrange(task_num)]
        max_test_score = np.mean([over_task_test_acc[task_index][max_index]
                                 for max_index, task_index in zip(max_indexs, xrange(task_num))])
        return max_test_score
