# coding=utf-8
import logging
import sys
import numpy as np
from __init__ import default_initializer, OOV_KEY

__author__ = 'roger'


sys.setrecursionlimit(10000)
logger = logging.getLogger(__name__)


# Load and Save

def save_random_state(filename):
    import cPickle
    with file(filename, 'wb') as fout:
        cPickle.dump(np.random.get_state(), fout)
    logger.info("Save Random State to %s" % filename)


def load_random_state(filename):
    import cPickle
    with file(filename, 'rb') as fin:
        np.random.set_state(cPickle.load(fin))
    logger.info("Load Random State from %s" % filename)


def save_dev_test_loss(filename, dev_losses, test_losses):
    import cPickle
    with file(filename, 'wb') as fout:
        cPickle.dump(dev_losses, fout)
        cPickle.dump(test_losses, fout)


def load_dev_test_loss(filename):
    import cPickle
    with file(filename, 'rb') as fin:
        dev_losses = cPickle.load(fin)
        test_losses = cPickle.load(fin)
    return dev_losses, test_losses


def load_model(filename):
    import cPickle
    with file(filename, 'rb') as fin:
        model = cPickle.load(fin)
    return model


def save_model(filename, model):
    import cPickle
    with file(filename, 'wb') as out:
        cPickle.dump(model, out)


# Theano Shared Variable
def shared_rand_matrix(shape, name=None, initializer=default_initializer):
    import theano
    matrix = initializer.generate(shape=shape)
    return theano.shared(value=np.asarray(matrix, dtype=theano.config.floatX), name=name)


def shared_matrix(w, name=None, dtype=None,):
    import theano
    if dtype is None:
        dtype = theano.config.floatX
    return theano.shared(np.asarray(w, dtype=dtype), name=name)


def shared_zero_matrix(shape, name=None, dtype=None):
    import theano
    if dtype is None:
        dtype = theano.config.floatX
    return theano.shared((np.zeros(shape, dtype=dtype)), name=name)


# Variable Operation
def as_floatx(variable):
    import theano
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)
    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)


def ndarray_slice(x, n, dim):
    if x.ndim == 1:
        return x[n * dim:(n + 1) * dim]
    if x.ndim == 2:
        return x[:, n * dim:(n + 1) * dim]
    if x.ndim == 3:
        return x[:, :, n * dim:(n + 1) * dim]
    raise ValueError('Invalid slice dims!')


def array2str(array, space=' '):
    return space.join(["%.6f" % b for b in array])


# Train Operation
def progress_bar_str(curr, final):
    pre = int(float(curr) / final * 50)
    remain = 50 - pre
    progress_bar = "[%s>%s] %d/%d" % ('=' * pre, ' ' * remain, curr, final)
    return progress_bar


def align_batch_size(train_index, batch_size):
    """
    对训练数据根据Batch大小对齐，少则随机抽取实例加入在末尾
    :param train_index: 训练顺序
    :param batch_size: Batch大小
    :return:
    """
    if len(train_index) % batch_size == 0:
        return train_index
    else:
        raw_len = len(train_index)
        remain = batch_size - len(train_index) % batch_size
        for i in range(remain):
            ran_i = np.random.randint(0, raw_len - 1)
            train_index.append(train_index[ran_i])
        return train_index


def get_train_sequence(train_x, batch_size):
    """
    依据Batch大小产生训练顺序
    :param train_x:
    :param batch_size:
    :return:
    """
    train_index = range(len(train_x))
    train_index = align_batch_size(train_index, batch_size)
    np.random.shuffle(train_index)
    return train_index


def read_file(filename, word_dict, split_symbol=" |||| ", low_case=False, add_unknown_word=False, encoding="utf8"):
    x = list()
    y = list()
    import codecs
    instances = codecs.open(filename, 'r', encoding=encoding).readlines()
    instances = [(int(line.split(split_symbol)[0]), line.split(split_symbol)[1].strip()) for line in instances]
    for instance in instances:
        label, sen = instance
        token = list()
        for word in sen.split():
            if low_case:
                word.lower()
            if word not in word_dict:
                if add_unknown_word:
                    word_dict[word] = len(word_dict) + 1
                else:
                    word = OOV_KEY
            token.append(word_dict[word])
        y.append(label)
        x.append(token)
    len_list = [len(tokens) for tokens in x]
    max_len = np.max(len_list)
    for instance, length in zip(x, len_list):
        for j in xrange(max_len - length):
            instance.append(0)
    return np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)


def read_sst(train_file, dev_file, test_file, split_symbol=" |||| ", low_case=False):
    word_dict = dict()

    train_x, train_y = read_file(train_file, word_dict=word_dict, split_symbol=split_symbol, low_case=low_case)
    dev_x, dev_y = read_file(dev_file, word_dict=word_dict, split_symbol=split_symbol, low_case=low_case)
    test_x, test_y = read_file(test_file, word_dict=word_dict, split_symbol=split_symbol, low_case=low_case)
    return [train_x, train_y], [dev_x, dev_y], [test_x, test_y], word_dict


def pre_logger(log_file_name, file_handler_level=logging.DEBUG, screen_handler_level=logging.INFO):
    # Logging configuration
    # Set the basic configuration of the logging system
    log_formatter = logging.Formatter(fmt='%(asctime)s [%(processName)s, %(process)s] [%(levelname)-5.5s]  %(message)s',
                                      datefmt='%m-%d %H:%M')
    init_logger = logging.getLogger()
    init_logger.setLevel(logging.DEBUG)
    # File logger
    file_handler = logging.FileHandler("log/{}.log".format(log_file_name))
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(file_handler_level)
    # Screen logger
    screen_handler = logging.StreamHandler()
    screen_handler.setLevel(screen_handler_level)
    init_logger.addHandler(file_handler)
    init_logger.addHandler(screen_handler)
    return init_logger


def make_cv_index(data_size, cv_num, random=False):
    instance_id = range(data_size)
    if random:
        np.random.shuffle(instance_id)
    for cv in xrange(cv_num):
        train, dev = list(), list()
        for i in instance_id:
            if i % cv_num == cv:
                dev.append(i)
            else:
                train.append(i)
        yield train, dev
