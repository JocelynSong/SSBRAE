import theano.tensor as T


__author__ = 'roger'


def iden(x):
    return x


class Activation(object):
    def __init__(self, method):
        self.method = method
        method_name = method.lower()
        if method_name == "sigmoid":
            self.func = T.nnet.sigmoid
        elif method_name == "tanh":
            self.func = T.tanh
        elif method_name == "relu":
            self.func = T.nnet.relu
        elif method_name == "elu":
            self.func = T.nnet.elu
        elif method_name == 'iden':
            self.func = iden
        else:
            raise ValueError('Invalid Activation function!')

    def activate(self, x):
        return self.func(x)
