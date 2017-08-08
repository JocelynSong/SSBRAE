from numpy import set_printoptions
import unittest
import sys

sys.path.append('../')


from src.embedding import WordEmbedding
from src.cnn import ShallowCNNClassifier
from src.utils import read_sst, load_model


__author__ = 'roger'


set_printoptions(threshold='nan')

"""def test_rae_classifier(self):
        word_idx = {"he": 1, "she": 2, "I": 3, "him": 4}
        embedding = WordEmbedding(word_idx, fname=u"F:\\Corpus\\imdb.50.bin")
        classifier = PhraseRAEClassifier(embedding=embedding, n_out=2, uniform_range=0.01, normalize=False,
                                         weight_rec=0.5, weight_l2=0.01, dropout=0, verbose=True)
        rae = classifier.encoder
        phrase = [[1],
                  [2],
                  [3],
                  [4]]
        truth = [[0, 1]]
        for i in range(5):
            nodes, seq = rae.generate_node_path(phrase)
            result = classifier.compute_result_grad(nodes, seq, truth)
            loss = result[0]
            pred = result[1]
            grad = result[2:]
            a = np.asscalar(np.array([0.5]))
            rae.update_param(grad, a)
            print loss
            print pred
            # print grad
        nodes, seq = rae.generate_node_path(phrase)
        result = classifier.output(nodes, seq)
        print result
        """


def test_cnn():
    import numpy as np
    np.random.seed(0)
    train, dev, test, word_idx = read_sst(u"C:\\Users\\roger\\NLP\\Corpus\\sst_bi\\sst.bi.train",
                                          u"C:\\Users\\roger\\NLP\\Corpus\\sst_bi\\sst.bi.dev",
                                          u"C:\\Users\\roger\\NLP\\Corpus\\sst_bi\\sst.bi.test",
                                          )
    embedding = WordEmbedding(word_idx, dim=5)  # fname=u"F:\\Corpus\\GoogleNews-vectors-negative300.bin")
    classifier = ShallowCNNClassifier(embedding, n_out=2, verbose=True, weight_l2=0.001)
    classifier.fit(train, dev, test)
    acc, pred = classifier.test(test[0], test[1])
    print acc


def test_load():
    train, dev, test, word_idx = read_sst(u"C:\\Users\\roger\\NLP\\Corpus\\sst_bi\\sst.bi.train",
                                          u"C:\\Users\\roger\\NLP\\Corpus\\sst_bi\\sst.bi.dev",
                                          u"C:\\Users\\roger\\NLP\\Corpus\\sst_bi\\sst.bi.test",
                                          )
    classifier = load_model("cnn_model")
    acc, pred = classifier.test(test[0], test[1])
    print acc
    return pred


class MyTestCase(unittest.TestCase):
    def test_something(self):
        test_cnn()
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
