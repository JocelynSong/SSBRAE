from numpy import set_printoptions
import numpy as np
import sys

sys.path.append('../')
from src.config import BRAEConfig
from src.embedding import WordEmbedding
from src.rae import PhraseRAEClassifier, BilingualPhraseRAE
from src.utils import read_sst, read_forced_decode


__author__ = 'roger'

import unittest

set_printoptions(threshold='nan')


def test_rae_sentiment():
    train, dev, test, word_idx = read_sst(u"E:\\Corpus\\mr\\mr.shuffle.train",
                                          u"E:\\Corpus\\mr\\mr.shuffle.test",
                                          u"E:\\Corpus\\mr\\mr.shuffle.test",
                                          )
    embedding = WordEmbedding(word_idx, dim=3)  # fname=u"F:\\Corpus\\imdb.50.bin")
    classifier = PhraseRAEClassifier(embedding=embedding, n_out=2, uniform_range=0.01, normalize=False,
                                     weight_rec=0.001, weight_l2=0.01, dropout=0, verbose=True)
    classifier.fit(train, dev, test)


def test_brae():
    np.random.seed(0)
    src_word_idx = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}
    src_embedding = WordEmbedding(src_word_idx, dim=3)
    tar_word_idx = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}
    tar_embedding = WordEmbedding(tar_word_idx, dim=3)
    src_pos, src_neg = [[1], ], [[4]]
    tar_pos, tar_neg = [[1], ], [[4]]
    brae = BilingualPhraseRAE(src_embedding, tar_embedding)
    src_pos_nodes, src_pos_seq = brae.source_encoder.generate_node_path(src_pos)
    src_neg_nodes, src_neg_seq = brae.source_encoder.generate_node_path(src_neg)
    tar_pos_nodes, tar_pos_seq = brae.target_encoder.generate_node_path(tar_pos)
    tar_neg_nodes, tar_neg_seq = brae.target_encoder.generate_node_path(tar_neg)
    print brae.compute_result_grad(src_pos_nodes, src_pos_seq, src_neg_nodes, src_neg_seq,
                                   tar_pos_nodes, tar_pos_seq, tar_neg_nodes, tar_neg_seq)


def test_brae_corpus():
    source_phrase, target_phrase, src_tar_pair, src_word_dict, tar_word_dict = read_forced_decode("../data/fd.txt")
    config = BRAEConfig("../conf/brae.conf")
    src_embedding = WordEmbedding(src_word_dict, dim=50)
    tar_embedding = WordEmbedding(tar_word_dict, dim=50)
    brae = BilingualPhraseRAE(src_embedding, tar_embedding, config=config)
    brae.train_using_lbfgs(source_phrase, target_phrase, src_tar_pair)
    # brae.train(source_phrase, target_phrase, src_tar_pair)
    # brae.predict(src_tar_pair, "output.sem.txt")


class MyTestCase(unittest.TestCase):
    def test_something(self):
        test_brae_corpus()
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
