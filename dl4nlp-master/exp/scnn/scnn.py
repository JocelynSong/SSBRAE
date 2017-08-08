import sys

import numpy as np

sys.path.append('../../')
from src.cnn import ShallowCNNClassifier
from src.utils import read_sst, load_model


def fine():
    np.random.seed(0)
    train, dev, test, word_idx = read_sst(u"/home/roger/Corpus/sst_fine/sst.fine.train",
                                          u"/home/roger/Corpus/sst_fine/sst.fine.dev",
                                          u"/home/roger/Corpus/sst_fine/sst.fine.test",
                                          )
    # embedding = WordEmbedding(word_idx, filename=u"GoogleNews-vectors-negative300.bin")
    embedding = load_model("Google_for_fine.bin")
    classifier = ShallowCNNClassifier(embedding, n_out=5, verbose=True, weight_l2=0.001)
    classifier.fit(train, dev, test)
    acc, pred = classifier.test(test[0], test[1])
    print acc


def bi():
    np.random.seed(0)
    train, dev, test, word_idx = read_sst(u"/home/roger/Corpus/sst_bi/sst.bi.train",
                                          u"/home/roger/Corpus/sst_bi/sst.bi.dev",
                                          u"/home/roger/Corpus/sst_bi/sst.bi.test",
                                          )
    # embedding = WordEmbedding(word_idx, filename=u"GoogleNews-vectors-negative300.bin")
    embedding = load_model("Google_for_bi.bin")
    classifier = ShallowCNNClassifier(embedding, n_out=5, verbose=True, weight_l2=0.01)
    classifier.fit(train, dev, test)
    acc, pred = classifier.test(test[0], test[1])
    print acc


if __name__ == "__main__":
    if sys.argv[1] == "fine":
        fine()
    else:
        bi()
