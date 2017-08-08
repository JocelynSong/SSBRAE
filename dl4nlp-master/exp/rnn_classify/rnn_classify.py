import sys
sys.path.append('../../')
from src.Initializer import UniformInitializer, GlorotUniformInitializer
from src.recurrent import RecurrentEncoder, RecurrentNormEncoder
from src.embedding import WordEmbedding
from src.utils import read_sst
import numpy as np


def pre_logger():
    import logging
    # Logging configuration
    # Set the basic configuration of the logging system
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    screen_handler = logging.StreamHandler()
    screen_handler.setLevel(logging.DEBUG)
    logger.addHandler(screen_handler)


def bi(seed=0):
    pre_logger()
    np.random.seed(seed)
    train, dev, test, word_idx = read_sst(u"sst.fine.train",
                                          u"sst.fine.dev",
                                          u"sst.fine.test",
                                          )
    # embedding = WordEmbedding(word_idx, filename=u"GoogleNews-vectors-negative300.bin")
    embedding_initializer = UniformInitializer(scale=0.1)
    weight_initializer = GlorotUniformInitializer()
    # embedding = WordEmbedding(word_idx, filename=u"imdb.50.bin", initializer=embedding_initializer)
    embedding = WordEmbedding(word_idx, dim=64, initializer=embedding_initializer)
    from src.recurrent import RecurrentClassifier
    classifier = RecurrentClassifier(embedding, recurrent_encoder=RecurrentEncoder, in_dim=embedding.dim, hidden_dim=64,
                                     initializer=weight_initializer, batch_size=64,
                                     num_label=5, pooling="final", activation="tanh"
                                     )
    classifier.train(train, dev, test)


def bi_normal(seed):
    # pre_logger()
    np.random.seed(seed)
    train, dev, test, word_idx = read_sst(u"sst.bi.train",
                                          u"sst.bi.dev",
                                          u"sst.bi.test",
                                          )
    # embedding = WordEmbedding(word_idx, filename=u"GoogleNews-vectors-negative300.bin")
    embedding_initializer = UniformInitializer(scale=0.1)
    weight_initializer = GlorotUniformInitializer()
    # embedding = WordEmbedding(word_idx, filename=u"imdb.50.bin", initializer=embedding_initializer)
    embedding = WordEmbedding(word_idx, dim=64, initializer=embedding_initializer)
    from src.recurrent import RecurrentClassifier
    classifier = RecurrentClassifier(embedding, recurrent_encoder=RecurrentNormEncoder, in_dim=embedding.dim, hidden_dim=64,
                                     initializer=weight_initializer, batch_size=64,
                                     num_label=2, pooling="final", activation="tanh"
                                     )
    classifier.train(train, dev, test)


if __name__ == "__main__":
    bi()