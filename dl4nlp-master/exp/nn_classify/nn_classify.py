import sys
sys.path.append('../../')
from src.Initializer import UniformInitializer, GlorotUniformInitializer
from src.embedding import WordEmbedding
from src.utils import read_sst
from src.layers import EmbeddingClassifier


def pre_logger():
    import logging
    # Logging configuration
    # Set the basic configuration of the logging system
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    screen_handler = logging.StreamHandler()
    screen_handler.setLevel(logging.DEBUG)
    logger.addHandler(screen_handler)


def bi():
    pre_logger()
    train, dev, test, word_idx = read_sst(u"sst.bi.train",
                                          u"sst.bi.dev",
                                          u"sst.bi.test",
                                          )
    # embedding = WordEmbedding(word_idx, filename=u"GoogleNews-vectors-negative300.bin")
    embedding_initializer = UniformInitializer(scale=0.1)
    weight_initializer = GlorotUniformInitializer()
    # embedding = WordEmbedding(word_idx, filename=u"imdb.50.bin", initializer=embedding_initializer)
    embedding = WordEmbedding(word_idx, dim=50, initializer=embedding_initializer)
    classifier = EmbeddingClassifier(embedding, in_dim=embedding.dim, hidden_dim=50,
                                     initializer=weight_initializer, batch_size=64,
                                     num_label=2, activation="tanh",
                                     )
    classifier.train(train, dev, test)


if __name__ == "__main__":
    bi()
