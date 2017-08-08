# coding=utf-8
import sys
sys.path.append('../../../')
from src.embedding import WordEmbedding
from src.rae_batch import brae_predict, GraphBRAE
from src.utils import pre_logger, load_random_state
from src.config import GBRAEConfig
sys.setrecursionlimit(50000)
try:
    import cPickle as pickle
except:
    import pickle

__author__ = 'song'
rand_word_init = False


def pre_model(src_dict, tar_dict, config, verbose):
    if rand_word_init:
        src_embedding = WordEmbedding(src_dict, dim=config.dim)
        tar_embedding = WordEmbedding(tar_dict, dim=config.dim)
    else:
        en_embedding_name = "data/embedding/en.token.dim%d.bin" % config.dim
        zh_embedding_name = "data/embedding/zh.token.dim%d.bin" % config.dim
        src_embedding = WordEmbedding(src_dict, filename=zh_embedding_name, dim=config.dim)
        tar_embedding = WordEmbedding(tar_dict, filename=en_embedding_name, dim=config.dim)
    return GraphBRAE(src_embedding, tar_embedding, config=config, verbose=verbose)


def load_gbrae_data(filename):
    with open(filename, 'rb') as fin:
        print "Read Source Phrases Data ..."
        src_phrases = pickle.load(fin)
        print "Read Target Phrases Data ..."
        tar_phrases = pickle.load(fin)
        src_tar_pair = pickle.load(fin)
    return src_phrases, tar_phrases, src_tar_pair


def load_gbrae_dict(filename):
    with open(filename, 'rb') as fin:
        print "Read Word Dict ..."
        src_word_dict = pickle.load(fin)
        tar_word_dict = pickle.load(fin)
    return src_word_dict, tar_word_dict


def phrase_generator(filename):
    f = open(filename)
    for line in f:
        words = line.strip().split("|||")
        src = words[0].strip()
        tar = words[1].strip()
        yield src, tar


def load_sub_data_pair(filename, src_phrase2id, tar_phrase2id):
    pair_list = list()
    for src_phrase, tar_phrase in phrase_generator(filename):
        if src_phrase in src_phrase2id and tar_phrase in tar_phrase2id:
            pair_list.append((src_phrase2id[src_phrase], tar_phrase2id[tar_phrase]))
        else:
            if src_phrase not in src_phrase2id:
                print "[SRC] %s cannot found" % src_phrase
            if tar_phrase not in tar_phrase2id:
                print "[TAR] %s cannot found" % tar_phrase
    return pair_list


def main():
    config_name = sys.argv[1]
    test_file = "./data/250w/fre_phrase.pair"
    brae_config = GBRAEConfig(config_name)
    gbrae_data_name = "data/250w/gbrae.data.250w.min.count.%d.pkl" % brae_config.min_count
    gbrae_dict_name = "data/250w/gbrae.dict.250w.min.count.%d.pkl" % brae_config.min_count
    gbrae_phrase_dict_name = "data/250w/gbrae.250w.phrase.text.dict.pkl"
    model_name = sys.argv[2]
    random_state_name = sys.argv[3]
    logger_name = sys.argv[4]
    pre_logger(logger_name)
    load_random_state(random_state_name)
    src_word_dict, tar_word_dict = load_gbrae_dict(gbrae_dict_name)
    brae = pre_model(src_word_dict, tar_word_dict, brae_config, verbose=True)
    brae.load_model(model_name)
    src_phrases, tar_phrases, src_tar_pair = load_gbrae_data(gbrae_data_name)
    with open(gbrae_phrase_dict_name, 'rb') as fin:
        src_phrase2id = pickle.load(fin)
        tar_phrase2id = pickle.load(fin)
    test_pair = load_sub_data_pair(test_file, src_phrase2id, tar_phrase2id)
    brae.test_kl(src_phrases, tar_phrases, test_pair, brae_config)

if __name__ == '__main__':
    main()




