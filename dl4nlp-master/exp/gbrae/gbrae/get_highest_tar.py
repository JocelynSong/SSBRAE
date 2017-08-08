import sys
sys.path.append('../../../')
from src.data_utils.phrase_utils import phrase_list_generator1
from src.embedding import WordEmbedding
from src.rae_batch import brae_predict, GraphBRAE
from src.utils import pre_logger
from src.config import GBRAEConfig
from src.data_utils.phrase_utils import PARA_INDEX, TRAN_INDEX
import numpy as np
sys.setrecursionlimit(50000)
try:
    import cPickle as pickle
except:
    import pickle


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


def get_top_pair(src_phrases, src_phrase2id, test_phrase, filename):
    f = open(test_phrase)
    f2 = open(filename, 'w')
    for line in f:
        src = (line.strip().split("|||")[0]).strip()
        if src not in src_phrase2id.keys():
            continue
        src_ind = src_phrase2id[src]
        src_trans_candidate = src_phrases[src_ind][TRAN_INDEX]
        if src_trans_candidate is None or len(src_trans_candidate) == 0:
            continue
        sorted(src_trans_candidate.items(), key=lambda x: x[1])
        tar_ind = src_trans_candidate.items()[-1][0]
        f2.write(str(src_ind)+"|||"+str(tar_ind)+"\n")
    f.close()
    f2.close()


def save_phrase_pair(filename, pair):
    f = open(filename, 'w')
    for key in pair.keys():
        f.write(key+"|||"+pair[key]+"\n")
    f.close()


def main():
    gbrae_data_name = "data/250w/gbrae.data.250w.min.count.3.pkl"
    gbrae_phrase_dict_name = "data/250w/gbrae.250w.phrase.text.dict.pkl"
    test_phrase = "/home/sjs/jocelyn/SSBRAE/kl/dict/fre_phrase.file"
    src_phrases, tar_phrases, src_tar_pair = load_gbrae_data(gbrae_data_name)
    with open(gbrae_phrase_dict_name, 'rb') as fin:
        src_phrase2id = pickle.load(fin)
        tar_phrase2id = pickle.load(fin)
    get_top_pair(src_phrases, src_phrase2id, test_phrase, sys.argv[1])

if __name__ == '__main__':
    main()
