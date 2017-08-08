# coding=utf-8
# Generate GBRAE data .pkl file
import sys
sys.path.append('../../../')

from src.embedding import WordEmbedding
from src.data_utils.phrase_utils import read_phrase_list, read_trans_list, read_para_list, clean_text, TEXT_INDEX

try:
   import cPickle as pickle
except:
   import pickle

__author__ = 'roger'
rand_word_init = False


def main():
    min_count = int(sys.argv[1])
    dim = 50
    '''
    forced_decode_data = "data/brae.train.data"
    src_count_path = "data/src.trans.data"
    tar_count_path = "data/tar.trans.data"
    tar_para_path = "data/tar.para.data"
    src_para_path = "data/src.para.data"
    gbrae_data_name = "model/gbrae.data.min.count.%d.pkl" % min_count
    gbrae_dict_name = "model/gbrae.dict.min.count.%d.pkl" % min_count
    gbrae_phrase_dict_name = "model/gbrae.phrase.text.dict.pkl"
    '''
    forced_decode_data = "data/250w/tune_hyperparameter/tune.data"
    src_count_path = "data/250w/tune_hyperparameter/tune.data"
    #tar_count_path = "data/250w/phrase-table.filtered"
    tar_para_path = "data/250w/enBP_alignPhraProb.xml"
    src_para_path = "data/250w/chBP_alignPhraProb.xml"
    gbrae_data_name = "data/250w/tune_hyperparameter/gbrae.data.tune.min.count.%d.pkl" % min_count
    gbrae_dict_name = "data/250w/tune_hyperparameter/train/gbrae.dict.tune.min.count.%d.pkl" % min_count
    gbrae_phrase_dict_name = "data/250w/tune_hyperparameter/gbrae.tune.phrase.text.dict.pkl"
    print "Load Word Dict ..."
    en_embedding_name = "data/embedding/en.token.dim%d.bin" % dim
    zh_embedding_name = "data/embedding/zh.token.dim%d.bin" % dim
    tar_word_dict = WordEmbedding.load_word2vec_word_map(en_embedding_name, binary=True, oov=True)
    src_word_dict = WordEmbedding.load_word2vec_word_map(zh_embedding_name, binary=True, oov=True)
    print "Load All Data ..."
    src_phrases, tar_phrases, src_tar_pair = read_phrase_list(forced_decode_data, src_word_dict, tar_word_dict)
    print "Load Para Data ..."
    src_phrases = read_para_list(src_para_path, src_phrases, src_word_dict)
    tar_phrases = read_para_list(tar_para_path, tar_phrases, tar_word_dict)
    print "Load Trans Data ..."
    src_phrases, tar_phrases = read_trans_list(src_count_path, src_phrases, tar_phrases,
                                               src_word_dict, tar_word_dict)
    #tar_phrases, src_phrases = read_trans_list(tar_count_path, tar_phrases, src_phrases,
                                               #tar_word_dict, src_word_dict)
    src_phrase2id = dict()
    tar_phrase2id = dict()
    for phrase, i in zip(src_phrases, xrange(len(src_phrases))):
        src_phrase2id[phrase[TEXT_INDEX]] = i
    for phrase, i in zip(tar_phrases, xrange(len(tar_phrases))):
        tar_phrase2id[phrase[TEXT_INDEX]] = i
    src_phrases = clean_text(src_phrases)
    tar_phrases = clean_text(tar_phrases)
    with open(gbrae_dict_name, 'wb') as fout:
        print "Write Word Dict ..."
        pickle.dump(src_word_dict, fout)
        pickle.dump(tar_word_dict, fout)
    with open(gbrae_data_name, 'wb') as fout:
        print "Write Source Phrases Data ..."
        pickle.dump(src_phrases, fout)
        print "Write Target Phrases Data ..."
        pickle.dump(tar_phrases, fout)
        pickle.dump(src_tar_pair, fout)
    with open(gbrae_phrase_dict_name, 'wb') as fout:
        print "Write Source Phrases Dictionary ..."
        pickle.dump(src_phrase2id, fout)
        print "Write Target Phrases Dictionary ..."
        pickle.dump(tar_phrase2id, fout)


if __name__ == "__main__":
    main()
