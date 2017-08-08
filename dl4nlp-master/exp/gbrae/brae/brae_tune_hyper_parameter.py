import sys
sys.path.append('../../../')
from src.embedding import WordEmbedding
from src.rae_batch import BilingualPhraseRAE, brae_predict
from src.data_utils.phrase_utils import read_phrase_pair_vocab, read_phrase_list, filter_vocab, clean_text, clean_trans, \
    WORD_INDEX, TEXT_INDEX, phrase_list_generator1
from src.utils import pre_logger
from src.config import BRAEConfig
import numpy as np
try:
   import cPickle as pickle
except:
   import pickle

__author__ = 'roger'
rand_word_init = False


def pre_model(src_dict, tar_dict, config, verbose):
    if rand_word_init:
        src_embedding = WordEmbedding(src_dict, dim=config.dim)
        tar_embedding = WordEmbedding(tar_dict, dim=config.dim)
    else:
        en_embedding_name = "../gbrae/data/embedding/en.token.dim%d.bin" % config.dim
        zh_embedding_name = "../gbrae/data/embedding/zh.token.dim%d.bin" % config.dim
        src_embedding = WordEmbedding(src_dict, filename=zh_embedding_name, dim=config.dim)
        tar_embedding = WordEmbedding(tar_dict, filename=en_embedding_name, dim=config.dim)
    return BilingualPhraseRAE(src_embedding, tar_embedding, config=config, verbose=verbose)


def load_sub_data_pair(filename, src_phrase2id, tar_phrase2id):
    pair_list = list()
    for src_phrase, tar_phrase, _ , __ in phrase_list_generator1(filename):
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
    forced_decode_data = "../gbrae/data/250w/tune_hyperparameter/tune.data"
    brae_config = BRAEConfig(config_name)
    train_data = "../gbrae/data/250w/tune_hyperparameter/train/tune.train"
    dev_data = "../gbrae/data/250w/tune_hyperparameter/dev/tune.dev"
    test_data = "../gbrae/data/250w/tune_hyperparameter/test/tune.test"
    train_name = "dim%d_lrec%f_lsem%f_ll2%f_alpha%f_seed%d_batch%d_min%d_lr%f" % (brae_config.dim,
                                                                                  brae_config.weight_rec,
                                                                                  brae_config.weight_sem,
                                                                                  brae_config.weight_l2,
                                                                                  brae_config.alpha,
                                                                                  brae_config.random_seed,
                                                                                  brae_config.batch_size,
                                                                                  brae_config.min_count,
                                                                                  brae_config.optimizer.param["lr"],)
    model_name = "model/%s" % train_name
    temp_model = model_name + ".temp"
    start_iter = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    end_iter = int(sys.argv[4]) if len(sys.argv) > 4 else 26
    pre_logger("brae_" + train_name)
    np.random.seed(brae_config.random_seed)
    if start_iter == 0:
        print "Load Dict ..."
        en_embedding_name = "../gbrae/data/embedding/en.token.dim%d.bin" % brae_config.dim
        zh_embedding_name = "../gbrae/data/embedding/zh.token.dim%d.bin" % brae_config.dim
        tar_word_dict = WordEmbedding.load_word2vec_word_map(en_embedding_name, binary=True, oov=True)
        src_word_dict = WordEmbedding.load_word2vec_word_map(zh_embedding_name, binary=True, oov=True)
        print "Compiling Model ..."
        brae = pre_model(src_word_dict, tar_word_dict, brae_config, verbose=True)
        print "Load All Data ..."
        src_phrases, tar_phrases, src_tar_pair = read_phrase_list(forced_decode_data, src_word_dict, tar_word_dict)
        src_train = [p[WORD_INDEX] for p in src_phrases]
        tar_train = [p[WORD_INDEX] for p in tar_phrases]
        print "Write Binary Data ..."
        with open(temp_model, 'wb') as fout:
            pickle.dump(src_train, fout)
            pickle.dump(tar_train, fout)
            pickle.dump(src_tar_pair, fout)
            pickle.dump(brae, fout)
            pickle.dump(np.random.get_state(), fout)
        if end_iter == 1:
            exit(1)
    else:
        with open(temp_model, 'rb') as fin:
            src_train = pickle.load(fin)
            tar_train = pickle.load(fin)
            src_tar_pair = pickle.load(fin)
            brae = pickle.load(fin)
            np.random.set_state(pickle.load(fin))
    src_phrase2id = dict()
    tar_phrase2id = dict()
    for phrase, i in zip(src_phrases, xrange(len(src_phrases))):
        src_phrase2id[phrase[TEXT_INDEX]] = i
    for phrase, i in zip(tar_phrases, xrange(len(tar_phrases))):
        tar_phrase2id[phrase[TEXT_INDEX]] = i
    train_pair = load_sub_data_pair(train_data, src_phrase2id, tar_phrase2id)
    dev_pair = load_sub_data_pair(dev_data, src_phrase2id, tar_phrase2id)
    test_pair = load_sub_data_pair(test_data, src_phrase2id, tar_phrase2id)
    brae.tune_hyper_parameter(src_train, tar_train, train_pair, dev_pair, test_pair,
                              brae_config, model_name, start_iter=start_iter, end_iter=end_iter)
    brae.save_model("%s.tune.model" % model_name)


if __name__ == "__main__":
    main()
