# coding=utf-8
import sys
sys.path.append('../../../')
from src.data_utils.phrase_utils import phrase_list_generator1
from src.embedding import WordEmbedding
from src.rae_batch import brae_predict, GraphBRAE
from src.utils import pre_logger
from src.config import GBRAEConfig
from src.data_utils.phrase_utils import PARA_INDEX
import numpy as np
sys.setrecursionlimit(50000)
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


def has_zero_para(phrases):
    for phrase in phrases:
        if len(phrase[PARA_INDEX]) == 0:
            return True
    return False


def main():
    config_name = sys.argv[1]
    train_data = "data/250w/tune_hyperparameter/train/tune.train"
    dev_data = "data/250w/tune_hyperparameter/dev/tune.dev"
    test_data = "data/250w/tune_hyperparameter/test/tune.test"
    brae_config = GBRAEConfig(config_name)
    gbrae_data_name = "data/250w/tune_hyperparameter/gbrae.data.tune.min.count.%d.pkl" % brae_config.min_count
    gbrae_dict_name = "data/250w/tune_hyperparameter/gbrae.dict.tune.min.count.%d.pkl" % brae_config.min_count
    gbrae_phrase_dict_name = "data/250w/tune_hyperparameter/gbrae.tune.phrase.text.dict.pkl"
    train_name = "dim%d_lrec%f_lsem%f_lword%f_alpha%f_beta%f_gama%f_num%d_seed%d_batch%d_min%d_lr%f" % (brae_config.dim,
                                                                                                                brae_config.weight_rec,
                                                                                                                brae_config.weight_sem,
                                                                                                                brae_config.weight_l2,
                                                                                                                brae_config.alpha,
                                                                                                                brae_config.beta,
                                                                                                                brae_config.gama,
                                                                                                                brae_config.trans_num,
                                                                                                                brae_config.random_seed,
                                                                                                                brae_config.batch_size,
                                                                                                                brae_config.min_count,
                                                                                                                brae_config.optimizer.param["lr"])
    model_name = "model/%s" % "gbrae_tune_hyper_" + train_name
    pre_train_model_file_name = None
    temp_model = model_name + ".temp"
    start_iter = int(sys.argv[2]) if len(sys.argv) > 3 else 0
    end_iter = int(sys.argv[3]) if len(sys.argv) > 4 else 26
    if len(sys.argv) > 5:
        pre_train_model_file_name = sys.argv[5]
        model_name += "_pred_%s" % pre_train_model_file_name
    pre_logger("gbrae_tune_hyper_" + train_name)
    np.random.seed(brae_config.random_seed)
    if start_iter == 0:
        print "Load Dict ..."
        src_word_dict, tar_word_dict = load_gbrae_dict(gbrae_dict_name)
        print "Compiling Model ..."
        brae = pre_model(src_word_dict, tar_word_dict, brae_config, verbose=True)
        if pre_train_model_file_name is not None:
            brae.load_model(pre_train_model_file_name)
        print "Write Binary Data ..."
        with open(temp_model, 'wb') as fout:
            pickle.dump(brae, fout)
            pickle.dump(np.random.get_state(), fout)
        if end_iter == 1:
            exit(1)
    else:
        with open(temp_model, 'rb') as fin:
            brae = pickle.load(fin)
            np.random.set_state(pickle.load(fin))
    src_phrases, tar_phrases, src_tar_pair = load_gbrae_data(gbrae_data_name)
    if has_zero_para(src_phrases):
        print "src has zero para"
    else:
        print "src has not zero para"
    if has_zero_para(tar_phrases):
        print "tar phrases has zero para"
    else:
        print("tar has not zero para")
    with open(gbrae_phrase_dict_name, 'rb') as fin:
        src_phrase2id = pickle.load(fin)
        tar_phrase2id = pickle.load(fin)
    train_pair = load_sub_data_pair(train_data, src_phrase2id, tar_phrase2id)
    dev_pair = load_sub_data_pair(dev_data, src_phrase2id, tar_phrase2id)
    test_pair = load_sub_data_pair(test_data, src_phrase2id, tar_phrase2id)
    brae.tune_hyper_parameter(src_phrases, tar_phrases, train_pair, dev_pair, test_pair,
                              brae_config, model_name, start_iter=start_iter, end_iter=end_iter)
    brae.save_model("%s.tune.model" % model_name)


if __name__ == "__main__":
    main()
