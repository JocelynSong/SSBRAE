# coding=utf-8
import sys
sys.path.append('../../../')
from src.embedding import WordEmbedding
from src.rae_batch import brae_predict, BilingualPhraseRAEISOMAP
from src.data_utils.phrase_utils import read_phrase_pair_vocab, read_phrase_list, filter_vocab, add_trans_word_vocab, read_trans_list, \
    clean_text
from src.utils import pre_logger
from src.config import BRAEISOMAPConfig
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
        src_embedding = WordEmbedding(src_dict, filename="data/zh.token.dim%d.bin" % config.dim, dim=config.dim)
        tar_embedding = WordEmbedding(tar_dict, filename="data/en.token.dim%d.bin" % config.dim, dim=config.dim)
    return BilingualPhraseRAEISOMAP(src_embedding, tar_embedding, config=config, verbose=verbose)


def main():
    train_test = sys.argv[1]
    if train_test not in ["train", "predict"]:
        sys.stderr("train or predict")
        exit(1)
    config_name = sys.argv[2]
    forced_decode_data = "data/brae.train.data"
    phrase_data_path = "data/phrase.list"
    src_count_path = "data/src.trans.data"
    tar_count_path = "data/tar.trans.data"
    brae_config = BRAEISOMAPConfig(config_name)
    train_name = "dim%d_lrec%f_lsem%f_ll2%f_alpha%f_beta%f_num%d_seed%d_batch%d_lr%f" % (brae_config.dim,
                                                                                         brae_config.weight_rec,
                                                                                         brae_config.weight_sem,
                                                                                         brae_config.weight_l2,
                                                                                         brae_config.alpha,
                                                                                         brae_config.beta,
                                                                                         brae_config.trans_num,
                                                                                         brae_config.random_seed,
                                                                                         brae_config.batch_size,
                                                                                         brae_config.optimizer.param["lr"])
    model_name = "model/%s" % train_name
    temp_model = model_name + ".temp"
    if train_test == "train":
        start_iter = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        end_iter = int(sys.argv[4]) if len(sys.argv) > 4 else 25
        pre_logger("braeisomap_" + train_name)
        np.random.seed(brae_config.random_seed)
        if start_iter == 0:
            src_word_dict, tar_word_dict = read_phrase_pair_vocab(forced_decode_data)
            src_word_dict, tar_word_dict = add_trans_word_vocab(src_count_path, src_word_dict, tar_word_dict)
            tar_word_dict, src_word_dict = add_trans_word_vocab(tar_count_path, src_word_dict, tar_word_dict)
            src_word_dict = filter_vocab(src_word_dict, min_count=0)
            tar_word_dict = filter_vocab(tar_word_dict, min_count=0)
            src_phrases, tar_phrases, src_tar_pair = read_phrase_list(forced_decode_data, src_word_dict, tar_word_dict)

            src_phrases, tar_phrases = read_trans_list(src_count_path, src_phrases, tar_phrases,
                                                       src_word_dict, tar_word_dict)
            tar_phrases, src_phrases = read_trans_list(tar_count_path, tar_phrases, src_phrases,
                                                       tar_word_dict, src_word_dict)
            src_phrases = clean_text(src_phrases)
            tar_phrases = clean_text(tar_phrases)
            brae = pre_model(src_word_dict, tar_word_dict, brae_config, verbose=True)
            with open(temp_model, 'wb') as fout:
                pickle.dump(src_phrases, fout)
                pickle.dump(tar_phrases, fout)
                pickle.dump(src_tar_pair, fout)
                pickle.dump(brae, fout)
                pickle.dump(np.random.get_state(), fout)
            if end_iter == 1:
                exit(1)
        else:
            with open(temp_model, 'rb') as fin:
                src_phrases = pickle.load(fin)
                tar_phrases = pickle.load(fin)
                src_tar_pair = pickle.load(fin)
                brae = pickle.load(fin)
                np.random.set_state(pickle.load(fin))
        brae.train(src_phrases, tar_phrases, src_tar_pair, brae_config, model_name, start_iter, end_iter)
        brae.save_model("%s.model" % model_name)
    elif train_test == "predict":
        num_process = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        brae_predict(phrase_data_path, train_name + ".pred", model_file="%s.model" % model_name,
                     bilinear=True, num_process=num_process)
    else:
        sys.stderr("train or predict")
        exit(1)


if __name__ == "__main__":
    main()
