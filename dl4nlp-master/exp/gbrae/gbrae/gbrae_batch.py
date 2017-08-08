# coding=utf-8
import sys

sys.path.append('../../../')
from src.embedding import WordEmbedding
from src.rae_batch import brae_predict, GraphBRAE
from src.utils import pre_logger
from src.config import GBRAEConfig
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


def train_brae(brae_trainer, source_target_pair, config, _train_name):
    return brae_trainer.train(source_target_pair, config, _train_name)


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


def main():
    train_test = sys.argv[1]
    if train_test not in ["train", "predict"]:
        sys.stderr("train or predict")
        exit(1)
    config_name = sys.argv[2]
    phrase_data_path = "data/phrase.list"
    brae_config = GBRAEConfig(config_name)
    gbrae_data_name = "data/250w/gbrae.data.250w.min.count.%d.pkl" % brae_config.min_count
    gbrae_dict_name = "data/250w/gbrae.dict.250w.min.count.%d.pkl" % brae_config.min_count
    #gbrae_data_name = "data/250w/tune_hyperparameter/train/gbrae.data.5w.tune.min.count.%d.pkl" % brae_config.min_count
    #gbrae_dict_name = "data/250w/tune_hyperparameter/train/gbrae.dict.5w.tune.min.count.%d.pkl" % brae_config.min_count
    if brae_config.para and brae_config.trans:
        train_name = "dim%d_lrec%f_lsem%f_lword%f_alpha%f_beta%f_gama%f_num%d_seed%d_batch%d_min%d_lr%f" % (
            brae_config.dim,
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
    elif brae_config.para:
        train_name = "para_dim%d_lrec%f_lsem%f_lword%f_alpha%f_beta%f_gama%f_num%d_seed%d_batch%d_min%d_lr%f" % (
            brae_config.dim,
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
    elif brae_config.trans:
        train_name = "trans_dim%d_lrec%f_lsem%f_lword%f_alpha%f_beta%f_gama%f_num%d_seed%d_batch%d_min%d_lr%f" % (
            brae_config.dim,
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

    model_name = "model/%s" % train_name
    pre_train_model_file_name = None
    temp_model = model_name + ".temp"
    if train_test == "train":
        start_iter = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        end_iter = int(sys.argv[4]) if len(sys.argv) > 4 else 26
        if len(sys.argv) > 5:
            pre_train_model_file_name = sys.argv[5]
            model_name += "_pred_%s" % pre_train_model_file_name
        pre_logger("gbrae_" + train_name)
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
        brae.train(src_phrases, tar_phrases, src_tar_pair, brae_config, model_name, start_iter, end_iter)
        brae.save_model("%s.model" % model_name)
    elif train_test == "predict":
        num_process = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        if len(sys.argv) > 4:
            pre_train_model_file_name = sys.argv[4]
            model_name += "_pred_%s" % pre_train_model_file_name
        brae_predict(phrase_data_path, train_name + ".pred", model_file="%s.model" % model_name,
                     bilinear=False, num_process=num_process)
    else:
        sys.stderr("train or predict")
        exit(1)


if __name__ == "__main__":
    main()
