import sys

sys.path.append('../../')
from src.Initializer import UniformInitializer, GlorotUniformInitializer
from src.embedding import WordEmbedding
from src.utils import make_cv_index, pre_logger
from multi_task_nn import MultiTaskHierarchicalClassifier
import numpy as np
import codecs


def get_ccf_word_map(filename, topk=30000, encoding='utf8'):
    word_freq = dict()
    instances = codecs.open(filename, 'r', encoding=encoding).readlines()
    for instance in instances:
        att = instance.split("\t")
        for query in att[4:]:
            for word in query.split():
                if word not in word_freq:
                    word_freq[word] = 1
                else:
                    word_freq[word] += 1
    word_freq = [(value, key) for key, value in word_freq.iteritems()]
    word_freq.sort(reverse=True)
    word_freq = [key for key, value in word_freq]
    word_map = dict()
    for word in word_freq[:topk]:
        word_map[word] = len(word_map) + 1
    return word_map



def read_ccf_file(filename, word_dict, split_symbol=" |||| ", low_case=False, add_unknown_word=False, encoding="utf8",
                  max_query_len=10):
    id_list = list()
    x = list()
    y_age = list()
    y_gender = list()
    y_education = list()
    instances = codecs.open(filename, 'r', encoding=encoding).readlines()
    for instance in instances:
        att = instance.split("\t")
        id_list.append(att[0])
        y_age.append(int(att[1]))
        y_gender.append(int(att[2]))
        y_education.append(int(att[3]))
        token = list()
        for query in att[4:]:
            q = list()
            for word in query.split():
                if word not in word_dict:
                    if add_unknown_word:
                        word_dict[word] = len(word_dict) + 1
                    else:
                        continue
                q.append(word_dict[word])
            if len(q) > max_query_len:
                q = q[:max_query_len]
            else:
                for i in xrange(max_query_len - len(q)):
                    q.append(0)
            token.append(q)
        x.append(token)
    len_list = [len(tokens) for tokens in x]
    max_len = np.max(len_list)
    for instance, length in zip(x, len_list):
        for j in xrange(max_len - length):
            instance.append([0] * max_query_len)
    y = np.concatenate([[y_age], [y_gender], [y_education]]).transpose()
    return np.array(x, dtype=np.int32), y.astype(np.int32) - 1


def pre_classifier(word_idx, embedding_name, labels_nums, word_dim, hidden_dims, batch_size, dropout, act):
    hidden_dims = [int(hidden) for hidden in hidden_dims.split("_")]
    embedding_initializer = UniformInitializer(scale=0.1)
    weight_initializer = GlorotUniformInitializer()
    embedding = WordEmbedding(word_idx, dim=word_dim, filename=embedding_name, binary=True,
                              initializer=embedding_initializer, add_unknown_word=True)
    classifier = MultiTaskHierarchicalClassifier(embedding, in_dim=embedding.dim, hidden_dims=hidden_dims,
                                     initializer=weight_initializer, batch_size=batch_size,
                                     dropout=dropout, labels_nums=labels_nums, activation=act,
                                     )
    return classifier


def bi(embedding_name, word_dim, hidden_dims, batch_size, dropout, act,
       cv_num=5, iter_num=25):
    model_name = "wdim_%d_hdim_%s_batch_%d_dropout_%f_act_%s" % (word_dim, hidden_dims,
                                                                 batch_size, dropout, act)
    log_file_name = "%s.log" % model_name
    logger = pre_logger(log_file_name)
    word_idx = get_ccf_word_map("user_tag_query.2W.seg.utf8.TRAIN")
    data_x, data_y = read_ccf_file("user_tag_query.2W.seg.utf8.TRAIN", word_idx, add_unknown_word=False)
    labels_nums = [len(np.unique(data_y[:, task_index])) - 1 for task_index in xrange(data_y.shape[1])]
    max_dev_acc_list = list()
    # predict_result_list = list()
    for train_index, dev_index in make_cv_index(data_x.shape[0], cv_num):
        train_x = data_x[train_index]
        train_y = data_y[train_index]
        dev_x = data_x[dev_index]
        dev_y = data_y[dev_index]
        classifier = pre_classifier(word_idx, embedding_name, labels_nums, word_dim,
                                    hidden_dims, batch_size, dropout, act)
        max_dev_acc = classifier.train([train_x, train_y], [dev_x, dev_y], iter_num=iter_num)
        max_dev_acc_list.append(max_dev_acc)
    logger.info("Aver Dev Acc: %f" % np.mean(max_dev_acc_list))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--embedding', type=str, help='Word Embedding File')
    parser.add_argument('-w', '--word_dim', type=int, help='Dim of Word Embedding')
    parser.add_argument('--hidden_dims', type=str, default='10', help='Hidden Dim')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch Size')
    parser.add_argument('-d', '--dropout', type=float, default=0, help='Dropout Rate')
    parser.add_argument('-a', '--act', type=str, default="sigmoid", help='Act Function')
    parser.add_argument('-c', '--cv', type=int, default=5, help='Cross Validation Number')
    parser.add_argument('-i', '--iter', type=int, default=25, help='Iter Number')
    args = parser.parse_args()
    bi(args.embedding, args.word_dim, args.hidden_dims, args.batch_size,
       args.dropout, args.act, args.cv, args.iter)
