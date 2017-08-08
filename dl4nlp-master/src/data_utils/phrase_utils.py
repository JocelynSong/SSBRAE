# -*- coding: utf-8 -*-
from collections import OrderedDict

from src import OOV_KEY
from src.data_utils import para_weight_threshold


WORD_INDEX = 0
TEXT_INDEX = 1
PARA_INDEX = 2
TRAN_INDEX = 3


def phrase_list_generator1(filename, src_max=7, tar_max=7):
    f = open(filename)
    for line in f:
        s = line.strip().split('|||')
        src_phrase = s[0].strip()
        tar_phrase = s[1].strip()
        if len(src_phrase.split()) > src_max or len(tar_phrase.split()) > tar_max:
            continue
        weights = s[3].strip().split()
        weight1 = weights[0]
        weight2 = weights[2]
        yield src_phrase, tar_phrase, weight1, weight2


def para_list_generator(filename, src_max=7, tar_max=7):
    f = open(filename)
    for line in f:
        s = line.strip().split('|||')
        src_phrase = s[0].strip()
        tar_phrase = s[1].strip()
        if len(src_phrase.split()) > src_max or len(tar_phrase.split()) > tar_max:
            continue
        weight1 = s[3].strip()
        weight2 = s[4].strip()
        yield src_phrase, tar_phrase, weight1, weight2


def trans_list_generator(filename, src_max=7, tar_max=7):
    f = open(filename)
    for line in f:
        s = line.strip().split('|||')
        src_phrase = s[0].strip()
        tar_phrase = s[1].strip()
        if len(src_phrase.split()) > src_max or len(tar_phrase.split()) > tar_max:
            continue
        weights = s[3].strip().split()
        weight1 = weights[0]
        weight2 = weights[2]
        yield src_phrase, tar_phrase, weight1, weight2


def phrase_list_generator(filename, src_max=7, tar_max=7):
    with file(filename, 'r') as fin:
        for line in fin:
            line = line.replace('\n', '').replace('\r', '').decode('utf-8')
            if "0 0 0 -50 -50" in line:
                continue
            if len(line.split(" ||| ")) < 3:
                src_phrase, tar_phrase = line.split(" ||| ")[:2]
                src_phrase = src_phrase.strip()
                tar_phrase = tar_phrase.strip()
                count = 0
                if len(src_phrase.split()) > src_max or len(tar_phrase.split()) > tar_max:
                    continue
                yield src_phrase, tar_phrase, count
            else:
                src_phrase, tar_phrase, att = line.split(" ||| ")[:3]
                src_phrase = src_phrase.strip()
                tar_phrase = tar_phrase.strip()
                count = att.split()[0]
                if len(src_phrase.split()) > src_max or len(tar_phrase.split()) > tar_max:
                    continue
                yield src_phrase, tar_phrase, count


def add_word(vocab, word):
    """
    在词表中加入词汇
    :param vocab 词表
    :param word  需添加的词
    """
    if word not in vocab:
        vocab[word] = {"index": len(vocab) + 1, "count": 1}
    else:
        vocab[word]["count"] += 1


def words2index(vocab, words):
    """
    将多个词转化为词表中的索引 找不到则返回OOV
    :param vocab:
    :param words:
    :return:
    """
    return [vocab[word] if word in vocab else vocab[OOV_KEY] for word in words.split()]


def read_phrase_pair_vocab(filename):
    """
    读入短语表中所有出现的词汇
    :param filename: 短语表文件名
    :return:
    """
    src_words = {}
    tar_words = {}
    for src, tar, _ in phrase_list_generator(filename):
        for word in src.split():
            add_word(src_words, word)
        for word in tar.split():
            add_word(tar_words, word)
    return src_words, tar_words


def add_para_word_vocab(filename, word_idx):
    for src, tar, weight in phrase_list_generator(filename):
        if weight < para_weight_threshold:
            continue
        for word in src.split():
            add_word(word_idx, word)
        for word in tar.split():
            add_word(word_idx, word)
    return word_idx


def get_phrase_instance(word_idx, phrase):
    # [words indexs, text, para list, trans dict]
    return [words2index(word_idx, phrase), phrase, list(), dict()]


def add_trans_word_vocab(filename, src_word_idx, tar_word_idx):
    for src, tar, _ in phrase_list_generator(filename):
        for word in src.split():
            add_word(src_word_idx, word)
        for word in tar.split():
            add_word(tar_word_idx, word)
    return src_word_idx, tar_word_idx


def filter_vocab(word_idx, min_count=5):
    """
    依据词频对词汇进行过滤
    :param word_idx:
    :param min_count:
    :return:
    """
    filter_dict = dict()
    for word, value in word_idx.iteritems():
        if value["count"] >= min_count:
            # 词典索引从1开始 将0索引留出来
            filter_dict[word] = len(filter_dict) + 1
    filter_dict[OOV_KEY] = len(filter_dict) + 1
    return filter_dict

'''
def read_phrase_list(phrase_file, src_word_idx, tar_word_idx, src_max_len=7, tar_max_len=7):
    src_phrase_idx = dict()
    tar_phrase_idx = dict()
    src_phrase_list = list()
    tar_phrase_list = list()
    bi_phrase_pair = list()
    src_index = 0
    tar_index = 0
    for src_phrase, tar_phrase, count in phrase_list_generator(phrase_file, src_max=src_max_len, tar_max=tar_max_len):
        if src_phrase not in src_phrase_idx:
            src_phrase_idx[src_phrase] = src_index
            src_phrase_list.append(get_phrase_instance(src_word_idx, src_phrase))
            src_index += 1
        if tar_phrase not in tar_phrase_idx:
            tar_phrase_idx[tar_phrase] = tar_index
            tar_phrase_list.append(get_phrase_instance(tar_word_idx, tar_phrase))
            tar_index += 1

        src_idx = src_phrase_idx[src_phrase]
        tar_idx = tar_phrase_idx[tar_phrase]
        src_phrase_list[src_idx][TRAN_INDEX][tar_idx] = int(count)
        tar_phrase_list[tar_idx][TRAN_INDEX][src_idx] = int(count)
        bi_phrase_pair.append((src_phrase_idx[src_phrase], tar_phrase_idx[tar_phrase]))
    return src_phrase_list, tar_phrase_list, bi_phrase_pair
'''


def read_phrase_list(phrase_file, src_word_idx, tar_word_idx, src_max_len=7, tar_max_len=7):
    src_phrase_idx = dict()
    tar_phrase_idx = dict()
    src_phrase_list = list()
    tar_phrase_list = list()
    bi_phrase_pair = list()
    src_index = 0
    tar_index = 0
    for src_phrase, tar_phrase, weight1, weight2 in phrase_list_generator1(phrase_file, src_max=src_max_len, tar_max=tar_max_len):
        if src_phrase not in src_phrase_idx:
            src_phrase_idx[src_phrase] = src_index
            src_phrase_list.append(get_phrase_instance(src_word_idx, src_phrase))
            src_index += 1
        if tar_phrase not in tar_phrase_idx:
            tar_phrase_idx[tar_phrase] = tar_index
            tar_phrase_list.append(get_phrase_instance(tar_word_idx, tar_phrase))
            tar_index += 1

        src_idx = src_phrase_idx[src_phrase]
        tar_idx = tar_phrase_idx[tar_phrase]
        src_phrase_list[src_idx][TRAN_INDEX][tar_idx] = float(weight1)
        tar_phrase_list[tar_idx][TRAN_INDEX][src_idx] = float(weight2)
        bi_phrase_pair.append((src_phrase_idx[src_phrase], tar_phrase_idx[tar_phrase]))
    return src_phrase_list, tar_phrase_list, bi_phrase_pair

'''
def read_para_list(para_file, phrase_list, word_idx):
    text2pid = dict()
    for phrase, i in zip(phrase_list, xrange(len(phrase_list))):
        text2pid[phrase[TEXT_INDEX]] = i
    for src, tar, weight in phrase_list_generator(para_file):
        if weight < para_weight_threshold:
            continue
        if src not in text2pid:
            phrase_list.append(get_phrase_instance(word_idx, src))
            text2pid[src] = len(phrase_list) - 1
        if tar not in text2pid:
            phrase_list.append(get_phrase_instance(word_idx, src))
            text2pid[tar] = len(phrase_list) - 1
        src_idx = text2pid[src]
        tar_idx = text2pid[tar]
        phrase_list[src_idx][PARA_INDEX].append((tar_idx, float(weight)))
        phrase_list[tar_idx][PARA_INDEX].append((src_idx, float(weight)))
    return phrase_list
'''


def read_para_list(para_file, phrase_list, word_idx):
    text2pid = dict()
    for phrase, i in zip(phrase_list, xrange(len(phrase_list))):
        text2pid[phrase[TEXT_INDEX]] = i
    for src, tar, weight1, weight2 in para_list_generator(para_file):
        if float(weight1) < para_weight_threshold or float(weight2) < para_weight_threshold:
            continue
        if src not in text2pid:
            continue
            '''
            phrase_list.append(get_phrase_instance(word_idx, src))
            text2pid[src] = len(phrase_list) - 1
            '''
        if tar not in text2pid:
            continue
            '''
            phrase_list.append(get_phrase_instance(word_idx, src))
            text2pid[tar] = len(phrase_list) - 1
            '''
        src_idx = text2pid[src]
        tar_idx = text2pid[tar]
        phrase_list[src_idx][PARA_INDEX].append((tar_idx, float(weight1)))
        phrase_list[tar_idx][PARA_INDEX].append((src_idx, float(weight2)))
    return phrase_list


def read_trans_list(bi_count_file, src_phrase_list, tar_phrase_list, src_word_idx, tar_word_idx):
    src_text2pid = dict()
    tar_text2pid = dict()
    for phrase, i in zip(src_phrase_list, xrange(len(src_phrase_list))):
        src_text2pid[phrase[TEXT_INDEX]] = i
    for phrase, i in zip(tar_phrase_list, xrange(len(tar_phrase_list))):
        tar_text2pid[phrase[TEXT_INDEX]] = i

    for src, tar, weight1, weight2 in trans_list_generator(bi_count_file):
        if src not in src_text2pid:
            src_phrase_list.append(get_phrase_instance(src_word_idx, src))
            src_text2pid[src] = len(src_phrase_list) - 1
        if tar not in tar_text2pid:
            tar_phrase_list.append(get_phrase_instance(tar_word_idx, tar))
            tar_text2pid[tar] = len(tar_phrase_list) - 1
        src_idx = src_text2pid[src]
        tar_idx = tar_text2pid[tar]
        src_phrase_list[src_idx][TRAN_INDEX][tar_idx] = float(weight1)
        tar_phrase_list[tar_idx][TRAN_INDEX][src_idx] = float(weight2)

    # calc translate probability
    def calc_translate_probability(phrase_list):
        for p in phrase_list:
            if "trans" not in p:
                continue
            else:
                count_sum = sum(p[TRAN_INDEX].values())
                for key, value in p[TRAN_INDEX].iteritems():
                    p[TRAN_INDEX][key] = float(value) / count_sum

    calc_translate_probability(src_phrase_list)
    calc_translate_probability(tar_phrase_list)
    # Sort Each Trans Dict from large to small
    for src_phrase in src_phrase_list:
        src_phrase[TRAN_INDEX] = OrderedDict(sorted(src_phrase[TRAN_INDEX].iteritems(),
                                                    key=lambda d: d[1], reverse=True))
    for tar_phrase in tar_phrase_list:
        tar_phrase[TRAN_INDEX] = OrderedDict(sorted(tar_phrase[TRAN_INDEX].iteritems(),
                                                    key=lambda d: d[1], reverse=True))
    # calc_translate_probability(tar_phrase_list)
    return src_phrase_list, tar_phrase_list


def clean_text(phrases):
    import gc
    for p in phrases:
        p[TEXT_INDEX] = None
    gc.collect()
    gc.collect()
    gc.collect()
    return phrases


def clean_trans(phrases):
    import gc
    for p in phrases:
        p[TRAN_INDEX] = None
    gc.collect()
    gc.collect()
    gc.collect()
    return phrases
