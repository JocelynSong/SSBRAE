# -*- coding: utf-8 -*-
import sys

cn_punc = {"［", "］", "～", "，", "；", "。", "、", "？", "（", "）"}
brae_file = sys.argv[1]
bisc_file = sys.argv[2]
gbrae_file = sys.argv[3]
score_dict = dict()


def has_cn(words):
    for w in words:
        if w in cn_punc:
            return True
    return False


def add_key(filename):
    with open(filename, 'r') as fin:
        for line in fin:
            score, _src, _tar = line.strip().split(" ||| ")
            if has_cn(_src) or has_cn(_tar):
                continue
            _key = "%s$$$%s" % (src, tar)
            score_dict[_key] = [0, 0, 0]


def add_score(filename, index):
    with open(filename, 'r') as fin:
        for line in fin:
            score, _src, _tar = line.strip().split(" ||| ")
            if has_cn(_src) or has_cn(_tar):
                continue
            score = float(score)
            _key = "%s$$$%s" % (_src, _tar)
            score_dict[_key][index] = score


add_key(brae_file)
add_score(brae_file, 2)
add_score(bisc_file, 1)
add_score(gbrae_file, 0)
score_list = list()
for key, scores in score_dict.iteritems():
    gbrae, bi, brae = scores
    src, tar = key.split("$$$")
    score_list.append((src, gbrae, bi, brae, tar))
fout = open("analysis.out", 'w')
score_list.sort(reverse=True)
for src, gbrae, bi, brae, tar in score_list:
    fout.write("%f ||| %f ||| %f ||| %s ||| %s\n" % (gbrae, bi, brae, src, tar))
fout.close()
