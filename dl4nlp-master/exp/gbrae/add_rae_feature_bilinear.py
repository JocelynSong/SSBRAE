# -*- coding: utf-8 -*-
import os
import sys
import numpy as np


__author__ = "roger"

RAW_INDEX = 0
EXP_INDEX = 1
SUM_INDEX = "|||SUM|||"

translation_rule_path = "translation_rule"
translation_rule_file = ["u8_nist05_src.token.pos.depe.sgm.phrase",
                         "u8_nist06_src.token.pos.depe.sgm.phrase",
                         "u8_nist08_src.token.pos.depe.sgm.phrase"]
output_path = ""
phrase_feature_file = sys.argv[1]
phrase_feature_dict = dict()
with open(phrase_feature_file, "r") as fin:
    fin.readline()
    for line in fin:
        line = line.strip().decode("utf-8")
        att = line.split(" ||| ")
        src_phrase, tar_phrase, score = att[:3]
        if score == 'nan':
            score = 0
        score = float(score)
        phrase_feature_dict["%s ||| %s" % (src_phrase, tar_phrase)] = [score, np.exp(score)]
print "%d translation rules in %s " % (len(phrase_feature_dict), phrase_feature_file)

for filename in translation_rule_file:
    tar_dict = dict()
    with open(translation_rule_path + os.sep + filename, 'r') as fin:
        for line in fin:
            line = line.strip().decode("utf-8")
            src, tar, feature = line.split(" ||| ")
            key = "%s ||| %s" % (src, tar)
            if key not in phrase_feature_dict:
                sys.stderr.write("Warning: %s can't found\n" % key.encode("utf-8"))
                continue
            if tar not in tar_dict:
                tar_dict[tar] = 0
            tar_dict[tar] += phrase_feature_dict[key][EXP_INDEX]

    with open(translation_rule_path + os.sep + filename, 'r') as fin:
        with open(filename + ".pred", "w") as fout:
            for line in fin:
                line = line.replace("\n", "").replace("\r", "").decode("utf-8")
                src, tar, feature = line.split(" ||| ")
                key = "%s ||| %s" % (src, tar)
                if key in phrase_feature_dict and tar in tar_dict:
                    score = phrase_feature_dict[key][EXP_INDEX] / tar_dict[tar]
                    feature += "%s" % score
                else:
                    sys.stderr.write("Warning: %s can't found\n" % key.encode("utf-8"))
                    feature += "0 0"
                to_write = "%s ||| %s |||   ||| %s\n" % (src, tar, feature)
                fout.write(to_write.encode("utf-8"))
