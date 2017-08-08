# coding=utf-8
import logging
import sys
import argparse
import numpy as np
import os
sys.path.append('../../../')
from src.rae_batch import brae_predict
from src.utils import pre_logger


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, help='Phrase File Name')
parser.add_argument('-m', '--model', type=str, help='Model File Name')
parser.add_argument('-o', '--output', type=str, help='Output File Name')
parser.add_argument('-b', '--bilinear', type=bool, default=False, help='Bilinear Score')
parser.add_argument('-p', '--process', type=int, default=1, help='Multi-Process, default is 1')
parser.add_argument('-n', '--normalize', type=bool, default=True, help='Pre-Trained Model Path')
parser.add_argument('--neg', type=str, help='Neg Phrase File Name')
parser.add_argument('--neg_out', type=str, help='Neg Out File Name')
args = parser.parse_args()
pre_logger("score_margin_exp_" + args.model.split(os.sep)[-1])
logger = logging.getLogger(__name__)

brae_predict(phrase_file=args.file, output_file=args.output, model_file=args.model, normalize=args.normalize,
             num_process=args.process, bilinear=args.bilinear)

if args.neg is not None:
    if args.neg_out is None:
        raise IOError("args.neg_out is None")
    brae_predict(phrase_file=args.neg, output_file=args.neg_out, model_file=args.model, normalize=args.normalize,
                 num_process=args.process, bilinear=args.bilinear)

score_list = list()
with open(args.output, 'r') as fin:
    fin.readline()
    for line in fin:
        if args.bilinear:
            _, _, score = line.strip().split(" ||| ")
            score_list.append([float(score)])
        else:
            _, _, score1, score2 = line.strip().split(" ||| ")
            score_list.append([float(score1), float(score2)])
scores = np.array(score_list)
logger.info("POS SCORE:")
logger.info("Avg Score: %s" % np.average(scores, axis=0))
logger.info("Max Score: %s" % np.max(scores, axis=0))
logger.info("Min Score: %s" % np.min(scores, axis=0))

neg_score_list = list()
with open(args.neg_out, 'r') as fin:
    fin.readline()
    for line in fin:
        if args.bilinear:
            _, _, score = line.strip().split(" ||| ")
            neg_score_list.append([float(score)])
        else:
            _, _, score1, score2 = line.strip().split(" ||| ")
            neg_score_list.append([float(score1), float(score2)])
neg_scores = np.array(neg_score_list)
difference = scores - neg_scores
logger.info("MARGIN SCORE:")
logger.info("Avg Score: %s" % np.average(difference, axis=0))
logger.info("Max Score: %s" % np.max(difference, axis=0))
logger.info("Min Score: %s" % np.min(difference, axis=0))
