import argparse
import sys

import numpy as np

from entity_scorer import ReasonTrainer

sys.path.append('../../')
from src.utils import pre_logger
from src.Initializer import UniformInitializer


def read_train_data_file(filename, _entity_id_dict, _relation_id_dict):
    instances = list()
    with open(filename, 'r') as fin:
        for line in fin:
            e1, r, e2 = line.strip().split('\t')[:3]
            instances.append((_entity_id_dict[e1], _entity_id_dict[e2], _relation_id_dict[r]))
    return instances


def read_eval_data_file(filename, _entity_id_dict, _relation_id_dict):
    instances = dict()
    with open(filename, 'r') as fin:
        for line in fin:
            e1, r, e2, label = line.strip().split('\t')
            e1_id = _entity_id_dict[e1]
            e2_id = _entity_id_dict[e2]
            r_id = _relation_id_dict[r]
            if r_id not in instances:
                instances[r_id] = {'pos': list(), 'neg': list()}
            if label == '1':
                instances[r_id]['pos'].append((e1_id, e2_id, r_id))
            elif label == '-1':
                instances[r_id]['neg'].append((e1_id, e2_id, r_id))
            else:
                raise KeyError
    for rid in instances.keys():
        instances[rid]['pos'] = np.array(instances[rid]['pos'])
        instances[rid]['neg'] = np.array(instances[rid]['neg'])
    return instances


def data_to_indexs(filenames):
    _entity_id_dict = dict()
    _relation_id_dict = dict()
    for fname in filenames:
        with open(fname, 'r') as fin:
            for line in fin:
                e1, r, e2 = line.strip().split('\t')[:3]
                if e1 not in _entity_id_dict:
                    _entity_id_dict[e1] = len(_entity_id_dict) + 1
                if e2 not in _entity_id_dict:
                    _entity_id_dict[e2] = len(_entity_id_dict) + 1
                if r not in _relation_id_dict:
                    _relation_id_dict[r] = len(_relation_id_dict)
    return _entity_id_dict, _relation_id_dict


if __name__ == "__main__":
    train_file = "train.txt"
    dev_file = "dev.txt"
    test_file = "test.txt"

    parser = argparse.ArgumentParser()
    parser.add_argument('--entity', type=int, default=50, help='Entity Dim')
    parser.add_argument('--seed', type=int, default=1993, help='Random Seed')
    parser.add_argument('--batch', type=int, default=50, help='Batch Size')
    parser.add_argument('--iter', type=int, default=500, help='Max Iter Number')
    parser.add_argument('--hidden', type=int, default=50, help='Hidden Dim')
    parser.add_argument('--scale', type=float, default=0.1, help='Uniform Initializer Scale')
    parser.add_argument('--negative', type=int, default=10, help='Negative Number')
    # parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate')
    args = parser.parse_args()
    log_file_name = "slm_e%s_h%s_b%s_n%d_seed%d_i" \
                    "%s_scale%s" % (args.entity, args.hidden, args.batch, args.negative, args.seed,
                                                                args.iter, args.scale)

    pre_logger(log_file_name)
    np.random.seed(args.seed)
    entity_dict, relation_dict = data_to_indexs([train_file, dev_file, test_file])
    train_data = read_train_data_file(train_file, entity_dict, relation_dict)
    dev_data = read_eval_data_file(dev_file, entity_dict, relation_dict)
    test_data = read_eval_data_file(test_file, entity_dict, relation_dict)
    trainer = ReasonTrainer(entity_dict, relation_dict, entity_dim=args.entity, k=args.hidden,
                            initializer=UniformInitializer(scale=args.scale))
    trainer.train_relation(train_data, dev_data, test_data, max_iter=args.iter, C=args.negative, batch_size=args.batch)
