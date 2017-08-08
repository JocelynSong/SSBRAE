__author__ = 'roger'

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, default=50, help='model dim')
parser.add_argument('-r', '--wrec', type=float, default=1e-3, help='Lambda Rec Params')
parser.add_argument('-s', '--wsem', type=float, default=1e-3, help='Lambda Sem Params')
parser.add_argument('-l', '--wl', type=float, default=1e-2, help='Lambda Word Params')
parser.add_argument('-a', '--alpha', type=float, default=0.15, help='Alpha for Rec and Sem')
parser.add_argument('-b', '--beta', type=float, default=0.3, help='Beta for ISOMAP on Trans')
parser.add_argument('-n', '--num', type=int, default=1, help='Num for Translation Candidate')
parser.add_argument('-m', '--min_count', type=int, default=1, help='Mini Count for Word')
parser.add_argument('--seed', type=int, default=1993, help='Random Seed')
parser.add_argument('--batch', type=int, default=50, help='Batch Size')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate')
args = parser.parse_args()

dim = args.dim
weight_rec = args.wrec
weight_sem = args.wsem
weight_l2 = args.wl
alpha = args.alpha
beta = args.beta
num = args.num
seed = args.seed
min_count = args.min_count
batch = args.batch
lr = args.lr

content = """
[functions]
activation=tanh

[architectures]
dim=%d
normalize=True
weight_rec=%f
weight_sem=%f
weight_l2=%f
alpha=%f
beta=%f
trans_num=%d

[parameters]
random_seed=%d
n_epoch=25
batch_size=%d
dropout=0
min_count=%d

[optimizer]
optimizer=SGD
lr=%f""" % (dim, weight_rec, weight_sem, weight_l2, alpha, beta, num, seed, batch, min_count, lr)

conf_name = "conf/dim%d_lrec%f_lsem%f_lword%f_alpha%f_beta%f_num%d_seed%d_batch%d_min%d_lr%f.config" % (dim, weight_rec,
                                                                                                        weight_sem,
                                                                                                        weight_l2,
                                                                                                        alpha, beta,
                                                                                                        num, seed,
                                                                                                        batch,
                                                                                                        min_count, lr)
with open(conf_name, "w") as fout:
    fout.write(content)
