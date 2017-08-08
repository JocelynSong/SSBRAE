import argparse
import random

__author__ = 'roger'


parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, default=50, help='model dim')
parser.add_argument('-r', '--wrec', type=float, default=1e-3, help='Lambda Rec Params')
parser.add_argument('-s', '--wsem', type=float, default=1e-3, help='Lambda Sem Params')
parser.add_argument('-l', '--wl', type=float, default=1e-2, help='Lambda Word Params')
parser.add_argument('-a', '--alpha', type=float, default=0.15, help='Alpha for Rec ')
parser.add_argument('-b', '--beta', type=float, default=0.55, help='Beta for Sem')
parser.add_argument('-g', '--gama', type=float, default=0.15, help='Gama for LE on Para')
parser.add_argument('-d', '--delta', type=float, default=0.15, help='Delta for ISOMAP on Trans')
parser.add_argument('-n', '--num', type=int, default=5, help='Num for Translation Candidate')
parser.add_argument('-m', '--min_count', type=int, default=3, help='Mini Count for Word')
parser.add_argument('--seed', type=int, default=1993, help='Random Seed')
parser.add_argument('--batch', type=int, default=100, help='Batch Size')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate')
parser.add_argument('--random', type=bool, default=False, help='Random for alpha beta gama delta')
args = parser.parse_args()

dim = args.dim
weight_rec = args.wrec
weight_sem = args.wsem
weight_l2 = args.wl
alpha = args.alpha
beta = args.beta
gama = args.gama
delta = args.delta
num = args.num
seed = args.seed
batch = args.batch
min_count = args.min_count
lr = args.lr

if args.random:
    alpha = random.random()
    beta = random.random()
    gama = random.random()
    delta = random.random()

hyper_sum = alpha + beta + gama + delta
if hyper_sum > 1:
    alpha, beta, gama, delta = alpha / hyper_sum, beta / hyper_sum, gama / hyper_sum, delta / hyper_sum

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
gama=%f
delta=%f
trans_num=%d

[parameters]
random_seed=%d
n_epoch=25
batch_size=%d
dropout=0
min_count=%d

[optimizer]
optimizer=SGD
lr=%f""" % (dim, weight_rec, weight_sem, weight_l2, alpha, beta, gama, delta, num, seed, batch, min_count, lr)

conf_name = "conf/dim%d" % dim
conf_name += "_lrec%f" % weight_rec
conf_name += "_lsem%f" % weight_sem
conf_name += "_lword%f" % weight_l2
conf_name += "_alpha%f" % alpha
conf_name += "_beta%f" % beta
conf_name += "_gama%f" % gama
conf_name += "_delta%f" % delta
conf_name += "_num%d" % num
conf_name += "_seed%d" % seed
conf_name += "_batch%d" % batch
conf_name += "_min%d" % min_count
conf_name += "_lr%f" % lr
conf_name += ".config"
with open(conf_name, "w") as fout:
    fout.write(content)
