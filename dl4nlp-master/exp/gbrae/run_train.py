import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, help='Model Python Program File Name')
parser.add_argument('-c', '--config', type=str, help='Config File Name')
parser.add_argument('-s', '--start', type=int, default=0, help='The Start Iter Num')
parser.add_argument('-p', '--pretrain', type=str, default=None, help='Pre-Trained Model Path')
args = parser.parse_args()


for i in xrange(args.start, 26):
    """
    zero for compile
    """
    if args.pretrain is not None:
        cmd = "python %s train %s %d %d %s" % (args.model, args.config, i, i + 1, args.pretrain)
    else:
        cmd = "python %s train %s %d %d" % (args.model, args.config, i, i + 1)
    print cmd
    os.system(cmd)

if args.pretrain is not None:
    cmd = "python %s predict %s 1 %s" % (args.model, args.config, args.pretrain)
else:
    cmd = "python %s predict %s 1" % (args.model, args.config)
print cmd
os.system(cmd)
