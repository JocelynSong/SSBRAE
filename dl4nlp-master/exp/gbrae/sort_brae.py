import sys

filename = sys.argv[1]
instances = list()
with open(filename, 'r') as fin:
    for line in fin:
        att = line.strip().split(" ||| ")
        src, tar = att[0], att[1]
        if len(att) == 3:
            score = float(att[2])
        elif len(att) == 4:
            score = (float(att[3]) + float(att[2])) / 2
        else:
            print line
            continue
        instances.append((score, src, tar))
instances.sort(reverse=True)
with open(filename, 'w') as fout:
    for score, src, tar in instances:
        fout.write("%f ||| %s ||| %s\n" % (score, src, tar))
