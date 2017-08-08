#!/bin/bash
THEANO_FLAGS='floatX=float32,optimizer=fast_run' python test_kl.py /home/jocelyn/SSBRAE/code/dl4nlp-master/conf/gbrae.conf \
            /home/jocelyn/SSBRAE/code/dl4nlp-master/exp/gbrae/gbrae/model/dim50_lrec0.000100_lsem0.000010_lword0.000010_alpha0.050000_beta0.700000_gama0.250000_num2_seed1993_batch50_min3_lr0.010000_iter25.model \
            /home/jocelyn/SSBRAE/code/dl4nlp-master/exp/gbrae/gbrae/model/dim50_lrec0.000100_lsem0.000010_lword0.000010_alpha0.050000_beta0.700000_gama0.250000_num2_seed1993_batch50_min3_lr0.010000_iter25.model.rs \
            test_kl_gbrae