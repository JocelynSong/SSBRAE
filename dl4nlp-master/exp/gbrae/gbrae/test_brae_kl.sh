#!/bin/bash
THEANO_FLAGS='floatX=float32,optimizer=fast_run' python test_kl.py /home/sjs/jocelyn/SSBRAE/dl4nlp-master/conf/gbrae1.conf \
            /home/sjs/jocelyn/SSBRAE/dl4nlp-master/exp/gbrae/brae/model/dim50_lrec0.000100_lsem0.000010_ll20.000010_alpha0.050000_seed1993_batch50_min3_lr0.010000_iter25.model \
            /home/sjs/jocelyn/SSBRAE/dl4nlp-master/exp/gbrae/brae/model/dim50_lrec0.000100_lsem0.000010_ll20.000010_alpha0.050000_seed1993_batch50_min3_lr0.010000_iter25.model.rs \
            test_brae