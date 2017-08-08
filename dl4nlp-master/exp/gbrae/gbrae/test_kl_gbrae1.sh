#!/bin/bash
THEANO_FLAGS='floatX=float32,optimizer=fast_run' python test_kl.py /home1/wcx/jocelyn/dl4nlp-master/conf/gbrae1.conf \
            /home1/wcx/jocelyn/dl4nlp-master/exp/gbrae/gbrae/model/dim50_lrec0.001000_lsem0.000100_lword0.000010_alpha0.050000_beta0.700000_gama0.250000_num2_seed1993_batch50_min3_lr0.010000_iter25.model \
            /home1/wcx/jocelyn/dl4nlp-master/exp/gbrae/gbrae/model/dim50_lrec0.001000_lsem0.000100_lword0.000010_alpha0.050000_beta0.700000_gama0.250000_num2_seed1993_batch50_min3_lr0.010000_iter25.model.rs \
            test_kl_gbrae_05725