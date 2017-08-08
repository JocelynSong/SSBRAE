#!/bin/bash
THEANO_FLAGS='floatX=float32,optimizer=fast_run';OMP_NUM_THREADS=1 python brae_tune_hyper_parameter.py /home/jocelyn/SSBRAE/code/dl4nlp-master/conf/brae1.conf