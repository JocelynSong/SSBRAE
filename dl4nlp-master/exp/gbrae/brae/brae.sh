#!/bin/bash
THEANO_FLAGS='floatX=float32,optimizer=fast_run' python brae_batch.py train /home/sjs/jocelyn/SSBRAE/dl4nlp-master/conf/brae.conf