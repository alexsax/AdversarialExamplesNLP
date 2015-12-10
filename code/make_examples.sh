#!/usr/bin/env bash

# load models at different epochs and save the vectors to file - then output the qvecs
python word2vec_basic.py --modelfile -m "$1" -g
for i in `seq 1 11`;
  do
  	python word2vec_basic.py --modelfile -m "$1" -g -a
  done    