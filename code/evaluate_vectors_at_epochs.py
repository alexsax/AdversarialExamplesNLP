#!/usr/bin/env bash

# load models at different epochs and save the vectors to file - then output the qvecs
touch results.txt
for i in `seq 6 10`;
  do
  	python word2vec_basic.py --modelfile "my-model-"$i"0000"
  	echo $i"0000: " >> results.txt
  	bash ./test_vectors.sh >> results.txt
  	echo "\n" >> results.txt
  done    