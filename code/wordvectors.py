from gensim import corpora, models, similarities
from nltk.corpus import reuters, brown
import os.path
import os
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model_file = "../models/reuters_model.w2v"
vectors_file = "../data/vectors.txt"
model = None

def loadModel():
  if os.path.isfile(model_file):
    model = models.Word2Vec.load(model_file)
  else:
    sentences = reuters.sents()
    model = models.Word2Vec(sentences, size=100, window=3, min_count=1, workers=4, sg=1)
    model.save(model_file)
  return model

def saveModelVectors(model):
  model.save_word2vec_format(vectors_file) 

model = loadModel()
saveModelVectors(model)
os.system("./qvec-master/qvec.py --in_vectors " + vectors_file + " --in_oracle  ./qvec-master/oracles/semcor_noun_verb.supersenses ")
