import gensim, logging
import zipfile

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def read_data(filename):
  f = zipfile.ZipFile(filename)
  for name in f.namelist():
    return f.read(name).split()
  f.close()

words = read_data('../data/text8.zip')
words = words[0:128*10001]
print len(words)
# train word2vec on the two sentences
# min_count 10 works well for whole dat set
model = gensim.models.Word2Vec([words], size=128, alpha=0.025, window=1, min_count=0, 
														max_vocab_size=50000, sample=0, seed=1, workers=8, 
														min_alpha=0.0001, sg=0, hs=0, negative=64, 
														cbow_mean=0, iter=1, null_word=1)
model.save_word2vec_format('../data/vectors.txt', fvocab=None, binary=False)