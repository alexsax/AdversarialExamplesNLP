from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.python.platform
from tensorflow.python.ops import gradients
import collections
import math
import numpy as np
import os
import random
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import urllib
import zipfile
import cPickle as pickle
# Step 1: Download the data.
url = 'http://mattmahoney.net/dc/'

# The path where previously stored models are saved. Set to "" to retrain from scratch
existing_graph_path = "my-model-80000"
generate_adversarial_examples = False
existing_auxiliary_graph_path = "../data/text8"
num_steps = 100001


def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urllib.urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception(
        'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename
filename = maybe_download('../data/text8.zip', 31344016)
# Read the data into a string.
def read_data(filename):
  f = zipfile.ZipFile(filename)
  for name in f.namelist():
    return f.read(name).split()
  f.close()
vocabulary_size = 30000
if not existing_graph_path:
  words = read_data(filename)
  # words = words[0:128*10001]
  print('Data size', len(words))
  # Step 2: Build the dictionary and replace rare words with UNK token.
  def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
      dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for i, word in enumerate(words):
      if word in dictionary:
        index = dictionary[word]
      else:
        index = 0  # dictionary['UNK']
        unk_count = unk_count + 1
      data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary
  data, count, dictionary, reverse_dictionary = build_dataset(words)
  del words  # Hint to reduce memory.
  print('Most common words (+UNK)', count[:5])
  print('Sample data', data[:10])
  print("Pickling dictionaries")
  pickle.dump(data, open(existing_auxiliary_graph_path + '-data', 'w+'))
  pickle.dump(dictionary, open(existing_auxiliary_graph_path + '-dictionary', 'w+'))
  pickle.dump(reverse_dictionary, open(existing_auxiliary_graph_path + '-reverse-dictionary', 'w+'))# Read in the data set
else: 
  print("loading from pickled dictionaries")
  data = pickle.load(open(existing_auxiliary_graph_path + '-data', 'r'))
  dictionary = pickle.load(open(existing_auxiliary_graph_path + '-dictionary', 'r'))
  reverse_dictionary = pickle.load(open(existing_auxiliary_graph_path + '-reverse-dictionary', 'r'))
  # data, dictionary, reverse_dictionary = pickle.load(open(existing_auxiliary_graph_path, 'r'))
  print("loaded")# Load from a persistent location
data_index = 0





# Step 4: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size, num_skips), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)

  # For generating a CBOW batch
  for i in range(batch_size):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [ skip_window ] # avoid label at center index
    labels[i] = buffer[skip_window]
    for j in range(num_skips):
      target = 0
      while target in targets_to_avoid:
        # target = random.randint(0, span - 1)
        target += 1
      targets_to_avoid.append(target)
      batch[i, j] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
def get_context(indices):
  return " ".join([reverse_dictionary[index] for index in indices])

for i in range(8):
  print(batch[i], '->', labels[i, 0])
  print(get_context(batch[i]), '->', reverse_dictionary[labels[i, 0]])





# Step 5: Build and train a skip-gram model.
batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.
# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(np.arange(valid_window), valid_size))
num_sampled = 64    # Number of negative examples to sample.

alpha = 0.5 # Interpolation constant for loss
eta = 0.01 # Adversarial examples: move away from correct
eps = 0.001 # Adversarial examples : move towards incorrect
graph = tf.Graph()
with graph.as_default():
  tf.set_random_seed(1337)
  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size*num_skips], name="train_inputs") # one option is to do reshapes
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1], name="train_labels")
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
  # Construct the variables.
  embeddings = tf.Variable(
      tf.random_uniform([vocabulary_size, embedding_size], -0.5/embedding_size, 0.5/embedding_size))
  nce_weights = tf.Variable(
      tf.truncated_normal([vocabulary_size, embedding_size],
                          stddev=1.0 / math.sqrt(embedding_size)))
  nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
  # Look up embeddings for inputs.

  curr_embedding = tf.nn.embedding_lookup(embeddings, train_inputs) # now sum over

  embed = tf.reshape(curr_embedding, [num_skips, batch_size, embedding_size ]) # NEW
  embed = tf.reduce_sum(embed, 0)
  # embed = tf.reshape(curr_embedding, [batch_size, num_skips, embedding_size]) # OLD
  # embed = tf.reduce_sum(embed, 1)

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
      tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                     num_sampled, vocabulary_size))


  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  lossGrad = gradients.gradients(loss, embed)[0]
  new_loss = alpha*loss + (1-alpha)*tf.reduce_mean(
      tf.nn.nce_loss(nce_weights, nce_biases, embed + eta*lossGrad, train_labels,
                     num_sampled, vocabulary_size))

  # Construct the SGD optimizer using a learning rate of 1.0.
  # optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
  # opt = tf.train.GradientDescentOptimizer(0.025)
  opt = tf.train.AdamOptimizer(0.007)
  # grads_and_vars = opt.compute_gradients(new_loss)
  grads_and_vars = opt.compute_gradients(loss)
  optimizer = opt.apply_gradients(grads_and_vars)

  print('initializing saver')
  variables_to_save = {'embeddings':embeddings, 
                        'nce_weights' : nce_weights, 
                        'nce_biases' : nce_biases}
                        # , 
                        # 'reverse_dictionary' : reverse_dictionary,
                        # 'dictionary' : dictionary,
                        # 'data' : data
                        # }
  saver = tf.train.Saver(variables_to_save)# Construct the neural network graph






# Step 6: Begin training
with tf.Session(graph=graph) as session:
  if existing_graph_path:
    print("Starting load")
    saver.restore(session, existing_graph_path)
    final_embeddings = embeddings.eval()
    print("Loaded - generating examples")
    if generate_adversarial_examples:
      batch_inputs, batch_labels = generate_batch(
      batch_size, num_skips, skip_window)
      batch_inputs = np.reshape(batch_inputs.T, batch_size*num_skips)
      feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}
      # Gradient - I think
      # I'm reasonably sure that this is the method by which we can set up and
      # evaluating a gradient without also doing backpropogation and updating 
      # the entire NN.
      # lossGrad = tf.stop_gradient(gradients.gradients(loss, embed)[0])

      real_grad = lossGrad.eval(feed_dict)

      # Pick a word that we want to turn everything into
      adversarial_labels = np.array([valid_examples[2]]*batch_size)
      adversarial_labels = np.reshape(adversarial_labels, [batch_size, 1])
      adversarial_feed_dict = {train_inputs : batch_inputs, train_labels : adversarial_labels}


      # real_grad = lossGrad.eval(feed_dict)
      adversarial_grad = np.sign(lossGrad.eval(adversarial_feed_dict))
      print(real_grad)
      print(adversarial_grad)
      # How to turn one word vector into another
      adversarial_perturbation =  eta*real_grad - eps*adversarial_grad
      print("ADVERSARIAL PERTURBATION")
      print(adversarial_perturbation.shape)


      # Embed contexts in the vector space and add the adversarial_perturbation
      [perturbed_embeddings] = session.run([tf.stop_gradient(curr_embedding)], feed_dict=feed_dict)
      print("PERTURBED")
      print(perturbed_embeddings.shape)

      for skip_num in xrange(num_skips):
        perturbed_embeddings[skip_num*batch_size:(skip_num+1)*batch_size] += adversarial_perturbation
      print(perturbed_embeddings)
      # Find most similar words to new embeddings
      current_embeddings = normalized_embeddings.eval() # Shape: vocab_size, embedding_size
      peturbed_similarity_matrix = np.dot(current_embeddings, perturbed_embeddings.T) # Shape
      print("Finding Nearest Neighbors")
      adversarial_word_vectors = np.zeros([batch_size, num_skips])
      for i in xrange(batch_size):
        for j in xrange(num_skips):
          adversarial_word_vectors[i][j] = (peturbed_similarity_matrix[:, j*batch_size + i]).argmax()
          # print((sorted(-peturbed_similarity_matrix[:, j*batch_size + i]))[0:10])

      print("Found Nearest Neighbors")
      adversarial_word_vectors = np.reshape(adversarial_word_vectors.T, batch_size*num_skips)

      def print_old_and_adversarial_contexts():
        for i in xrange(batch_size):
          for j in xrange(num_skips):
            print(reverse_dictionary[int(batch_inputs[i + batch_size*j])] + " ", end="")
            if j == num_skips/2-1:
              print("_"+reverse_dictionary[int(batch_labels[i])] + "_ ", end="")
          print(" -> ", end="")
          for j in xrange(num_skips):
            print(reverse_dictionary[int(adversarial_word_vectors[i + batch_size*j])] + " ", end="")
          print("")
        print(reverse_dictionary[valid_examples[2]])

      print_old_and_adversarial_contexts()

      adversarial_inputs_dict = {train_inputs : adversarial_word_vectors, train_labels : adversarial_labels}
      loss_results = session.run([lossGrad], feed_dict=adversarial_inputs_dict)[0]
      print(len(loss_results))
      for row in loss_results:
        norm = np.linalg.norm(row)
        if norm < 0.001:
          print(norm)
          print(row)  # This section is for loading and creating adversarial examples
  else: #This section is for training a new neural network
    # We must initialize all variables before we use them.
    tf.initialize_all_variables().run()
    print("Initialized")
    average_loss = 0
    post_adversarial_avg_loss = 0
    for step in xrange(num_steps):
      batch_inputs, batch_labels = generate_batch(
          batch_size, num_skips, skip_window)
      batch_inputs = np.reshape(batch_inputs.T, batch_size*num_skips)
      feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}
      # We perform one update step by evaluating the optimizer op (including it
      # in the list of returned values for session.run()
      _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)

      average_loss += loss_val
      if step % 500 == 0:
        if step > 0:
          average_loss = average_loss / 500
        # The average loss is an estimate of the loss over the last 2000 batches.
        print("Average loss at step ", step, ": ", average_loss)
        average_loss = 0
      # note that this is expensive (~20% slowdown if computed every 500 steps)
      if step % 10000 == 0:
        def print_similarities_to_valid_examples():
          sim = similarity.eval()
          for i in xrange(valid_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            def nearest_neighbors(idx, top_k=8):
              return (-sim[idx, :]).argsort()[1:top_k+1]
            top_k = 8
            nearest = nearest_neighbors(i)
            log_str = "Nearest to %s:" % valid_word
            for k in xrange(top_k):
              close_word = reverse_dictionary[nearest[k]]
              log_str = "%s %s," % (log_str, close_word)
            print(log_str)
        print('saving session')
        saver.save(session, 'my-model', global_step=step)
        print_similarities_to_valid_examples()
  final_embeddings = normalized_embeddings.eval()







# Step 7: Save the embeddings
output_file="../data/vectors.txt"
def save_to_file(embedding_size, filename='vectors.txt'):
  print("Saving the file into " + filename)
  with open(filename, 'w') as f:
    f.write(str(len(final_embeddings)) + " " + str(embedding_size) + "\n")
    for i, row in enumerate(final_embeddings):
      rowstring = ' '.join([reverse_dictionary[i]] + ['%.6f' % num for num in row] + ["\n"])
      f.write(rowstring)
  os.system('head -n 2 '+ filename)

save_to_file(embedding_size, output_file)

# Step 8: Visualize the embeddings.
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  #in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i,:]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
  plt.savefig(filename)
# try:
#   from sklearn.manifold import TSNE
#   import matplotlib.pyplot as plt
#   tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
#   plot_only = 500
#   low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
#   labels = list(dictionary.keys())[:plot_only]
#   plot_with_labels(low_dim_embs, labels)
# except ImportError:
#   print("Please install sklearn and matplotlib to visualize embeddings.")

# Step 9: Generate adversarial examples
