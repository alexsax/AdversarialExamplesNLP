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
import argparse
import sys
import subprocess

# Step 0: Command line parsing
parser = argparse.ArgumentParser(description='Pocess the command line args.')
parser.add_argument('-m','--modelfile', default="", nargs='?',
                   help="relative path to the file which contains the model")
parser.add_argument('-g', '--generate', dest='generate', action='store_true',
                      help="Generate adversarial examples")
parser.add_argument('-a', '--use_adversarial_examples', dest='use_adversarial_examples', action='store_true',
                      help="Use adversarial examples")
parser.set_defaults(generate=False)
parser.set_defaults(use_adversarial=False)


parser.add_argument('-p', '--printlabels', dest="printlabels", action='store_true',
                   help="Print the predicted labels")
parser.set_defaults(printlabels=False)

args = parser.parse_args(sys.argv[1:])

# Step 1: Download the data.
url = 'http://mattmahoney.net/dc/'


# The model has several parameters. It allows one to load a persistent model from disk or to 
# train one from scratch. If existing_graph path is nonempty, it will load a model.
existing_graph_path = args.modelfile if args.modelfile else ""#../data/baseline10000" # Previously stored models. Set to "" to retrain from scratc
existing_auxiliary_graph_path = "../data/text8" # Where the data dictionaries are stored
generate_adversarial_examples = args.generate   # whether to generate examples
use_adversarial_examples = args.use_adversarial_examples   # whether to generate examples
print_labels = args.printlabels
num_steps = 100001                      # number of training epochs


print(use_adversarial_examples)
# Variables that control generating adversarial examples
min_number_modified = 5
step_size = 1
initial_step_value = 1

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
  words = words[0:128*100001]
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
  print("loading from pickled dictionaries in " + existing_auxiliary_graph_path)
  data = pickle.load(open(existing_auxiliary_graph_path + '-data', 'r'))
  dictionary = pickle.load(open(existing_auxiliary_graph_path + '-dictionary', 'r'))
  reverse_dictionary = pickle.load(open(existing_auxiliary_graph_path + '-reverse-dictionary', 'r'))
  # data, dictionary, reverse_dictionary = pickle.load(open(existing_auxiliary_graph_path, 'r'))
  print("loaded")# Load from a persistent location
data_index = 0





# Step 4: Function to generate a training batch for the CBOW model.
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

# Step 5: Build and train a CBOW model.
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
eta = initial_step_value #0.001 # Adversarial examples: move away from correct
eps = initial_step_value #0.001 # Adversarial examples : move towards incorrect

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


  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  loss = tf.reduce_mean(
      tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                     num_sampled, vocabulary_size))
  lossGrad = gradients.gradients(loss, embed)[0]
  new_loss = alpha*loss + (1-alpha)*tf.reduce_mean(
      tf.nn.nce_loss(nce_weights, nce_biases, embed + eta*lossGrad, train_labels,
                     num_sampled, vocabulary_size))

  results = tf.nn.softmax(tf.matmul(embed, nce_weights, transpose_b=True) + nce_biases)

  # Construct the SGD optimizer using a learning rate of 1.0.
  # optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
  # opt = tf.train.GradientDescentOptimizer(0.025)
  opt = tf.train.AdamOptimizer(0.007)
  grads_and_vars = opt.compute_gradients(new_loss)
  # grads_and_vars = opt.compute_gradients(loss)
  optimizer = opt.apply_gradients(grads_and_vars)

  print('initializing saver')
  variables_to_save = {'embeddings':embeddings, 
                        'nce_weights' : nce_weights, 
                        'nce_biases' : nce_biases}
  saver = tf.train.Saver(variables_to_save)# Construct the neural network graph






# Step 6: Begin training
with tf.Session(graph=graph) as session:
  if existing_graph_path:
    print("Starting load from " + existing_graph_path)
    saver.restore(session, existing_graph_path)
    final_embeddings = embeddings.eval()
    print("Loaded")
    print(generate_adversarial_examples)

    if generate_adversarial_examples:
      # step_size = (np.linalg.norm(np.linalg.norm(embeddings.eval(), ord=-np.inf, axis = 1), ord=1)/vocabulary_size)/10
      print(step_size)
      print("Generating examples")
      alternate = True
      while (True):
        if(alternate):
          alternate = False
          eta  += 0#step_size
        else:
          alternate = False#True
          eps += step_size
        print("eta: " + str(eta))
        print("eps: " + str(eps))
        num_examples_modified = 0

        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        batch_inputs = np.reshape(batch_inputs.T, batch_size*num_skips)
        feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}


        real_grad = lossGrad.eval(feed_dict)

        # Pick a word that we want to turn everything into
        randomLabelIndex = 0
        while True:
          randomLabelIndex = random.randint(0, len(reverse_dictionary)) # Negative sampling
          if not randomLabelIndex in batch_labels:
            break
        adversarial_labels = np.array([randomLabelIndex]*batch_size)
        adversarial_labels = np.reshape(adversarial_labels, [batch_size, 1])
        adversarial_feed_dict = {train_inputs : batch_inputs, train_labels : adversarial_labels}


        
        # adversarial_grad = np.sign(lossGrad.eval(adversarial_feed_dict))
        adversarial_grad = lossGrad.eval(adversarial_feed_dict) 


        # How to turn one word vector into another
        adversarial_perturbation =  -adversarial_grad #eta*real_grad - eps*adversarial_grad
        normalized_gradient = (adversarial_perturbation.T/np.linalg.norm(adversarial_perturbation, axis=1)).T
        print(normalized_gradient.shape)

        # Embed contexts in the vector space and add the adversarial_perturbation
        [perturbed_embeddings] = session.run([tf.stop_gradient(curr_embedding)], feed_dict=feed_dict)


        # for skip_num in xrange(num_skips):
        #   perturbed_embeddings[skip_num*batch_size:(skip_num+1)*batch_size] += adversarial_perturbation

        # a_results = tf.nn.softmax(tf.matmul(embed + adversarial_perturbation, nce_weights, transpose_b=True) + nce_biases)

        # Find most similar words to new embeddings
        adversarial_word_vectors = np.zeros([batch_size*num_skips])
        # current_embeddings = normalized_embeddings.eval() # Shape: vocab_size, embedding_size
        # p_norm = tf.sqrt(tf.reduce_sum(tf.square(perturbed_embeddings), 1, keep_dims=True))
        # normalized_p_embeddings = perturbed_embeddings / p_norm
        for example_num in xrange(batch_size):
          for skip_num in xrange(num_skips):
            diffs = embeddings.eval() - perturbed_embeddings[skip_num*batch_size + example_num, :]
            norms = np.linalg.norm(diffs, ord=2, axis=1)
            diffs = (diffs.T/norms).T

            result = np.dot(diffs, normalized_gradient[example_num,:])
            # bool_arr = np.greater(norms, eps, dtype=bool)
            # print(np.empty(vocabulary_size).fill(eps))
            result[norms > eps] = -1
            result[np.isnan(result)] = 0
            adversarial_word_vectors[skip_num*batch_size + example_num] = result.argmax() #result.argmax()
        # # peturbed_similarity_matrix = np.dot(current_embeddings, normalized_p_embeddings.eval().T) # Shape
        # peturbed_similarity_matrix = np.dot(current_embeddings, normalized_p_embeddings.eval().T) # Shape

        # # print("Finding Nearest Neighbors")
        # adversarial_word_vectors = np.zeros([batch_size*num_skips])
        # for i in xrange(batch_size):
        #   for j in xrange(num_skips):
        #     adversarial_word_vectors[j*batch_size + i] = (peturbed_similarity_matrix[:, j*batch_size + i]).argmax()

        adversarial_feed_dict = {train_inputs: adversarial_word_vectors, train_labels: adversarial_labels}


        def context_to_str(context, label="_", already_string=False):
          split_idx = int(len(context)/2)
          prior = context[0:split_idx]
          post = context[split_idx:]
          output = []
          if already_string:
            output = prior + ["_" + label + "_"] + post
          else:
            output = [reverse_dictionary[word_idx] for word_idx in prior] + ["_"+reverse_dictionary[label]+"_"] + [reverse_dictionary[word_idx] for word_idx in post]
          return " ".join(output)

        def not_all_same_word(context):
          first_word = context[0]
          for word in context:
            if word != first_word: 
              return True
          return False
        # Check to see if the context predicts the correct label, only keep contexts that do not correctly
        # predict the correct label
        # Given the adversarial feed dict, returns all the ones successfully modified
        def evaluate_successful_examples(modified_context_and_label_tuples):
          o_embed, orig_results = session.run([embed, results], feed_dict=feed_dict)
          orig_sim_to_correct = [orig_results[i, batch_labels[i]][0] for i in xrange(batch_size)]
          orig_sim_to_adv = [orig_results[i, adversarial_labels[i]][0] for i in xrange(batch_size)]

          a_embed, adv_results = session.run([embed, results], feed_dict=adversarial_feed_dict)
          sim_to_correct = [adv_results[i, batch_labels[i]][0] for i in xrange(batch_size)]
          sim_to_adv = [adv_results[i, adversarial_labels[i]][0] for i in xrange(batch_size)]


          # for i in xrange(batch_size):
          #   if(batch_inputs[i] == adversarial_word_vectors[i] and batch_inputs[i] == adversarial_word_vectors[i]): continue
          #   print()
          #   print((orig_sim_to_adv[i] < orig_sim_to_correct[i] and sim_to_adv[i] > sim_to_correct[i]))
          #   print(orig_sim_to_adv[i], "<", orig_sim_to_correct[i], " & ", sim_to_adv[i], ">", sim_to_correct[i])
          correct = [(orig_sim_to_adv[i] < orig_sim_to_correct[i] and sim_to_adv[i] > sim_to_correct[i] and 
                        not_all_same_word(modified_context_and_label_tuples[i][0]))
                        for i in xrange(batch_size)]
          return correct

        def save_and_exit_if_min_number_of_examples_generated(modified_context_and_label_tuples):
          correct = evaluate_successful_examples(modified_context_and_label_tuples)
          adversarial_examples_to_save = [modified_context_and_label_tuples[i] for i, good in enumerate(correct) if good]
          if len(adversarial_examples_to_save) >= min_number_modified:        
            print("pickling " + str(len(adversarial_examples_to_save)) + " examples")
            print("Target label: " + reverse_dictionary[randomLabelIndex])
            # for example_idx, example in enumerate(adversarial_examples_to_save): # Also save the randomly chosen target word
               # temp = list(example)
               # temp.append(reverse_dictionary[randomLabelIndex]) 
               # adversarial_examples_to_save[example_idx] = tuple(temp)
            # Check if there is an existing pickled array to save to:
            if use_adversarial_examples:
              previously_saved_examples = pickle.load(open(existing_auxiliary_graph_path + '-adversarial_examples', 'r')) 
              # print("printing previously_saved_examples:")
              # print(previously_saved_examples)
              # print("printing new examples to save")
              # print(adversarial_examples_to_save)
              adversarial_examples_to_save += previously_saved_examples
              # print("printing combination of new and old examples to save")
              print(adversarial_examples_to_save)
              print("you now have saved", len(adversarial_examples_to_save), " examples")
            pickle.dump(adversarial_examples_to_save, open(existing_auxiliary_graph_path + '-adversarial_examples', 'w+'))
            print("done pickling")
            exit()
          else:
            print("only found " + str(len(adversarial_examples_to_save)) + " valid adversarial examples, continuing...")

        def get_prediction(results_mat, i):
          return reverse_dictionary[np.argmax(results_mat[i,:])]


        def print_old_and_adversarial_contexts():
          number_of_examples_modified = 0
          modified_context_and_label_tuples = [] 
          original_results = results.eval(feed_dict)
          adversarial_results = results.eval(adversarial_feed_dict)
          for i in xrange(batch_size): # print out all the new examples
            original_words = []
            new_words = []
            label = reverse_dictionary[int(batch_labels[i])]
            for j in xrange(num_skips):
              original_words += [reverse_dictionary[int(batch_inputs[i + batch_size*j])]]
            for j in xrange(num_skips):
              new_words += [reverse_dictionary[int(adversarial_word_vectors[i + batch_size*j])]]

            if original_words != new_words and not_all_same_word(new_words):
              number_of_examples_modified += 1

              print( context_to_str(original_words, label, already_string=True) +
                      " --> " +
                      context_to_str(new_words, label, already_string=True) + 
                      " | but predicts: " +
                      get_prediction(original_results, i) + " -> " +
                      get_prediction(adversarial_results, i))
            modified_context_and_label_tuples += [(new_words, label, original_words, reverse_dictionary[randomLabelIndex])]
          if number_of_examples_modified >= min_number_modified:
            print("done")
            # print(modified_context_and_label_tuples)
            # if evaluate_number_successful_examples() > min_number_modified:
            #   exit()
            save_and_exit_if_min_number_of_examples_generated(modified_context_and_label_tuples)
            # exit()
          print("")

        print_old_and_adversarial_contexts()

        adversarial_inputs_dict = {train_inputs : adversarial_word_vectors, train_labels : adversarial_labels}



        if print_labels:
          # orig_batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)

          # batch_inputs = np.reshape(orig_batch_inputs.T, batch_size*num_skips)
          # feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

          feed_dict_results = results.eval(adversarial_feed_dict)
          print(feed_dict_results.shape)
          print(batch_labels[:,0].shape)
          correct = tf.nn.in_top_k(feed_dict_results, adversarial_labels[:,0], 10000).eval()

          def context_to_str(context, label="_"):
            prior = context[0:len(context)/2]
            post = context[len(context)/2:]
            output = [reverse_dictionary[word_idx] for word_idx in prior] + ["_"+reverse_dictionary[label]+"_"] + [reverse_dictionary[word_idx] for word_idx in post]
            " ".join(output)

          for i in xrange(128):
            if correct[i]:
              print(context_to_str(orig_batch_inputs[i], label=adversarial_labels[i,0]))
              predicted_words = [reverse_dictionary[word_idx] for word_idx in feed_dict_results[i,:].argsort()[0:100]]
              predicted_probs = sorted(feed_dict_results[i,:], reverse = True)[0:100]
              print(zip(predicted_words, predicted_probs))
          print(len([True for ex in correct if ex]))

  else: #This section is for training a new neural network
    # We must initialize all variables before we use them.
    adversarial_examples = {}
    if use_adversarial_examples:
      #open up the adversarial example dictionary if available
      adversarial_examples = pickle.load(open(existing_auxiliary_graph_path + '-adversarial_examples', 'r')) 
      print("printing adversarial_examples from storage")
      print(adversarial_examples)
    else: 
      print("not loading adversarial_examples")
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
      _, loss_val = session.run([optimizer, new_loss], feed_dict=feed_dict)
      average_loss += loss_val

      if step % 500 == 0:
        if step > 0:
          average_loss = average_loss / 500
          # The average loss is an estimate of the loss over the last 2000 batches.
        print("Average loss at step ", step, " before: ", average_loss)
        average_loss = 0

      if adversarial_examples != {} and step % 500 == 0:
          # generate new adversarial examples
          print("start subprocess to generate adversarial examples")
          os.remove('../data/text8-adversarial_examples')
          subprocess.call('make_examples.sh "../data/baseline-100000"', shell=True)
          print("finished running subproccess to generate adversarial examples")

          print("batch size is: ", batch_size)
          for example_num in range(batch_size):
            context, label, _, _ = adversarial_examples[example_num % len(adversarial_examples)] # Wrap around for the labels
            for context_word_num, context_word in enumerate(context):
              if example_num + context_word_num*batch_size >= batch_size:
                break
              batch_inputs[example_num + context_word_num*batch_size] = dictionary[context_word]
              # print(example_num + context_word_num*batch_size)
              batch_labels[example_num + context_word_num*batch_size] = dictionary[label] # I don't trust this
            feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}
            _, loss_val = session.run([optimizer, new_loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step > 0:
              print("Average loss at step ", step, " after adversarial_examples: ", average_loss)
              average_loss = 0


      # note that this is expensive (~20% slowdown if computed every 500 steps)
      if step % 1000 == 0:
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
        basename_os = '../data/'
        filename_os = 'sample-test'
        new_folder = 'example/'
        if use_adversarial_examples:
          saver.save(session, basename_os+filename_os, global_step=step)
        else:
          saver.save(session, basename_os+filename_os, global_step=step)
        
        # Move the file to a safe folder
        extension = '-'+str(step)
        os.rename(basename_os+filename_os+extension, basename_os+new_folder+filename_os+extension)
          
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

