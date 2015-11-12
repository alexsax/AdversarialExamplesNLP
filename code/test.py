import numpy as np
import os
final_embeddings = np.array([[1.*10**(-3), 2., 3.],[4., 5., 6.]])
reverse_dictionary = ["the", "of"]
embedding_size = len(final_embeddings[0])

filename = 'vectors.txt'

def save_to_file(embedding_size, filename='vectors.txt'):
	with open(filename, 'w') as f:
		f.write(str(len(final_embeddings)) + " " + str(embedding_size) + "\n")
		for i, row in enumerate(final_embeddings):
			rowstring = ' '.join([reverse_dictionary[i]] + ['%.6f' % num for num in row] + ["\n"])
			f.write(rowstring)

save_to_file(embedding_size, filename)
os.system('cat '+ filename)