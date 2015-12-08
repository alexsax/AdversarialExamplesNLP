import numpy as np
batch_size = 5
embedding_size = 3
num_skips = 2

original = np.zeros((5*2,3), dtype=object)
for batch_k in xrange(batch_size):
	for skip_j in xrange(num_skips):
		for dim_i in xrange(embedding_size):
			original[batch_k + batch_size*skip_j, dim_i] = "example" + str(batch_k) + ":skip" + str(skip_j) + ":dim" + str(dim_i)

arr = np.zeros((num_skips , batch_size, embedding_size), dtype=object)
arr = np.reshape(original, [num_skips, batch_size, embedding_size ])

# for dim_i in xrange(dim):
# 	for skip_j in xrange(num_skips):
# 		for batch_k in xrange(batch_size):
# 			arr[batch_k, skip_j, dim_i] = "example" + str(batch_k) + ":skip" + str(skip_j) + ":batch" + str(dim_i)

# arr = np.zeros((5,2,3), dtype=object)
# for dim_i in xrange(dim):
# 	for skip_j in xrange(num_skips):
# 		for batch_k in xrange(batch_size):
# 			arr[batch_k, skip_j, dim_i] = "example" + str(batch_k) + ":skip" + str(skip_j) + ":batch" + str(dim_i)
print original
print "\n\n"
for dim_i in xrange(embedding_size):
	for skip_j in xrange(num_skips):
		for batch_k in xrange(batch_size):
			print "example" + str(batch_k) + ":skip" + str(skip_j) + ":dim" + str(dim_i), arr[skip_j, batch_k, dim_i ]
arr = np.sum(arr, 0)
print arr
print arr.shape