import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial import distance

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def batch(iterable, n=1):
    '''Generate batches of size n unless we're at the end of the collection'''
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

# Parameters
h_dim = 50
batch_size = 100
learning_rate = 0.001
max_iter = 50000

with open('./data/train_trimmed.txt') as f:
    vectorizer = CountVectorizer()
    train_docs = f.read().splitlines()
    documents = vectorizer.fit_transform(train_docs).toarray()
    input_dim = len(vectorizer.vocabulary_)
    word2idx = vectorizer.vocabulary_
    idx2word = np.array(vectorizer.get_feature_names())
    n_samples = documents.shape[0]

# Init weights
W = 0.001 * np.random.randn(input_dim, h_dim)
a = 0.001 * np.random.randn(h_dim)
b = 0.001 * np.random.randn(input_dim)

learning_rate /= batch_size

reconstruction_error = []
for epoch in xrange(0, max_iter):
    print("--- Epoch:", epoch)

    reconstruction_error = []
    for i, x in enumerate(batch(documents, batch_size)):
        D = x.sum(axis=1)
        v_batch_size = x.shape[0]
        # 1. Gibbs sample 1 time
        # - from p(h|x)
        h_1_prob = sigmoid(np.dot(x, W) + np.outer(D, a))
        h_1_sample = np.array(np.random.rand(v_batch_size, h_dim) < h_1_prob,
                              dtype=int)

        # - from p(x|h)
        x_1_prob = np.exp(np.dot(h_1_sample, W.T) + b)
        x_1_prob /= x_1_prob.sum(axis=1).reshape(v_batch_size, 1)
        x_1_sample = np.zeros(x_1_prob.shape)
        for i in xrange(v_batch_size):
            x_1_sample[i] = np.random.multinomial(D[i], x_1_prob[i], size=1)

        # - compute p(h|x)
        h_2_prob = sigmoid(np.dot(x_1_sample, W) + np.outer(D, a))

        # 2. Update parameters
        W += learning_rate * (np.dot(x.T, h_1_prob) - np.dot(x_1_sample.T, h_2_prob))
        a += learning_rate * (h_1_prob.sum(axis=0) - h_2_prob.sum(axis=0))
        b += learning_rate * (x.sum(axis=0) - x_1_sample.sum(axis=0))

        reconstruction_error.append(np.linalg.norm(x_1_sample - x)**2
                                    / (input_dim*v_batch_size))

    print "Mean reconstrction error: {}".format(np.mean(reconstruction_error))
    # Find closest word
    # w = W[word2idx['terrorist'], :].reshape(1, h_dim)
    # closest = distance.cdist(w, W, metric='cosine')[0].argsort()
    # print idx2word[closest[:10]]
