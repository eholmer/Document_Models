import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial import distance
import pickle
from scipy.misc import logsumexp

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
learning_rate = 0.0001
max_iter = 1000
momentum = 0.9

# Load data
with open('./data/train_trimmed.txt') as f:
    vectorizer = CountVectorizer()
    train_docs = f.read().splitlines()
    train = vectorizer.fit_transform(train_docs).toarray()
    np.random.shuffle(train)
    input_dim = len(vectorizer.vocabulary_)
    word2idx = vectorizer.vocabulary_
    idx2word = np.array(vectorizer.get_feature_names())
    n_samples = train.shape[0]

with open('./data/test_trimmed.txt') as f:
    test_docs = f.read().splitlines()
    test = vectorizer.transform(test_docs).toarray()
    test = test[test.sum(axis=1) != 0]
    np.random.shuffle(test)


# Init weights
W = 0.001 * np.random.randn(input_dim, h_dim)
a = 0.001 * np.random.randn(h_dim)
b = 0.001 * np.random.randn(input_dim)

W_upd = np.zeros((input_dim, h_dim))
a_upd = np.zeros(h_dim)
b_upd = np.zeros(input_dim)

learning_rate /= batch_size


reconstruction_error = []
for epoch in xrange(0, max_iter):
    print("--- Epoch:", epoch)

    reconstruction_error = []
    for i, x in enumerate(batch(train, batch_size)):
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
        W_upd = W_upd*momentum + np.dot(x.T, h_1_prob) - \
                                 np.dot(x_1_sample.T, h_2_prob)
        a_upd = a_upd*momentum + h_1_prob.sum(axis=0) - h_2_prob.sum(axis=0)
        b_upd = b_upd*momentum + x.sum(axis=0) - x_1_sample.sum(axis=0)

        W += learning_rate * W_upd
        a += learning_rate * a_upd
        b += learning_rate * b_upd

        reconstruction_error.append(np.linalg.norm(x_1_sample - x)**2
                                    / (input_dim*v_batch_size))

    print "Mean reconstrction error: {}".format(np.mean(reconstruction_error))

    # Find closest word
    w = W[word2idx['book'], :].reshape(1, h_dim)
    closest = distance.cdist(w, W, metric='cosine')[0].argsort()
    print idx2word[closest[:10]]

# Save model------------------------------------
model = {'W': W,
'a': a,
'b': b}

with open('models/model.mdl', 'wb') as file:
    pickle.dump(model, file)

with open('models/model.mdl', 'rb') as file:
    model = pickle.load(file)
    W = model['W']
    a = model['a']
    b = model['b']

# Compute perplexity:
M = 100
steps = 1000
log_ps = []
for i in np.random.permutation(test.shape[0])[:50]:
    x = test[i]
    D = x.sum()
    f = 0

    b_k = np.linspace(0.0, 1.0, steps)

    steps = len(b_k)
    lw = np.zeros(M)

    # Uniform initial distribution
    x_1_prob = np.ones(input_dim) / input_dim

    # Sample visible v_1:
    x_1_sample = np.zeros((M, input_dim))
    for i in xrange(M):
        x_1_sample[i, :] = np.random.multinomial(D, x_1_prob, size=1)

    def p_k(b_k, v):
        return np.dot(x_1_sample, b)*(1-f) + \
               np.log1p(np.exp(b_k*(np.dot(v, W) + D*a))).sum(axis=1)

    for s in xrange(0, steps-1):

        # Add p_(k+1)(v_k) and subtract p_k(v_k)
        lw += p_k(b_k[s+1], x_1_sample)
        lw -= p_k(b_k[s], x_1_sample)

        # Sample hidden from p_k(h | x)
        h_1_prob = sigmoid(b_k[s+1] * (np.dot(x_1_sample, W) + D*a))
        h_1_sample = np.array(np.random.rand(M,  h_dim) < h_1_prob, dtype=int)

        # Sample visible v_(k+1) from p_k(x | h)
        x_1_prob = np.exp(b_k[s+1] * (np.dot(h_1_sample, W.T) + b) +
                          b*f*(1-b_k[s+1]))
        x_1_prob /= x_1_prob.sum(axis=1).reshape(M, 1)
        for i in xrange(M):
            x_1_sample[i, :] = np.random.multinomial(D, x_1_prob[i], size=1)

    logZ = logsumexp(lw) - np.log(M)
    logZ += h_dim * np.log(2) + D * logsumexp(f*b)

    p = np.log1p(np.exp(np.dot(x.T, W) + D*a)).sum() + np.dot(x.T, b) - logZ
    log_ps.append(float(p)/D)

print np.exp(-np.mean(log_ps))
