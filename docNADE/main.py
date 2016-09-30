import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial import distance

def weight_variable(shape):
    '''Helper function to create a weight variable initialized with
    a normal distribution
    Parameters
    ----------
    shape : list
        Size of weight variable
    '''
    initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    '''Helper function to create a bias variable initialized with
    a constant value.
    Parameters
    ----------
    shape : list
        Size of weight variable
    '''
    initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
    return tf.Variable(initial)

with open('./train_trimmed.txt') as f:
    vectorizer = CountVectorizer()
    train_docs = f.read().splitlines()
    vectorizer.fit(train_docs)
    train = []
    for line in train_docs:
        d = map(vectorizer.vocabulary_.get, line.split())
        np.random.shuffle(d)
        train.append(d)
    train = np.array(train)
    np.random.shuffle(train)
    input_dim = len(vectorizer.vocabulary_)
    word2idx = vectorizer.vocabulary_
    idx2word = np.array(vectorizer.get_feature_names())
    n_samples = train.shape[0]

with open('./test_trimmed.txt') as f:
    test_docs = f.read().splitlines()
    test = []
    for line in test_docs:
        d = map(vectorizer.vocabulary_.get, line.split())
        np.random.shuffle(d)
        test.append(d)

    test = np.array(test)

# Parameters
h_dim = 50
batch_size = 100
learning_rate = 0.001
max_iter = 1000
voc_size = 2000

# Variables
x = tf.placeholder(tf.int32, [None])
input_dim = tf.shape(x)[0]
W = weight_variable([voc_size, h_dim])
b = bias_variable([voc_size])
c = bias_variable([h_dim])
V = weight_variable([h_dim, voc_size])

W_cum = tf.cumsum(tf.gather(W, x)) + c
H = tf.sigmoid(W_cum)
P = tf.nn.log_softmax(tf.matmul(H, V) + b)
idx_flattened = tf.range(0, input_dim) * voc_size + x
p_x_i = tf.gather(tf.reshape(P, [-1]), idx_flattened)
nll = tf.reduce_sum(-p_x_i)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(nll)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for epoch in range(max_iter):
    losses = []
    for j, doc in enumerate(train):
        _, loss = sess.run([optimizer, nll], feed_dict={x: doc})
        losses.append(loss)
        if j % 1000 == 0:
            print "Processing doc {}".format(j)
        if j % 5000 == 0:
            W_out = sess.run(W)
            w1 = W_out[word2idx['windows'], :].reshape(1, h_dim)
            w2 = W_out[word2idx['book'], :].reshape(1, h_dim)
            closest1 = distance.cdist(w1, W_out, metric='cosine')[0].argsort()
            closest2 = distance.cdist(w2, W_out, metric='cosine')[0].argsort()
            print "Closest to \'windows\'"
            print idx2word[closest1[:10]]
            print "Closest to \'book\'"
            print idx2word[closest2[:10]]

    print "Loss: {}".format(np.mean(losses))
    perps = []
    for doc in test:
        loss = sess.run(nll, feed_dict={x: doc})
        perps.append(float(loss) / len(doc))
    print "Perplexity: {}".format(np.exp(np.mean(perps)))
