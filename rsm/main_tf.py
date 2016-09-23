import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer


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

# Parameters
h_dim = 50
batch_size = 10
learning_rate = 0 # 0.0001
max_iter = 50000

with open('./data/train_trimmed.txt') as f:
    vectorizer = CountVectorizer()
    train_docs = f.read().splitlines()
    documents = vectorizer.fit_transform(train_docs).toarray()
    input_dim = len(vectorizer.vocabulary_)
    n_samples = documents.shape[0]

x = tf.placeholder(tf.float32, [batch_size, input_dim], name="input")

learning_rate /= batch_size

# Init weights
W = weight_variable([input_dim, h_dim])
a = bias_variable([h_dim])
b = bias_variable([input_dim])

D = tf.reduce_sum(x, 1)

h_1_prob = tf.sigmoid(tf.matmul(x, W) +
                      tf.matmul(tf.expand_dims(D, 0), tf.expand_dims(a, 0),
                      transpose_a=True))
unif_random = tf.random_uniform((batch_size, h_dim), minval=0, maxval=1)
h_1_sample = tf.to_float(tf.less(unif_random, h_1_prob))

x_1_prob = tf.matmul(h_1_sample, W, transpose_b=True) + b
multi_tensors = []
for i in xrange(batch_size):
    tmp_samples = tf.multinomial(tf.expand_dims(x_1_prob[i], 0), tf.to_int32(D[i]))
    multi_tensors.append(tf.reduce_sum(tf.one_hot(tmp_samples, input_dim), 1))
x_1_sample = tf.concat(0, multi_tensors)

h_2_prob = tf.sigmoid(tf.matmul(x_1_sample, W) +
                      tf.matmul(tf.expand_dims(D, 0), tf.expand_dims(a, 0),
                      transpose_a=True))

Wu = learning_rate * (tf.matmul(x, h_1_prob, transpose_a=True) -
                      tf.matmul(x_1_sample, h_2_prob, transpose_a=True))
au = learning_rate * tf.reduce_sum(h_1_prob - h_2_prob, 0)
bu = learning_rate * tf.reduce_sum(x - x_1_sample, 0)

reconstruction_error = tf.reduce_sum(tf.square(x_1_sample - x)) \
                                     / (input_dim * batch_size)
updt = [W.assign_add(Wu), a.assign_add(au), b.assign_add(bu)]

tf.stop_gradient(W)
tf.stop_gradient(a)
tf.stop_gradient(b)
tf.stop_gradient(D)
tf.stop_gradient(h_1_prob)
tf.stop_gradient(unif_random)
tf.stop_gradient(h_1_sample)
tf.stop_gradient(x_1_prob)
tf.stop_gradient(x_1_sample)
tf.stop_gradient(h_2_prob)
tf.stop_gradient(Wu)
tf.stop_gradient(au)
tf.stop_gradient(bu)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
errors = []
for step in xrange(0, max_iter):
    indices = np.random.choice(n_samples, batch_size)
    x_inp = documents[indices]

    sess.run(updt, feed_dict={x: x_inp})

    errors.append(sess.run(reconstruction_error, feed_dict={x: x_inp}))
    if step % 100 == 0:
        print "Mean reconstrction error: {}".format(np.mean(errors))
        errors = []
