import tensorflow as tf
import numpy as np
import time
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial import distance

flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of adam optimizer [0.001]")
flags.DEFINE_float("decay_rate", 0.96, "Decay rate of learning rate [0.96]")
flags.DEFINE_float("decay_step", 10000, "# of decay step for learning rate decaying [10000]")
flags.DEFINE_integer("max_iter", 450000, "Maximum of iteration [450000]")
flags.DEFINE_integer("h_dim", 50, "The dimension of latent variable [50]")
flags.DEFINE_integer("embed_dim", 500, "The dimension of word embeddings [500]")
FLAGS = flags.FLAGS


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

def batch(iterable, n=1):
    '''Generate batches of size n unless we're at the end of the collection'''
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


np.set_printoptions(threshold=np.inf)

# Define parameters
alternating = True
learning_rate = FLAGS.learning_rate
step = tf.Variable(0, trainable=False)
max_iter = FLAGS.max_iter
h_dim = FLAGS.h_dim
embed_dim = FLAGS.embed_dim
batch_size = 100

with open('./data/train_trimmed.txt') as f:
    vectorizer = CountVectorizer()
    train_docs = f.read().splitlines()
    documents = vectorizer.fit_transform(train_docs).toarray()
    np.random.shuffle(documents)
    word2idx = vectorizer.vocabulary_
    idx2word = np.array(vectorizer.get_feature_names())
    input_dim = len(vectorizer.vocabulary_)
    n_samples = documents.shape[0]


# Build model
x = tf.placeholder(tf.float32, [None, input_dim], name="input")
v_batch_size = tf.shape(x)[0]
# Encoder -----------------------------------
with tf.variable_scope("Encoder"):
    W1 = weight_variable([input_dim, embed_dim])
    b1 = bias_variable([embed_dim])
    l1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    W2 = weight_variable([embed_dim, embed_dim])
    b2 = bias_variable([embed_dim])
    l2 = tf.nn.relu(tf.matmul(l1, W2) + b2)

    W_mu = weight_variable([embed_dim, h_dim])
    b_mu = bias_variable([h_dim])
    W_sigma = weight_variable([embed_dim, h_dim])
    b_sigma = bias_variable([h_dim])

    mu = tf.matmul(l2, W_mu) + b_mu
    log_sigma_sq = tf.matmul(l2, W_sigma) + b_sigma
    eps = tf.random_normal((v_batch_size, h_dim), 0, 1, dtype=tf.float32)

    sigma = tf.sqrt(tf.exp(log_sigma_sq))
    h = mu + sigma*eps

# Generator -------------------------------------
with tf.variable_scope("Generator"):
    R = weight_variable([input_dim, h_dim])
    b = bias_variable([input_dim])
    E = -tf.matmul(h, R, transpose_b=True) - b
    p_x_i = tf.nn.log_softmax(E)

# Optimizer ---------------------------------------

# d_kl(q(z|x)||p(z))
# 0.5*sum(1 + log(sigma^2) - mu^2 - sigma^2)
encoder_loss = -0.5 * (tf.reduce_sum(1 + log_sigma_sq - tf.square(mu) -
                                     tf.exp(log_sigma_sq), 1))
# p(X|h)
# sum(log(p_x_i)) for x_i in document
generator_loss = -tf.reduce_sum(x * p_x_i, 1)

total_loss = tf.reduce_mean(encoder_loss + generator_loss)

tf.scalar_summary('Encoder loss', tf.reduce_mean(encoder_loss))
tf.scalar_summary('Generator loss', tf.reduce_mean(generator_loss))
tf.scalar_summary('Total loss', total_loss)

encoder_var_list, generator_var_list = [], []
for var in tf.trainable_variables():
    if "Encoder" in var.name:
        encoder_var_list.append(var)
    elif "Generator" in var.name:
        generator_var_list.append(var)

e_optimizer = tf.train.AdamOptimizer(learning_rate) \
              .minimize(total_loss, var_list=encoder_var_list)

g_optimizer = tf.train.AdamOptimizer(learning_rate) \
              .minimize(total_loss, var_list=generator_var_list)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

# Train ---------------------------------------------
sess = tf.Session()
sess.run(tf.initialize_all_variables())

merged_sum = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("./logs/", sess.graph)

start_time = time.time()

for epoch in xrange(0, max_iter):
    print("--- Epoch:", epoch)
    losses = []
    for i, batch_i in enumerate(batch(documents, batch_size)):

        if alternating:
            _, eloss = sess.run([e_optimizer, encoder_loss],
                                feed_dict={x: batch_i})

            _, gloss, loss = sess.run([g_optimizer, generator_loss, total_loss],
                                      feed_dict={x: batch_i})
        else:
            _, eloss, loss = sess.run([optimizer, encoder_loss, total_loss],
                                      feed_dict={x: batch_i})

        losses.append(loss)
        if i % 2 == 0:
            summary = sess.run(merged_sum, feed_dict={x: batch_i})
            writer.add_summary(summary, epoch)
        if i % 10 == 0:
            # print "Encoder_loss: {}".format(eloss)
            print "Step: {}, time: {}, loss: {}" \
                .format(i, time.time() - start_time,
                        loss)
    print('--- Avg loss:', np.mean(losses))

    # Find closest words
    R_out = sess.run(R)
    w = R_out[word2idx['jews'],:].reshape(1, h_dim)
    closest = distance.cdist(w, R_out, metric='cosine')[0].argsort()
    print idx2word[closest[:10]]
