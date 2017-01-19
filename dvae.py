from __future__ import print_function
from common import batch, bias_variable, weight_variable, feed_from_sparse
import tensorflow as tf
import numpy as np
from scipy.spatial import distance
from scipy.misc import logsumexp


class DVAE():
    def __init__(self, voc_size, word2idx=None, idx2word=None, h_dim=50,
                 activation='sigmoid'):
        """ Initate Tensorflow graph for DeepDocNADE.

        Parameters
        ----------
        voc_size : Vocabulary size.
        word2idx : Dict with mappings from word to its' index in the input
                   vector.
        idx2word : Array where each index corresponds to a word.
        h_dim : Dimension of hidden units.

        Notes
        -----
        Based on:
        Lauly, S., Zheng, Y., Allauzen, A., and Larochelle, H.
        Document Neural Autoregressive Distribution Estimation
        """

        # Parameters
        self.voc_size = voc_size
        self.h_dim = h_dim
        self.word2idx = word2idx
        self.idx2word = idx2word
        activations = {'sigmoid': tf.sigmoid,
                       'tanh': tf.tanh}
        activ = activations[activation]

        self.x_s = tf.sparse_placeholder(tf.float32, name="Input")
        x = tf.sparse_tensor_to_dense(self.x_s, validate_indices=False)

        # Small hack to avoid all words ending up in "low". Solved by
        # uniformly sampling one word for "high" to begin with and then
        # proceeed as usual.
        high = tf.one_hot(tf.squeeze(tf.multinomial(tf.log(x), 1)), voc_size)
        nx = x - high
        low = tf.floor(tf.random_uniform(tf.shape(nx)) * (nx + 1))
        high = x - low
        Dv = tf.reduce_sum(x, 1)
        i = tf.reduce_sum(low, 1)

        # Testing
        self.x_test = tf.placeholder(tf.int32, shape=[None])

        with tf.variable_scope("Encoder"):
            # Variables
            self.W0 = weight_variable([voc_size, 10*h_dim], name="W0")
            b0 = bias_variable([10*h_dim], name="b0")
            W_mu = weight_variable([10*h_dim, h_dim], name="W_mu")
            b_mu = bias_variable([h_dim], name="b_mu")
            W_sigma = weight_variable([10*h_dim, h_dim], name="W_sigma")
            b_sigma = bias_variable([h_dim], name="b_sigma")

            l1 = activ(tf.matmul(low, self.W0) + b0)
            self.mu = tf.matmul(l1, W_mu) + b_mu
            log_sigma_sq = tf.matmul(l1, W_sigma) + b_sigma
            self.sigma = tf.sqrt(tf.exp(log_sigma_sq))
            eps = tf.random_normal((tf.shape(low)[0], h_dim), 0, 1,
                                   dtype=tf.float32)

            h = self.mu + self.sigma*eps

        with tf.variable_scope("Decoder"):
            V = weight_variable([h_dim, voc_size], name="V")
            b = bias_variable([voc_size], name="b")
            p_x_i = tf.nn.log_softmax(tf.matmul(h, V) + b)

        self.encoder_loss = -0.5 * (tf.reduce_sum(1 + log_sigma_sq -
                                                  tf.square(self.mu) - 
                                                  tf.exp(log_sigma_sq), 1))
        factor = 1./(Dv - i)
        nll = tf.reduce_sum(-p_x_i * high, 1)
        self.generator_loss = Dv*factor*nll
        self.total_loss = tf.reduce_mean(self.encoder_loss +
                                         self.generator_loss)
        self.perp = tf.exp(tf.reduce_mean((self.encoder_loss +
                                           self.generator_loss) / Dv))
            
        # Test Flow
        W_cum = tf.pad(tf.cumsum(tf.gather(self.W0, self.x_test[:-1])),
                       [[1, 0], [0, 0]])
        W_cum += b0
        L1 = activ(W_cum)
        MU = tf.matmul(L1, W_mu) + b_mu
        LSS = tf.matmul(L1, W_sigma) + b_sigma
        SIGMA = tf.sqrt(tf.exp(LSS))
        EPS = tf.random_normal(tf.shape(W_cum), 0, 1, dtype=tf.float32)
        H = MU + SIGMA*EPS
        softmax = tf.nn.sparse_softmax_cross_entropy_with_logits
        P_X_I = softmax(tf.matmul(H, V) + b, self.x_test)
        E_loss = -0.5 * (tf.reduce_sum(1 + LSS -
                                       tf.square(MU) +
                                       tf.exp(LSS), 1))
        self.nll_test = tf.reduce_sum(P_X_I) + tf.reduce_mean(E_loss)

        # Representation of Documents
        # self.rep = activ(tf.matmul(activ(tf.matmul(x, self.W) + c), W1) + c1)
        # self.rep = activ(tf.matmul(x, self.W) + c)

        self.sess = tf.Session()
        self.saver = tf.train.Saver()

    def train(self, train, test, max_iter=1000, learning_rate=0.001):
        """ Train the model using ADAM optimizer.

        Parameters
        ----------
        train : Matrix of training data.
        test : Matrix of testing data.
        max_iter : Maximum number of iterations in training.
        learning_rate : Learning rate for updating paramters.
        """
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)\
            .minimize(self.total_loss)

        best = np.inf
        self.sess.run(tf.initialize_all_variables())
        for epoch in range(max_iter):
            losses = []
            print("-------Epoch: {}".format(epoch))
            for j, doc in enumerate(batch(train, 100)):
                # for j, doc in enumerate(train):
                feed = feed_from_sparse(doc, self.x_s)
                _, loss = self.sess.run([optimizer, self.total_loss],
                                        feed_dict=feed)
                losses.append(loss)

            print("Loss: {}".format(np.mean(losses)))
            # perplexity = self.perplexity(test, False)
            # print("Perplexity: {}".format(perplexity))
            # if perplexity < best:
            #     self.saver.save(self.sess, "checkpoints/deep_docnade.ckpt")
            #     best = perplexity
            # print("Closest to \"weapons\":")
            # print(self.closest_words("weapons"))

            # print("Closest to \"books\":")
            # print(self.closest_words("books"))

    def restore(self, path):
        """ Restores a previous model from path.

        Parameters
        ----------
        path : Path to the stored model.
        """
        self.saver.restore(self.sess, path)

    def perplexity(self, data, test=False, ensembles=1):
        """ Calculate perplexity using 1 or more samples of the hidden unit.

        Parameters
        ----------
        data : Matrix of data to calculate perplexity on.
        test : Whether or not to perform exact perplexity calculation.
        ensembles : Number of ensembles to use for perplexity calculation.
        """
        if test:
            perps = []
            for doc in data:
                if len(doc) == 0:
                    continue
                losses = []
                for i in range(ensembles):
                    np.random.shuffle(doc)
                    loss = self.sess.run(self.nll_test,
                                         feed_dict={self.x_test: doc})
                    losses.append(-loss)
                ensemble_loss = -1*(logsumexp(losses)
                                    - np.log(ensembles))*1./len(doc)
                perps.append(ensemble_loss)
            return np.exp(np.mean(perps))
        else:
            feed = feed_from_sparse(data, self.x_s)
            return self.sess.run(self.perp, feed_dict=feed)

    def closest_words(self, word, n=10):
        if self.word2idx is None or self.idx2word is None:
            return "No word to index mappings provided"
        W_out = self.sess.run(self.W)
        w = W_out[self.word2idx[word], :].reshape(1, self.h_dim)
        closest = distance.cdist(w, W_out, metric='cosine')[0].argsort()
        return self.idx2word[closest[:n]]

    def get_representation(self, data):
        feed = feed_from_sparse(data, self.x_s)
        return self.sess.run(self.rep, feed_dict=feed)
