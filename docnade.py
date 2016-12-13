from __future__ import print_function
from common import batch, bias_variable, weight_variable, feed_from_sparse
import tensorflow as tf
import numpy as np
from scipy.spatial import distance


class DocNADE():
    def __init__(self, voc_size, word2idx=None, idx2word=None, h_dim=50):
        """ Initate Tensorflow graph for DocNADE.

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
        Larochelle, H., and Lauly, S. A neural autoregressive topic model.
        """

        # Parameters
        self.voc_size = voc_size
        self.h_dim = h_dim
        self.word2idx = word2idx
        self.idx2word = idx2word

        self.x = tf.placeholder(tf.int32, [None], name="input")

        # Variables
        self.W = weight_variable([voc_size, h_dim], name="W")
        b = bias_variable([voc_size], name="b")
        c = bias_variable([h_dim], name="c")
        V = weight_variable([h_dim, voc_size], name="V")

        # Flow
        W_cum = tf.pad(tf.cumsum(tf.gather(self.W, self.x[:-1])),
                       [[1, 0], [0, 0]])
        W_cum += c
        H = tf.sigmoid(W_cum)
        softmax = tf.nn.sparse_softmax_cross_entropy_with_logits
        p_x_i = softmax(tf.matmul(H, V) + b, self.x)
        self.nll = tf.reduce_sum(p_x_i)

        # Tensor for getting the representation of a document
        self.x_batch = tf.sparse_placeholder(tf.float32)
        x = tf.sparse_tensor_to_dense(self.x_batch, validate_indices=False)
        self.rep = tf.sigmoid(tf.matmul(x, self.W) + c)

        self.sess = tf.Session()
        self.saver = tf.train.Saver()

    def train(self, train, test, max_iter=1000, learning_rate=0.001):
        """ Train the model using ADAM optimizer.

        Parameters
        ----------
        train : Matrix of training data.
        test : Matrix of testing data.
        max_iter : Maximum number of epochs in training.
        learning_rate : Learning rate for updating paramters.
        """
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)\
            .minimize(self.nll)

        best = np.inf
        self.sess.run(tf.initialize_all_variables())
        for epoch in range(max_iter):
            losses = []
            for j, doc in enumerate(train):
                _, loss = self.sess.run([optimizer, self.nll],
                                        feed_dict={self.x: doc})
                losses.append(loss)
                if j % 1000 == 0:
                    print("Processing doc {}".format(j))
                if j % 5000 == 0:
                    print("Closest to \"weapons\":")
                    print(self.closest_words("weapons"))

                    print("Closest to \"books\":")
                    print(self.closest_words("books"))

            print("Loss: {}".format(np.mean(losses)))
            perplexity = self.perplexity(test)
            print("Perplexity: {}".format(perplexity))
            if perplexity < best:
                self.saver.save(self.sess, "checkpoints/docnade_p={}.ckpt"
                                           .format(int(perplexity)))
                best = perplexity

    def restore(self, path):
        """ Restores a previous model from path.

        Parameters
        ----------
        path : Path to the stored model.
        """
        self.saver.restore(self.sess, path)

    def perplexity(self, data):
        """ Calculate perplexity using 1 or more samples of the hidden unit.

        Parameters
        ----------
        data : Matrix of data to calculate perplexity on.
        """

        perps = []
        for doc in data:
            if len(doc) == 0:
                continue
            loss = self.sess.run(self.nll, feed_dict={self.x: doc})
            perps.append(loss/len(doc))
        return np.exp(np.mean(perps))

    def closest_words(self, word, n=10):
        if self.word2idx is None or self.idx2word is None:
            return "No word to index mappings provided"
        W_out = self.sess.run(self.W)
        w = W_out[self.word2idx[word], :]\
            .reshape(1, self.h_dim)
        closest = distance.cdist(w, W_out, metric='cosine')[0].argsort()
        return self.idx2word[closest[:n]]

    def get_representation(self, data):
        reps = []
        for b in batch(data, 10000):
            feed = feed_from_sparse(b, self.x_batch)
            reps.append(self.sess.run(self.rep, feed))
        return np.vstack(reps)
