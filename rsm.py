from __future__ import print_function
from common import batch
import numpy as np
from scipy.spatial import distance
import pickle
from scipy.misc import logsumexp


class RSM():
    def __init__(self, input_dim, word2idx=None, idx2word=None, h_dim=50):
        """ Initiate parameters and weights for the Replicated Softmax.

        Parameters
        ----------
        input_dim : Dimension of input vector.
        word2idx : Dict with mappings from word to its' index in the input
                   vector.
        idx2word : Array where each index corresponds to a word.
        h_dim : Dimension of hidden unit.

        Notes
        -----
        Based on:
        Hinton, G. E., and Salakhutdinov, R. R. Replicated softmax: an undi-
        rected topic model.
        """

        # Parameters
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.h_dim = h_dim
        self.input_dim = input_dim

        # Init weights
        self.W = 0.001 * np.random.randn(input_dim, self.h_dim)
        self.a = 0.001 * np.random.randn(self.h_dim)
        self.b = 0.001 * np.random.randn(input_dim)
        self.W_upd = np.zeros((input_dim, self.h_dim))
        self.a_upd = np.zeros(self.h_dim)
        self.b_upd = np.zeros(input_dim)

    def train(self, train, learning_rate=0.001, batch_size=100,
              max_iter=1000, momentum=0.9):
        """ Start training the RSM. Uses Contrastive Divergece with 1 sampling
        step to approximate the gradients.

        Parameters
        ----------
        train : Matrix of training data.
        learning_rate : Learning rate for updating paramters.
        batch_size : Size of each training batch.
        max_iter : Maximum number of iterations in training.
        momentum : The momentum used in each update.
        """

        learning_rate /= batch_size
        reconstruction_error = []
        for epoch in range(0, max_iter):
            print("--- Epoch:", epoch)
            reconstruction_error = []
            for i, x in enumerate(batch(train, batch_size)):
                D = x.sum(axis=1)
                v_batch_size = x.shape[0]

                # 1. Gibbs sample 1 time
                # - from p(h|x)
                h_1_prob = self._sigmoid(x.dot(self.W)
                                         + np.outer(D, self.a))
                h_1_sample = np.array(np.random.rand(v_batch_size, self.h_dim)
                                      < h_1_prob, dtype=int)

                # - from p(x|h)
                x_1_prob = np.exp(np.dot(h_1_sample, self.W.T) + self.b)
                x_1_prob /= x_1_prob.sum(axis=1).reshape(v_batch_size, 1)
                x_1_sample = np.zeros(x_1_prob.shape)
                for j in range(v_batch_size):
                    x_1_sample[j] = np.random.multinomial(D[j], x_1_prob[j])

                # - compute p(h|x)
                h_2_prob = self._sigmoid(np.dot(x_1_sample, self.W)
                                         + np.outer(D, self.a))

                # 2. Update parameters
                self.W_upd *= momentum
                self.W_upd += x.transpose().dot(h_1_prob)\
                    - np.dot(x_1_sample.T, h_2_prob)

                self.a_upd *= momentum
                self.a_upd = h_1_prob.sum(axis=0) - h_2_prob.sum(axis=0)

                self.b_upd *= momentum
                self.b_upd = x.sum(axis=0).A1 - x_1_sample.sum(axis=0)

                self.W += learning_rate * self.W_upd
                self.a += learning_rate * self.a_upd
                self.b += learning_rate * self.b_upd

                reconstruction_error.append(np.linalg.norm(x_1_sample - x)**2
                                            / (self.input_dim*v_batch_size))

                if i % 100 == 0:
                    print("Step: {}".format(i))

            print("Mean reconstrction error: {}"
                  .format(np.mean(reconstruction_error)))

            # Find closest word
            # print(self.closest_word("weapons"))
            # print(self.closest_word("books"))

            model = {'W': self.W,
                     'a': self.a,
                     'b': self.b}
            with open('checkpoints/rsm.ckpt', 'w') as f:
                pickle.dump(model, f)

    def restore(self, path):
        """ Restores a previous model from path.

        Parameters
        ----------
        path : Path to the stored model.
        """

        with open(path, 'rb') as f:
            model = pickle.load(f)
            self.W = model['W']
            self.a = model['a']
            self.b = model['b']

    def perplexity(self, data, M=100, steps=1000):
        """ Calculate perplexity by estimating the partition function
        using Annealed Importance Sampling (AIS).

        Parameters
        ----------
        data : Matrix with rows of unseen data
        M : Number of separate runs.
        steps : Number of intermediate distributions

        Notes
        -----
        Implemented following the description in:
        Salakhutdinov, R. R. and Murray, I. On the Quantitative Analysis of
        Deep Belief Networks.
        """

        log_ps = []
        for count, i in enumerate(np.random.permutation(data.shape[0])[:M]):
            x = data[i]
            D = x.sum()
            b_k = np.linspace(0.0, 1.0, steps)
            steps = len(b_k)
            lw = np.zeros(M)

            # Uniform initial distribution
            x_1_prob = np.ones(self.input_dim) / self.input_dim

            # Sample visible v_1:
            x_1_sample = np.zeros((M, self.input_dim))
            for j in range(M):
                x_1_sample[j, :] = np.random.multinomial(D, x_1_prob, size=1)

            def p_k(b_k, v):
                return np.dot(v, self.b)*b_k\
                       + np.log1p(np.exp(b_k*(np.dot(v, self.W) + D*self.a)))\
                       .sum(axis=1)

            for s in range(0, steps-1):

                # Add p_(k+1)(v_k) and subtract p_k(v_k)
                lw += p_k(b_k[s+1], x_1_sample)
                lw -= p_k(b_k[s], x_1_sample)

                # Sample hidden from p_k(h | x)
                h_1_prob = self._sigmoid(b_k[s+1] * (np.dot(x_1_sample, self.W)
                                                     + D*self.a))
                h_1_sample = np.array(np.random.rand(M, self.h_dim)
                                      < h_1_prob, dtype=int)

                # Sample visible v_(k+1) from p_k(x | h)
                x_1_prob = np.exp(b_k[s+1] * (np.dot(h_1_sample, self.W.T)
                                              + self.b) + self.b)
                x_1_prob /= x_1_prob.sum(axis=1).reshape(M, 1)
                for j in range(M):
                    x_1_sample[j, :] = np.random.multinomial(D, x_1_prob[j],
                                                             size=1)

            logZ = logsumexp(lw) - np.log(M)
            logZ += self.h_dim * np.log(2) + D * np.log(self.input_dim)

            p = np.log1p(np.exp(np.dot(x.T, self.W) + D*self.a)).sum()\
                + np.dot(x.T, self.b) - logZ
            log_ps.append(float(p)/D)
            if count % 10 == 0:
                print("AIS trial: {}".format(count))

        return np.exp(-np.mean(log_ps))

    def closest_word(self, word, n=10):
        if self.word2idx is None or self.idx2word is None:
            return "No word to index mapping provided"
        w = self.W[self.word2idx[word], :].reshape(1, self.h_dim)
        closest = distance.cdist(w, self.W, metric='cosine')[0].argsort()
        return self.idx2word[closest[:10]]

    def get_representation(self, data):
        D = data.sum(axis=1)
        return self._sigmoid(data.dot(self.W) + np.outer(D, self.a))

    def _sigmoid(self, x):
        return 1/(1 + np.exp(-x))
