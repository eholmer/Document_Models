from __future__ import print_function
from common import batch, bias_variable, weight_variable, feed_from_sparse
import tensorflow as tf
import numpy as np
from scipy.spatial import distance


class NVDM():
    def __init__(self, input_dim, word2idx=None, idx2word=None, h_dim=50,
                 embed_dim=500):
        """ Initate Tensorflow graph for the Neural Variational Document Model.

        Parameters
        ----------
        input_dim : Dimension of input data.
        word2idx : Dict with mappings from word to its' index in the input
        vector.
        idx2word : Array where each index corresponds to a word.
        h_dim : Dimension of hidden unit.
        embed_dim : Dimension of embedding layers.

        Notes
        -----
        Based on:
        Miao, Y., Yu, L., and Blunsom, P. Neural variational inference for text
        processing.
        """
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.h_dim = h_dim
        self.x_s = tf.sparse_placeholder(tf.float32, name="input")
        v_batch_size = tf.shape(self.x_s)[0]
        x = tf.sparse_tensor_to_dense(self.x_s, validate_indices=False)

        # Encoder -----------------------------------
        with tf.variable_scope("Encoder"):
            W1 = weight_variable([input_dim, embed_dim], name="W1")
            b1 = bias_variable([embed_dim], name="b1")
            l1 = tf.nn.relu(tf.matmul(x, W1) + b1)
            W2 = weight_variable([embed_dim, embed_dim], name="W2")
            b2 = bias_variable([embed_dim], name="b2")
            l2 = tf.nn.relu(tf.matmul(l1, W2) + b2)

            W_mu = weight_variable([embed_dim, h_dim], name="W_mu")
            b_mu = bias_variable([h_dim], name="b_mu")
            W_sigma = weight_variable([embed_dim, h_dim], name="W_sigma")
            b_sigma = bias_variable([h_dim], name="b_sigma")

            self.mu = tf.matmul(l2, W_mu) + b_mu
            log_sigma_sq = tf.matmul(l2, W_sigma) + b_sigma
            self.sigma = tf.sqrt(tf.exp(log_sigma_sq))
            eps = tf.random_normal((v_batch_size, h_dim), 0, 1,
                                   dtype=tf.float32)

            h = self.mu + self.sigma*eps

        # Generator -------------------------------------
        with tf.variable_scope("Generator"):
            self.R = weight_variable([input_dim, h_dim], name="R")
            b = bias_variable([input_dim], name="b")
            E = -tf.matmul(h, self.R, transpose_b=True) - b
            p_x_i = tf.nn.log_softmax(E)

        # Optimizer ---------------------------------------

        # d_kl(q(z|x)||p(z)) = 0.5*sum(1 + log(sigma^2) - mu^2 - sigma^2)
        self.encoder_loss = -0.5 * (tf.reduce_sum(1 + log_sigma_sq -
                                                  tf.square(self.mu) -
                                                  tf.exp(log_sigma_sq), 1))
        # p(X|h) = sum(log(p_x_i)) for x_i in document
        self.generator_loss = -tf.reduce_sum(x * p_x_i, 1)

        self.total_loss = tf.reduce_mean(self.encoder_loss +
                                         self.generator_loss)

        Nd = tf.reduce_sum(x, 1)  # Length of each document

        # Used to calculate perplexity
        new_lower_bound = tf.reduce_mean((self.encoder_loss +
                                          self.generator_loss) / Nd)
        self.perp = tf.exp(new_lower_bound)
        self.sess = tf.Session()
        self.saver = tf.train.Saver()

    def train(self, train, valid, max_iter=1000, batch_size=100,
              alternating=False, learning_rate=0.001):
        """ Train the model using the ADAM optimizer.

        Parameters
        ----------
        train : Matrix of training data.
        valid : Matrix of validation data.
        learning_rate : Learning rate for updating paramters.
        batch_size : Size of each training batch.
        max_iter : Maximum number of iterations in training.
        alternating : Whether or not to update one part of the network
                      (keeping the other part fixed), as in the paper.
        """
        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(learning_rate, global_step, 250, 1,
                                        staircase=True)

        encoder_var_list, generator_var_list = [], []
        for var in tf.trainable_variables():
            if "Encoder" in var.name:
                encoder_var_list.append(var)
            elif "Generator" in var.name:
                generator_var_list.append(var)

        e_optimizer = tf.train.AdamOptimizer(learning_rate)\
            .minimize(self.total_loss, var_list=encoder_var_list)

        g_optimizer = tf.train.AdamOptimizer(learning_rate)\
            .minimize(self.total_loss, var_list=generator_var_list)

        optimizer = tf.train.AdamOptimizer(lr)\
            .minimize(self.total_loss, global_step=global_step)

        self.sess.run(tf.initialize_all_variables())

        tf.scalar_summary('Encoder loss',
                          tf.reduce_mean(self.encoder_loss))
        tf.scalar_summary('Generator loss',
                          tf.reduce_mean(self.generator_loss))
        tf.scalar_summary('Total loss',
                          self.total_loss)
        merged_sum = tf.merge_all_summaries()

        perp = tf.scalar_summary('Perplexity', self.perp)
        writer = tf.train.SummaryWriter("./logs/alt={}/batch_size={}/lr={}"
                                        .format(alternating, batch_size,
                                                learning_rate),
                                        self.sess.graph)

        from sklearn.utils import shuffle
        for epoch in range(0, max_iter):
            print("--- Epoch:", epoch)
            losses = []
            train = shuffle(train)
            if epoch/10 % 2 == 0:
                print("Encoder optimizing")
            else:
                print("Decoder optimizing")
            for i, batch_i in enumerate(batch(train, batch_size)):
                feed = feed_from_sparse(batch_i, self.x_s)
                if alternating:
                    if epoch/10 % 2 == 0:
                        _, loss = self.sess.run([e_optimizer, self.total_loss],
                                                feed_dict=feed)
                    else:
                        _, loss = self.sess.run([g_optimizer, self.total_loss],
                                                feed_dict=feed)
                else:
                    _, loss = self.sess.run([optimizer, self.total_loss],
                                            feed_dict=feed)

                # loss = self.sess.run([self.total_loss],
                #                      feed_dict={self.x: batch_i})
                losses.append(loss)
                if i % 5000 == 0:
                    print("Step: {}, loss: {}".format(i, loss))

            print('--- Avg loss:', np.mean(losses))
            if epoch % 2 == 0:
                feed = feed_from_sparse(train, self.x_s)
                summary = self.sess.run(merged_sum, feed_dict=feed)
                writer.add_summary(summary, epoch)

            if epoch % 10 == 0:
                feed = feed_from_sparse(valid, self.x_s)
                summary = self.sess.run(perp, feed_dict=feed)
                writer.add_summary(summary, epoch)
                self.saver.save(self.sess, "checkpoints/nvdm.ckpt")

            # Find closest words
            # print(self.closest_words("weapons"))
            # print(self.closest_words("books"))

            # Report crude perplexity
            # print(self.perplexity(valid))

    def perplexity(self, data, samples=1):
        """ Calculate perplexity using 1 or more samples of the hidden unit.

        Parameters
        ----------
        data : Matrix of data to calculate perplexity on.
        samples : Number of samples to use.
        """
        feed = feed_from_sparse(data, self.x_s)
        if samples == 1:
            return self.sess.run(self.perp, feed_dict=feed)
        else:
            losses = []
            for i in range(samples):
                losses.append(self.sess.run(self.generator_loss,
                                            feed_dict=feed))
            generator_loss = np.mean(losses, axis=0)
            encoder_loss = self.sess.run(self.encoder_loss, feed_dict=feed)
            D = data.sum(axis=1)
            total_loss = np.mean((generator_loss + encoder_loss) / D)
            return np.exp(total_loss)

    def restore(self, path):
        """ Restores a previous model from path.

        Parameters
        ----------
        path : Path to the stored model.
        """
        self.saver.restore(self.sess, path)

    def closest_words(self, word, n=10):
        if self.word2idx is None or self.idx2word is None:
            return "No word to index mapping provided"
        R_out = self.sess.run(self.R)
        w = R_out[self.word2idx[word], :].reshape(1, self.h_dim)
        closest = distance.cdist(w, R_out, metric='cosine')[0].argsort()
        return self.idx2word[closest[:n]]

    def get_representation(self, data):
        feed = feed_from_sparse(data, self.x_s)
        return self.sess.run(self.mu, feed_dict=feed)
