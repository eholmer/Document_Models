from __future__ import print_function
from common import batch, bias_variable, weight_variable, feed_from_sparse
import tensorflow as tf
import numpy as np


class VAENADE():
    def __init__(self, voc_dim, embed_dim=500, h_dim=50, window_size=5):
        with tf.variable_scope("Input"):
            # self.x_s = tf.sparse_placeholder(tf.float32, name="sparse_bow")
            self.x_bow = tf.sparse_placeholder(tf.float32, name="sparse_bow")
            x = tf.sparse_tensor_to_dense(self.x_bow, validate_indices=False)
            self.x_seq = tf.placeholder(tf.int32, [None],
                                        name="sequence_input")
            # v_batch_size = tf.shape(self.x_s)[0]
            seq_len = tf.shape(self.x_seq)[0]
            # x = tf.sparse_tensor_to_dense(self.x_s, validate_indices=False,
            #                               name="dense_bow")

        with tf.variable_scope("Encoder"):
            # Encoder variables -->
            W1 = weight_variable([voc_dim, embed_dim], name="W1")
            b1 = bias_variable([embed_dim], name="b1")
            W_mu = weight_variable([embed_dim, h_dim], name="W_mu")
            b_mu = bias_variable([h_dim], name="b_mu")
            W_sigma = weight_variable([embed_dim, h_dim], name="W_sigma")
            b_sigma = bias_variable([h_dim], name="b_sigma")

            # Encoder flow -->
            l1 = tf.nn.relu(tf.matmul(x, W1) + b1, name="l1")
            self.mu = tf.add(tf.matmul(l1, W_mu), b_mu, name="mu")
            log_sigma_sq = tf.add(tf.matmul(l1, W_sigma), b_sigma,
                                  name="log_sigma_sq")
            self.sigma = tf.sqrt(tf.exp(log_sigma_sq), name="sigma")
            eps = tf.random_normal((1, h_dim), 0, 1,
                                   dtype=tf.float32, name="eps")

            # Encoder loss is KL-divergence -->
            self.e_loss = -0.5 * (tf.reduce_sum(1 + log_sigma_sq -
                                                tf.square(self.mu) -
                                                tf.exp(log_sigma_sq),
                                                name="Encoder_loss"))

            # Sample latent variable and tile it `seq_len` times -->
            h = tf.tile(tf.add(self.mu, self.sigma*eps), [seq_len, 1],
                        name="h")

        with tf.variable_scope("Decoder"):
            # Decoder variables -->
            W_emb = weight_variable([voc_dim, h_dim], name="W_embedding_mat")
            b_emb = bias_variable([h_dim], name="b_embedding")
            W_softm = weight_variable([2*h_dim, voc_dim], name="W_softmax")
            b_softm = bias_variable([voc_dim], name="b_softmax")

            # Decoder flow -->
            # Create windows of length window_size of the previous words
            # to predict current.
            embeddings = tf.pad(tf.gather(W_emb, self.x_seq[:-1]),
                                [[1, 0], [0, 0]], name="embeddings")
            cumbeddings = tf.cumsum(embeddings, name="cumulative_embeddnings")
            transposed_cumbeddnings = tf.pad(cumbeddings[:-window_size],
                                             [[window_size, 0], [0, 0]],
                                             name="transposed_cumbeddings")
            h_windows = cumbeddings - transposed_cumbeddnings + b_emb
            h_windows = tf.sigmoid(h_windows, name="h_windows")

            # Concat window representations and global representation from
            # encoder.
            h_concat = tf.concat(1, [h_windows, h], name="h_concat")

            # Decoder loss is negative log-likelihood.
            softmax = tf.nn.sparse_softmax_cross_entropy_with_logits
            p_x_i = softmax(tf.matmul(h_concat, W_softm) + b_softm,
                            self.x_seq, name="Cross_entropy")

            self.g_loss = tf.reduce_sum(p_x_i, name="Decoder_loss")

        self.tot_loss = self.e_loss + self.g_loss

        self.sess = tf.Session()
        self.saver = tf.train.Saver()

    def train(self, train, train_seq, max_epochs=1000, learning_rate=0.00001):
        optimizer = tf.train.AdamOptimizer(learning_rate)\
            .minimize(self.tot_loss)
        self.sess.run(tf.initialize_all_variables())
        valid = train[:100]
        train = train[100:]
        valid_seq = train_seq[:100]
        train_seq = train_seq[100:]
        num_docs = train.shape[0]

        for epoch in range(0, max_epochs):
            print("--- Epoch:", epoch)
            losses = []
            for i in range(0, num_docs):
                feed = feed_from_sparse(train[i], self.x_bow)
                feed[self.x_seq] = train_seq[i]
                _, loss = self.sess.run([optimizer, self.tot_loss], feed)
                losses.append(loss)
                if i % 500 == 0:
                    print("Processing doc: {}".format(i))
                    print("Loss: {}".format(np.mean(losses)))
            print("--- Avg loss: {}".format(np.mean(losses)))
            perplexity = self.perplexity(valid, valid_seq)
            print("--- Perplexity: {}".format(perplexity))
            self.saver.save(self.sess, "checkpoints/vaenade.ckpt")

    def perplexity(self, data, data_seq):
        perps = []
        num_docs = data.shape[0]
        for i in range(0, num_docs):
            if len(data_seq[i]) == 0:
                continue
            feed = feed_from_sparse(data[i], self.x_bow)
            feed[self.x_seq] = data_seq[i]
            loss = self.sess.run(self.tot_loss, feed)
            perps.append(loss/data[i].sum())
        return np.exp(np.mean(perps))

    def restore(self, path):
        """ Restores a previous model from path.

        Parameters
        ----------
        path : Path to the stored model.
        """
        self.saver.restore(self.sess, path)

    def get_representation(self, data):
        rep = []
        for b in batch(data, 10000):
            feed = feed_from_sparse(b, self.x_bow)
            r = self.sess.run(self.mu, feed)
            rep.append(r)
        return np.vstack(rep)
