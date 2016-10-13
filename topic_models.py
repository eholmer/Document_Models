import numpy as np
import tensorflow as tf
from scipy.spatial import distance
from scipy.misc import logsumexp
import cPickle as pickle


def weight_variable(shape, name=None):
    '''Helper function to create a weight variable initialized with
    a normal distribution
    Parameters
    ----------
    shape : list
        Size of weight variable
    '''
    initial = tf.random_normal_initializer(mean=0.0, stddev=0.01)
    return tf.get_variable(name, shape=shape, initializer=initial)


def bias_variable(shape, name=None):
    '''Helper function to create a bias variable initialized with
    a constant value.
    Parameters
    ----------
    shape : list
        Size of weight variable
    '''
    initial = tf.random_normal_initializer(mean=0.0, stddev=0.01)
    return tf.get_variable(name, shape=shape, initializer=initial)


def batch(iterable, n=1):
    ''' Generate batches of size n unless we're at the end of the collection

    Parameters
    ----------
    iterable : Matrix to take batches from.
    n : Size of each batch.
    '''
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


class DocNADE():
    def __init__(self, word2idx, idx2word, h_dim=50, voc_size=2000):
        """ Initate Tensorflow graph for DocNADE.

        Parameters
        ----------
        word2idx : Dict with mappings from word to its' index in the input
                   vector.
        idx2word : Array where each index corresponds to a word.
        h_dim : Dimension of hidden units.
        voc_size : Vocabulary size.

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
        input_dim = tf.shape(self.x)[0]

        # Variables
        self.variables = {}
        self.W = weight_variable([voc_size, h_dim], name="W")
        b = bias_variable([voc_size], name="b")
        self.c = bias_variable([h_dim], name="c")
        V = weight_variable([h_dim, voc_size], name="V")

        # Flow
        W_cum = tf.pad(tf.cumsum(tf.gather(self.W, self.x[:-1])),
                       [[1, 0], [0, 0]])
        W_cum += self.c
        H = tf.sigmoid(W_cum)
        P = tf.nn.log_softmax(tf.matmul(H, V) + b)
        idx_flattened = tf.range(0, input_dim) * voc_size + self.x
        p_x_i = tf.gather(tf.reshape(P, [-1]), idx_flattened)
        self.nll = tf.reduce_sum(-p_x_i)

        # Tensor for getting the representation of a document
        self.rep = tf.sigmoid(tf.reduce_sum(tf.gather(self.W, self.x), 0)
                              + self.c)

        self.sess = tf.Session()
        self.saver = tf.train.Saver()

    def train(self, train, test, max_iter, learning_rate=0.001):
        """ Train the model using ADAM optimizer.

        Parameters
        ----------
        train : Matrix of training data.
        test : Matrix of testing data.
        learning_rate : Learning rate for updating paramters.
        max_iter : Maximum number of iterations in training.
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

    def ir(self, train, test, train_target, test_target):
        """ Perform Information Retrieval test. Measures the precision for
        different pre-defined Recall rates.

        Parameters
        ----------
        train : Matrix of training samples.
        test : Matrix of testing samples.
        train_target : Array of target labels for each training sample.
        test_target : Array of target labels for each testing sample.
        """

        fracs = np.rint(np.array([0.0002, 0.001, 0.004, 0.016, 0.064, 0.256])
                        * len(train_target))
        print("Getting train representations")
        train_rep = self.get_representation(train)

        print("Getting test representations")
        test_rep = self.get_representation(test)

        print("Calculating distance")
        closest = distance.cdist(test_rep, train_rep, metric='cosine')\
            .argsort(axis=1)
        closest_t = []
        for row in closest:
            closest_t.append(train_target[row].flatten())
        test_target = np.reshape(test_target, (len(test_target), 1))
        correct = np.mean(closest_t == test_target, axis=0)
        prec = []
        for frac in fracs:
            subset = correct[:frac+1]
            prec.append(np.mean(subset))
        return prec

    def closest_words(self, word, n=10):
        W_out = self.sess.run(self.W)
        w = W_out[self.word2idx[word], :]\
            .reshape(1, self.h_dim)
        closest = distance.cdist(w, W_out, metric='cosine')[0].argsort()
        return self.idx2word[closest[:n]]

    def get_representation(self, data):
        reps = []
        for doc in data:
            reps.append(self.sess.run(self.rep, feed_dict={self.x: doc}))
        return np.array(reps)

    def wiki_test(self):
        with open('data/wiki', 'r') as f:
            wiki = pickle.load(f)

        with open('data/random', 'r') as f:
            random = pickle.load(f)

        losses1 = []
        losses2 = []
        for doc1, doc2 in zip(wiki, random):
            loss1 = self.sess.run(self.nll, feed_dict={self.x: doc1})
            loss2 = self.sess.run(self.nll, feed_dict={self.x: doc2})
            losses1.append(loss1/len(doc1))
            losses2.append(loss2/len(doc2))
        print(np.exp(np.mean(losses1))/np.exp(np.mean(losses2)))


class RSM():
    def __init__(self, word2idx, idx2word, h_dim=50, input_dim=2000):
        """ Initiate parameters and weights for the Replicated Softmax.

        Parameters
        ----------
        h_dim : Dimension of hidden unit.
        input_dim : Dimension of input vector.
        word2idx : Dict with mappings from word to its' index in the input
                   vector.
        idx2word : Array where each index corresponds to a word.

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

    def train(self, train, test, learning_rate=0.001, batch_size=100,
              max_iter=1000, momentum=0.9):
        """ Start training the RSM. Uses Contrastive Divergece with 1 sampling
        step to approximate the gradients.

        Parameters
        ----------
        train : Matrix of training data.
        test : Matrix of testing data.
        learning_rate : Learning rate for updating paramters.
        batch_size : Size of each training batch.
        max_iter : Maximum number of iterations in training.
        momentum : The momentum used in each update.
        """

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
                h_1_prob = self._sigmoid(np.dot(x, self.W)
                                         + np.outer(D, self.a))
                h_1_sample = np.array(np.random.rand(v_batch_size, self.h_dim)
                                      < h_1_prob, dtype=int)

                # - from p(x|h)
                x_1_prob = np.exp(np.dot(h_1_sample, self.W.T) + self.b)
                x_1_prob /= x_1_prob.sum(axis=1).reshape(v_batch_size, 1)
                x_1_sample = np.zeros(x_1_prob.shape)
                for i in xrange(v_batch_size):
                    x_1_sample[i] = np.random.multinomial(D[i], x_1_prob[i])

                # - compute p(h|x)
                h_2_prob = self._sigmoid(np.dot(x_1_sample, self.W)
                                         + np.outer(D, self.a))

                # 2. Update parameters
                self.W_upd *= momentum
                self.W_upd += np.dot(x.T, h_1_prob)\
                    - np.dot(x_1_sample.T, h_2_prob)

                self.a_upd *= momentum
                self.a_upd = h_1_prob.sum(axis=0) - h_2_prob.sum(axis=0)

                self.b_upd *= momentum
                self.b_upd = x.sum(axis=0) - x_1_sample.sum(axis=0)

                self.W += learning_rate * self.W_upd
                self.a += learning_rate * self.a_upd
                self.b += learning_rate * self.b_upd

                reconstruction_error.append(np.linalg.norm(x_1_sample - x)**2
                                            / (self.input_dim*v_batch_size))

            print("Mean reconstrction error: {}"
                  .format(np.mean(reconstruction_error)))

            # Find closest word
            print(self.closest_word("weapons"))
            print(self.closest_word("books"))

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
            for j in xrange(M):
                x_1_sample[j, :] = np.random.multinomial(D, x_1_prob, size=1)

            def p_k(b_k, v):
                return np.dot(v, self.b)*b_k\
                       + np.log1p(np.exp(b_k*(np.dot(v, self.W) + D*self.a)))\
                       .sum(axis=1)

            for s in xrange(0, steps-1):

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
                for j in xrange(M):
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

    def ir(self, train, test, train_target, test_target):
        """ Perform Information Retrieval test. Measures the precision for
        different pre-defined Recall rates.

        Parameters
        ----------
        train : Matrix of training samples.
        test : Matrix of testing samples.
        train_target : Array of target labels for each training sample.
        test_target : Array of target labels for each testing sample.
        """

        fracs = np.rint(np.array([0.0002, 0.001, 0.004, 0.016, 0.064, 0.256])
                        * len(train_target))
        train_rep = self.get_representation(train)
        test_rep = self.get_representation(test)

        print("Calculating distance")
        closest = distance.cdist(test_rep, train_rep, metric='cosine')\
            .argsort(axis=1)
        closest_t = []
        for row in closest:
            closest_t.append(train_target[row].flatten())
        test_target = np.reshape(test_target, (len(test_target), 1))
        correct = np.mean(closest_t == test_target, axis=0)

        prec = []
        for frac in fracs:
            subset = correct[:frac+1]
            prec.append(np.mean(subset))
        print(prec)

    def closest_word(self, word, n=10):
        w = self.W[self.word2idx[word], :].reshape(1, self.h_dim)
        closest = distance.cdist(w, self.W, metric='cosine')[0].argsort()
        return self.idx2word[closest[:10]]

    def get_representation(self, data):
        D = data.sum(axis=1)
        return self._sigmoid(np.dot(data, self.W) + np.outer(D, self.a))

    def _sigmoid(self, x):
        return 1/(1 + np.exp(-x))


class NVDM():
    def __init__(self, word2idx, idx2word, input_dim=2000, h_dim=50,
                 embed_dim=500):
        """ Initate Tensorflow graph for the Neural Variational Document Model.

        Parameters
        ----------
        word2idx : Dict with mappings from word to its' index in the input
        vector.
        idx2word : Array where each index corresponds to a word.
        input_dim : Dimension of input data.
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
        self.x = tf.placeholder(tf.float32, [None, input_dim], name="input")
        v_batch_size = tf.shape(self.x)[0]

        # Encoder -----------------------------------
        with tf.variable_scope("Encoder"):
            W1 = weight_variable([input_dim, embed_dim], name="W1")
            b1 = bias_variable([embed_dim], name="b1")
            l1 = tf.nn.relu(tf.matmul(self.x, W1) + b1)
            W2 = weight_variable([embed_dim, embed_dim], name="W2")
            b2 = bias_variable([embed_dim], name="b2")
            l2 = tf.nn.relu(tf.matmul(l1, W2) + b2)

            W_mu = weight_variable([embed_dim, h_dim], name="W_mu")
            b_mu = bias_variable([h_dim], name="b_mu")
            W_sigma = weight_variable([embed_dim, h_dim], name="W_sigma")
            b_sigma = bias_variable([h_dim], name="b_sigma")

            self.mu = tf.matmul(l2, W_mu) + b_mu
            log_sigma_sq = tf.matmul(l2, W_sigma) + b_sigma
            eps = tf.random_normal((v_batch_size, h_dim), 0, 1,
                                   dtype=tf.float32)

            sigma = tf.sqrt(tf.exp(log_sigma_sq))
            h = self.mu + sigma*eps

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
        self.generator_loss = -tf.reduce_sum(self.x * p_x_i, 1)

        self.total_loss = tf.reduce_mean(self.encoder_loss +
                                         self.generator_loss)

        tf.scalar_summary('Encoder loss',
                          tf.reduce_mean(self.encoder_loss))
        tf.scalar_summary('Generator loss',
                          tf.reduce_mean(self.generator_loss))
        tf.scalar_summary('Total loss',
                          self.total_loss)

        Nd = tf.reduce_sum(self.x, 1)  # Length of each document

        # Used to calculate perplexity
        self.new_lower_bound = tf.reduce_mean((self.encoder_loss +
                                               self.generator_loss) / Nd)
        self.sess = tf.Session()
        self.saver = tf.train.Saver()

    def train(self, train, test, max_iter=1000, batch_size=100,
              alternating=False, learning_rate=0.001):
        """ Train the model using the ADAM optimizer.

        Parameters
        ----------
        train : Matrix of training data.
        test : Matrix of testing data.
        learning_rate : Learning rate for updating paramters.
        batch_size : Size of each training batch.
        max_iter : Maximum number of iterations in training.
        alternating : Whether or not to update one part of the network
                      (keeping the other part fixed), as in the paper.
        """

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

        optimizer = tf.train.AdamOptimizer(learning_rate)\
            .minimize(self.total_loss)

        self.sess.run(tf.initialize_all_variables())
        merged_sum = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter("./logs/", self.sess.graph)

        for epoch in xrange(0, max_iter):
            print("--- Epoch:", epoch)
            losses = []
            for i, batch_i in enumerate(batch(train, batch_size)):

                if alternating:
                    self.sess.run([e_optimizer],
                                  feed_dict={self.x: batch_i})
                    self.sess.run([g_optimizer],
                                  feed_dict={self.x: batch_i})
                else:
                    self.sess.run([optimizer],
                                  feed_dict={self.x: batch_i})

                loss = self.sess.run([self.total_loss],
                                     feed_dict={self.x: batch_i})
                losses.append(loss)
                if i % 2 == 0:
                    summary = self.sess.run(merged_sum,
                                            feed_dict={self.x: batch_i})
                    writer.add_summary(summary, epoch)
                if i % 10 == 0:
                    print("Step: {}, loss: {}".format(i, loss))

            print('--- Avg loss:', np.mean(losses))

            if epoch % 10 == 0:
                self.saver.save(self.sess, "checkpoints/nvdm.ckpt")

            # Find closest words
            print(self.closest_words("weapons"))
            print(self.closest_words("books"))

            # Report crude perplexity
            print(self.perplexity(test))

    def perplexity(self, data, samples=1):
        """ Calculate perplexity using 1 or more samples of the hidden unit.

        Parameters
        ----------
        data : Matrix of data to calculate perplexity on.
        samples : Number of samples to use.
        """
        if samples == 1:
            return np.exp(self.sess.run(self.new_lower_bound,
                                        feed_dict={self.x: data}))
        else:
            losses = []
            for i in range(samples):
                losses.append(self.sess.run(self.generator_loss,
                                            feed_dict={self.x: data}))
            generator_loss = np.mean(losses, axis=0)
            encoder_loss = self.sess.run(self.encoder_loss,
                                         feed_dict={self.x: data})
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

    def ir(self, train, test, train_target, test_target):
        """ Perform Information Retrieval test. Measures the precision for
        different pre-defined Recall rates.

        Parameters
        ----------
        train : Matrix of training samples.
        test : Matrix of testing samples.
        train_target : Array of target labels for each training sample.
        test_target : Array of target labels for each testing sample.
        """
        fracs = np.rint(np.array([0.0002, 0.001, 0.004, 0.016, 0.064, 0.256])
                        * len(train_target))

        print("Getting train representations")
        train_rep = self.get_representation(train)

        print("Getting test representations")
        test_rep = self.get_representation(test)

        print("Calculating distance")
        closest = distance.cdist(test_rep, train_rep, metric='cosine')\
            .argsort(axis=1)
        closest_t = []
        for row in closest:
            closest_t.append(train_target[row].flatten())
        test_target = np.reshape(test_target, (len(test_target), 1))
        correct = np.mean(closest_t == test_target, axis=0)
        prec = []
        for frac in fracs:
            subset = correct[:frac+1]
            prec.append(np.mean(subset))
        return prec

    def closest_words(self, word, n=10):
        R_out = self.sess.run(self.R)
        w = R_out[self.word2idx[word], :].reshape(1, self.h_dim)
        closest = distance.cdist(w, R_out, metric='cosine')[0].argsort()
        return self.idx2word[closest[:n]]

    def get_representation(self, data):
        return self.sess.run(self.mu, feed_dict={self.x: data})
