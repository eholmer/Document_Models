from __future__ import print_function
import numpy as np
import tensorflow as tf
from scipy.spatial import distance
from scipy.misc import logsumexp
import pickle


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
    l = iterable.shape[0]
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def ir(train, test, train_target, test_target, model, multi_label=False):
    """ Perform Information Retrieval test. Measures the precision for
    different pre-defined Recall rates.

    Parameters
    ----------
    train : Matrix of training samples.
    test : Matrix of testing samples.
    train_target : Array of target labels for each training sample.
    test_target : Array of target labels for each testing sample.
    model : Model from which to get the document representations.
    intervals : List of recall rates.
    multi_label: True if a document can belong to multiple classes.
    """
    print("Getting train representations")
    train_rep = model.get_representation(train)

    print("Getting test representations")
    test_rep = model.get_representation(test)

    print("Calculating distance")
    if multi_label:
        correct = np.zeros(train.shape[0])
        # for i, row in enumerate(test_rep):
        batch_size = 500
        for ndx in range(0, test_rep.shape[0], batch_size):
            closest = distance.cdist(test_rep[ndx:ndx+batch_size], train_rep,
                                     metric='cosine').argsort(axis=1)
            target_batch = test_target[ndx:ndx+batch_size]
            for i, row in enumerate(closest):
                print("Row: {}".format(i+ndx))
                res = (train_target[row].multiply(target_batch[i]) 
                       .sum(axis=1)/(target_batch[i].sum()*1.0)).A1
                correct += res
        correct /= test_target.shape[0]
    else:
        closest = distance.cdist(test_rep, train_rep, metric='cosine')\
                .argsort(axis=1)
        closest_t = []
        for row in closest:
            closest_t.append(train_target[row].flatten())
        test_target = np.reshape(test_target, (len(test_target), 1))
        correct = np.mean(closest_t == test_target, axis=0)
    return correct


def feed_from_sparse(data, target):
    data = data.tocoo()
    ind = np.vstack([data.row, data.col]).T
    return {target: (ind, data.data, data.shape)}


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


class DeepDocNADE():
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

        # Variables
        self.W = weight_variable([voc_size, h_dim], name="W")
        b = bias_variable([voc_size], name="b")
        c = bias_variable([h_dim], name="c")
        V = weight_variable([h_dim, voc_size], name="V")

        factor = 1./(Dv - i)

        # Flow
        h = activ(tf.matmul(low, self.W) + c)
        p = tf.nn.log_softmax(tf.matmul(h, V) + b)
        nll = tf.reduce_sum(-p * high, 1)
        self.perp = tf.exp(tf.reduce_mean(nll * factor))
        self.nll = tf.reduce_mean(Dv*factor*nll, 0)

        # Test Flow
        W_cum = tf.pad(tf.cumsum(tf.gather(self.W, self.x_test[:-1])),
                       [[1, 0], [0, 0]])
        W_cum += c
        H = activ(W_cum)
        softmax = tf.nn.sparse_softmax_cross_entropy_with_logits
        p_x_i = softmax(tf.matmul(H, V) + b, self.x_test)
        self.nll_test = tf.reduce_sum(p_x_i)

        # Representation of Documents
        self.rep = activ(tf.matmul(x, self.W) + c)

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
            .minimize(self.nll)

        best = np.inf
        self.sess.run(tf.initialize_all_variables())
        for epoch in range(max_iter):
            losses = []
            print("-------Epoch: {}".format(epoch))
            for j, doc in enumerate(batch(train, 100)):
                # for j, doc in enumerate(train):
                feed = feed_from_sparse(doc, self.x_s)
                _, loss = self.sess.run([optimizer, self.nll],
                                        feed_dict=feed)
                losses.append(loss)

            print("Loss: {}".format(np.mean(losses)))
            perplexity = self.perplexity(test, False)
            print("Perplexity: {}".format(perplexity))
            if perplexity < best:
                self.saver.save(self.sess, "checkpoints/deep_docnade.ckpt")
                best = perplexity
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
