from __future__ import print_function
import tensorflow as tf
import numpy as np
from scipy.spatial import distance


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


def similarity(train, test, train_target, test_target, model, second=False):
    labels = set(train_target)
    train_rep = model.get_representation(train)
    test_rep = model.get_representation(test)
    dists = []
    cluster_centers = []
    for label in labels:
        cluster_centers.append(train_rep[train_target == label].mean(axis=0))

    cluster_centers = np.array(cluster_centers)
    dist_all = distance.cdist(test_rep, cluster_centers, metric='cosine')
    if second:
        for i, label in enumerate(labels):
            dist_except = dist_all[np.ix_(test_target == label,
                                          np.arange(len(cluster_centers)) != i)]
            dist_except.sort(axis=1)
            second_center = dist_except[:, 0]
            mean_dist_all = dist_all[test_target == label].mean(axis=1)
            dists.append((second_center/mean_dist_all).mean())
    else:
        for i, label in enumerate(labels):
            dist_center = dist_all[test_target == label, i]
            mean_dist_all = dist_all[test_target == label].mean(axis=1)
            dists.append((dist_center/mean_dist_all).mean())
    return dists


def tsne(data, data_target, model):
    from matplotlib import pyplot as plt
    from sklearn.manifold import TSNE
    rep = model.get_representation(data)
    tsne = TSNE(perplexity = 30, n_components = 2, init = 'pca', n_iter = 5000)
    print("Fitting")
    two_d_embeddings = tsne.fit_transform(rep)
    print(two_d_embeddings)
    plt.scatter(two_d_embeddings[:, 0], two_d_embeddings[:, 1], c=data_target)
    plt.show()


def feed_from_sparse(data, target):
    data = data.tocoo()
    ind = np.vstack([data.row, data.col]).T
    return {target: (ind, data.data, data.shape)}
