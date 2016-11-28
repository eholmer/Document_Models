from __future__ import print_function
from topic_models import DocNADE, RSM, NVDM, DeepDocNADE, VAENADE
import numpy as np
import scipy.sparse as sp
from sklearn import linear_model, metrics 
from sklearn.utils import shuffle
from sklearn.datasets import load_svmlight_file
import pickle


def load_data(input_file):
    npzfile = np.load(input_file)
    mat = sp.csr_matrix((npzfile['data'], npzfile['indices'],
                         npzfile['indptr']),
                        shape=tuple(list(npzfile['shape'])))
    return mat


def load_seq():
    train, train_target = load_svmlight_file('data/20ng_seq/train_log')
    test, test_target = load_svmlight_file('data/20ng_seq/test_log')
    with open('data/20ng_seq/seq_train', 'r') as f:
        dump = pickle.load(f)
        train_seq = dump['X']
    with open('data/20ng_seq/seq_test', 'r') as f:
        dump = pickle.load(f)
        test_seq = dump['X']
    with open('data/20ng_seq/meta_data', 'r') as f:
        dump = pickle.load(f)
        word2idx = dump['w2i']
        idx2word = dump['i2w']
    return (train, train_seq, train_target, test, test_seq, test_target,
            word2idx, idx2word)


def load_20ng():
    train = load_data('data/20ng/train_data.npz')
    train_target = np.load('data/20ng/train_labels.npy')
    validation = load_data('data/20ng/valid_data.npz')
    validation_target = np.load('data/20ng/valid_labels.npy')
    test = load_data('data/20ng/test_data.npz')
    test_target = np.load('data/20ng/test_labels.npy')
    train, train_target = shuffle(train, train_target)
    validation, validation_target = shuffle(validation, validation_target)
    test, test_target = shuffle(test, test_target)
    with open('data/20ng/vocab.txt', 'r') as f:
        voc = f.read().splitlines()
    input_dim = len(voc)
    idx2word = np.array(voc)
    word2idx = dict(zip(voc, range(0, input_dim)))
    return (train, train_target, validation, validation_target, test,
            test_target, idx2word, word2idx)


def load_reuters():
    train = load_data('data/reuters/train_data.npz')
    train_target = np.load('data/reuters/train_labels.npz')
    validation = load_data('data/reuters/valid_data.npz')
    validation_target = np.load('data/reuters/valid_labels.npz')
    test = load_data('data/reuters/test_data.npz')
    test_target = np.load('data/reuters/test_labels.npz')
    train, train_target = shuffle(train, train_target)
    validation, validation_target = shuffle(validation, validation_target)
    test, test_target = shuffle(test, test_target)
    return (train, train_target, validation, validation_target, test,
            test_target)

# Load data-------------------------------------
# Bag of words vectors for NVDM and RSM.
# Sequence of word-indicies for DocNADE

(train, train_seq, train_target, test, test_seq, test_target,
 word2idx, idx2word) = load_seq()
# (train, train_target, validation, validation_target, test, test_target,
#  idx2word, word2idx) = load_20ng()

# (train, train_target, validation, validation_target, test, test_target)\
#         = load_reuters()

# Convert sparse bag of words to DocNADE sequences with randomized order.
# train_dn = []
# for i, doc in enumerate(train):
#     d = []
#     for idx, count in zip(doc.indices, doc.data):
#         for j in range(int(count)):
#             d.append(idx)
#     np.random.shuffle(d)
#     train_dn.append(d)

# valid_dn = []
# for i, doc in enumerate(validation):
#     d = []
#     for idx, count in zip(doc.indices, doc.data):
#         for j in range(int(count)):
#             d.append(idx)
#     np.random.shuffle(d)
#     valid_dn.append(d)

# test_dn = []
# for i, doc in enumerate(test):
#     d = []
#     for idx, count in zip(doc.indices, doc.data):
#         for j in range(int(count)):
#             d.append(idx)
#     np.random.shuffle(d)
#     test_dn.append(d)


# Use models.
# DocNADE
# dn = DocNADE(voc_size=train.shape[1])
# dn.restore('checkpoints/docnade_p=875.ckpt')
# dn.wiki_test()
# dn.train(train_dn, valid_dn)
# print(dn.closest_words("medical"))
# print(dn.perplexity(test_dn))
# print(dn.ir(train_dn, test_dn, train_target, test_target))

# RSM
# rsm = RSM(input_dim=train.shape[1])
# rsm.restore('checkpoints/rsm_p=975.ckpt')
# rsm.train(train, max_iter=100)
# print(rsm.closest_word("medical"))
# print(rsm.perplexity(test, steps=1000))
# print(rsm.ir(train, test, train_target, test_target))

# NVDM
# nvdm = NVDM(input_dim=train.shape[1], word2idx=word2idx, idx2word=idx2word)
# nvdm.restore('checkpoints/nvdm.ckpt')
# nvdm.train(train, test, alternating=True, learning_rate=0.0005, max_iter=10000,
#            batch_size=10)
# print(nvdm.closest_words("medical"))
# print(nvdm.get_perplexity(test))
# print(nvdm.ir(train, test, train_target, test_target))

# DeepDocNADE
# ddn = DeepDocNADE(word2idx=word2idx, idx2word=idx2word, voc_size=2000)
# ddn.train(train, validation, learning_rate=0.0005)
# ddn.restore('checkpoints/deep_docnade.ckpt')
# ddn.restore('checkpoints/docnade_p=818.ckpt')
# print(ddn.perplexity(test, False))
# print(ddn.perplexity(test_dn, True, ensembles=1))
# print(ddn.perplexity(valid_dn, True))
# print(ddn.ir(train, test, train_target, test_target))

# VAENADE
# vn = VAENADE(voc_dim=train.shape[1])
# vn.train(train, train_seq)
# X_train = nvdm.get_representation(train)
# X_test = nvdm.get_representation(test)
# logistic = linear_model.LogisticRegression()
# logistic.fit(X_train, train_target)
# pred = logistic.predict(X_test)
# print(metrics.classification_report(test_target, pred))
# print(metrics.confusion_matrix(test_target, pred))
# print(logistic.fit(X_train, train_target.flatten()).score(X_test, test_target))
