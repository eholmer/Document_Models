from topic_models import DocNADE, RSM, NVDM
import numpy as np
import scipy.sparse as sp


def load_data(input_file):
    npzfile = np.load(input_file)
    mat = sp.csr_matrix((npzfile['data'], npzfile['indices'],
                         npzfile['indptr']),
                        shape=tuple(list(npzfile['shape'])))
    return mat


def load_20ng():
    train = load_data('data/20ng/train_data.npz')
    train_target = np.load('data/20ng/train_labels.npy')
    validation = load_data('data/20ng/valid_data.npz')
    validation_target = np.load('data/20ng/valid_labels.npy')
    test = load_data('data/20ng/test_data.npz')
    test_target = np.load('data/20ng/test_labels.npy')
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
    return (train, train_target, validation, validation_target, test,
            test_target)

# Load data-------------------------------------
# Bag of words vectors for NVDM and RSM.
# Sequence of word-indicies for DocNADE

(train, train_target, validation, validation_target, test, test_target,
 idx2word, word2idx) = load_20ng()

# (train, train_target, validation, validation_target, test, test_target)\
        # = load_reuters()

# Convert sparse bag of words to DocNADE sequences with randomized order.
train_dn = []
for i, doc in enumerate(train):
    d = []
    for idx, count in zip(doc.indices, doc.data):
        for j in range(int(count)):
            d.append(idx)
    np.random.shuffle(d)
    train_dn.append(d)

valid_dn = []
for i, doc in enumerate(validation):
    d = []
    for idx, count in zip(doc.indices, doc.data):
        for j in range(int(count)):
            d.append(idx)
    np.random.shuffle(d)
    valid_dn.append(d)

test_dn = []
for i, doc in enumerate(test):
    d = []
    for idx, count in zip(doc.indices, doc.data):
        for j in range(int(count)):
            d.append(idx)
    np.random.shuffle(d)
    test_dn.append(d)

# Use models.
# DocNADE
# dn = DocNADE(voc_size=train.shape[1])
# dn.restore('checkpoints/docnade_p=875.ckpt')
# dn.train(train_dn, valid_dn)
# print(dn.closest_words("medical"))
# print(dn.perplexity(test_dn))
# print(dn.ir(train_dn, test_dn, train_target, test_target))

# RSM
# rsm = RSM(input_dim=10000)
# rsm.restore('checkpoints/rsm_p=975.ckpt')
# rsm.train(train, max_iter=100)
# print(rsm.closest_word("medical"))
# print(rsm.perplexity(test, steps=1000))
# print(rsm.ir(train, test, train_target, test_target))

# NVDM
# nvdm = NVDM(input_dim=train.shape[1])
# nvdm.restore('checkpoints/nvdm_p=890.ckpt')
# nvdm.train(train, validation)
# print(nvdm.closest_words("medical"))
# print(nvdm.perplexity(test))
# print(nvdm.ir(train, test, train_target, test_target))
