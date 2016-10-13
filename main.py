from docNADE.docnade import DocNADE, RSM, NVDM
import numpy as np
import scipy.sparse as sp


def load_data(input_file):
    npzfile = np.load(input_file)
    mat = sp.csr_matrix((npzfile['data'], npzfile['indices'],
                         npzfile['indptr']),
                        shape=tuple(list(npzfile['shape'])))
    return mat

# Load data-------------------------------------
# Bag of words vectors for NVDM and RSM.
# Sequence of word-indicies for DocNADE

train = load_data('data/train_data.npz')
train_target = np.load('data/train_labels.npy')
validation = load_data('data/valid_data.npz')
validation_target = np.load('data/valid_labels.npy')
test = load_data('data/test_data.npz')
test_target = np.load('data/test_labels.npy')
with open('data/vocab.txt', 'rb') as f:
    voc = f.read().splitlines()
input_dim = len(voc)
n_samples = train.shape[0]
idx2word = np.array(voc)
word2idx = dict(zip(voc, range(0, input_dim)))

# Convert sparse bag of words to DocNADE sequences with randomized order.
train_dn = []
for i, doc in enumerate(train):
    d = []
    for idx, count in zip(doc.indices, doc.data):
        for j in range(int(count)):
            d.append(idx)
    np.random.shuffle(d)
    train_dn.append(d)

test_dn = []
for i, doc in enumerate(test):
    d = []
    for idx, count in zip(doc.indices, doc.data):
        for j in range(int(count)):
            d.append(idx)
    np.random.shuffle(d)
    test_dn.append(d)

train = train.toarray()
test = test.toarray()

# Use models.

# DocNADE
# dn = DocNADE(word2idx, idx2word)
# dn.restore('checkpoints/docnade_p=875.ckpt')
# # dn.train(train_dn, test_dn, 2)
# print(dn.closest_words("medical"))
# print(dn.perplexity(test_dn))
# print(dn.ir(train_dn, test_dn, train_target, test_target))

# RSM
# rsm = RSM(word2idx, idx2word)
# rsm.restore('checkpoints/rsm_p=975.ckpt')
# # rsm.train(train, test, max_iter=10)
# print(rsm.closest_word("medical"))
# print(rsm.perplexity(test, steps=1000))
# print(rsm.ir(train, test, train_target, test_target))

# NVDM
# nvdm = NVDM(word2idx, idx2word)
# nvdm.restore('checkpoints/nvdm_p=890.ckpt')
# # nvdm.train(train, test, max_iter=5, learning_rate=0.0001)
# print(nvdm.closest_words("medical"))
# print(nvdm.perplexity(test))
# print(nvdm.ir(train, test, train_target, test_target))
