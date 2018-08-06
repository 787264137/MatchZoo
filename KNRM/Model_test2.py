from keras.models import Model
from keras.layers import *
from keras.initializers import RandomUniform
import random
<<<<<<< HEAD
import numpy as np
import keras.backend as K
from keras.optimizers import Adam, Adagrad, SGD
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import matplotlib.pyplot as plt
import time
=======
import keras.backend as K
from keras.optimizers import Adam, Adagrad, SGD
import matplotlib.pyplot as plt
import time
from keras.metrics import m_a_p as MAP
import numpy as np
>>>>>>> 6636718fe1739e29021c42bc19be0f43de359b94

np.set_printoptions(np.nan)


class PairGenerator():
<<<<<<< HEAD
    def __init__(self):
=======
    def __init__(self, rel_file):
>>>>>>> 6636718fe1739e29021c42bc19be0f43de359b94
        self.__name = 'PairGenerator'
        self.data1_maxlen = 10
        self.data2_maxlen = 40
        self.fill_word = 407870 - 1
<<<<<<< HEAD
        rel_file = './data/WikiQA/relation_train.txt'
=======
>>>>>>> 6636718fe1739e29021c42bc19be0f43de359b94
        self.rel = read_relation(filename=rel_file)
        self.batch_size = 100
        self.check_list = ['relation_file', 'batch_size']
        self.point = 0
        self.pair_list = self.make_pair_static(self.rel)
        print('[PairGenerator] init done', end='\n')

    # 把文件中读取的数据，转化成 pair_list
    def make_pair_static(self, rel):
        rel_set = {}
        pair_list = []
        for label, d1, d2 in rel:
            if d1 not in rel_set:
                rel_set[d1] = {}
            if label not in rel_set[d1]:
                rel_set[d1][label] = []
            rel_set[d1][label].append(d2)
        for d1 in rel_set:
            label_list = sorted(rel_set[d1].keys(), reverse=True)
            for hidx, high_label in enumerate(label_list[:-1]):
                for low_label in label_list[hidx + 1:]:
                    for high_d2 in rel_set[d1][high_label]:
                        for low_d2 in rel_set[d1][low_label]:
                            pair_list.append((d1, high_d2, low_d2))
        print('Pair Instance Count:', len(pair_list), end='\n')
        return pair_list

    def get_batch_generator(self):
        while True:
            sample = self.get_batch_static()
            if not sample:
                break
            X1, X1_len, X2, X2_len, Y = sample
            yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len}, Y)

    # 读取每一个batch的数据
    def get_batch_static(self):
        X1 = np.zeros((self.batch_size * 2, self.data1_maxlen), dtype=np.int32)
        X1_len = np.zeros((self.batch_size * 2,), dtype=np.int32)
        X2 = np.zeros((self.batch_size * 2, self.data2_maxlen), dtype=np.int32)
        X2_len = np.zeros((self.batch_size * 2,), dtype=np.int32)
        Y = np.zeros((self.batch_size * 2,), dtype=np.int32)

<<<<<<< HEAD
        # Y[::2] = 1
=======
        Y[::2] = 1
>>>>>>> 6636718fe1739e29021c42bc19be0f43de359b94
        X1[:] = self.fill_word
        X2[:] = self.fill_word
        for i in range(self.batch_size):
            d1, d2p, d2n = random.choice(self.pair_list)
            # print(self.data1)
            d1_cont = list(data1[d1])
            d2p_cont = list(data2[d2p])
            d2n_cont = list(data2[d2n])
            d1_len = min(self.data1_maxlen, len(d1_cont))
            d2p_len = min(self.data2_maxlen, len(d2p_cont))
            d2n_len = min(self.data2_maxlen, len(d2n_cont))
            X1[i * 2, :d1_len], X1_len[i * 2] = d1_cont[:d1_len], d1_len
            X2[i * 2, :d2p_len], X2_len[i * 2] = d2p_cont[:d2p_len], d2p_len
            X1[i * 2 + 1, :d1_len], X1_len[i * 2 + 1] = d1_cont[:d1_len], d1_len
            X2[i * 2 + 1, :d2n_len], X2_len[i * 2 + 1] = d2n_cont[:d2n_len], d2n_len

        return X1, X1_len, X2, X2_len, Y

    def reset(self):
        self.point = 0


def Kernel_layer(mu, sigma):
    def kernel(x):
        return K.tf.exp(-0.5 * (x - mu) * (x - mu) / sigma / sigma)

    return Activation(kernel)


def rank_hinge_loss(y_true, y_pred):
    # output_shape = K.int_shape(y_pred)
    y_pos = Lambda(lambda a: a[::2, :], output_shape=(1,))(y_pred)
    y_neg = Lambda(lambda a: a[1::2, :], output_shape=(1,))(y_pred)
    loss = K.maximum(0., 1.0 + y_neg - y_pos)
    return K.mean(loss)


def rank_lsep_loss(y_true, y_pred):
    # output_shape = K.int_shape(y_pred)
    y_pos = Lambda(lambda a: a[::2, :], output_shape=(1,))(y_pred)
    y_neg = Lambda(lambda a: a[1::2, :], output_shape=(1,))(y_pred)
    expp = K.exp(y_neg - y_pos)
    summ = K.mean(expp)
    return K.log(1 + summ)


def rank_BPMLL_loss(y_true, y_pred):
    # output_shape = K.int_shape(y_pred)
    y_pos = Lambda(lambda a: a[::2, :], output_shape=(1,))(y_pred)
    y_neg = Lambda(lambda a: a[1::2, :], output_shape=(1,))(y_pred)
    expp = K.exp(y_neg - y_pos)
    return K.mean(expp)


def read_data(filename, word_dict=None):
    data = {}
    for line in open(filename):
        line = line.strip().split()
        tid = line[0]
        if word_dict is None:
            data[tid] = list(map(int, line[1:]))
        else:
            data[tid] = []
            for w in line[1:]:
                if w not in word_dict:
                    word_dict[w] = len(word_dict)
                data[tid].append(word_dict[w])
    print('[%s]\n\tData size: %s' % (filename, len(data)), end='\n')
    return data, word_dict


def read_relation(filename, verbose=True):
    data = []
    for line in open(filename):
        line = line.strip().split()
        data.append((int(line[0]), line[1], line[2]))
    if verbose:
        print('[%s]\n\tInstance size: %s' % (filename, len(data)), end='\n')
    return data


data1, _ = read_data('./data/WikiQA/corpus_preprocessed.txt')
data2, _ = read_data('./data/WikiQA/corpus_preprocessed.txt')

query = Input(name='query', shape=(10,))
doc = Input(name='doc', shape=(40,))
# que = Dropout(0.2)(query)
# do = Dropout(0.2)(doc)
# Embedding层把query中每个词映射成词向量
embed = np.float32(np.random.uniform(-0.2, 0.2, [407870, 300]))
embedding = Embedding(407870, 300, weights=[embed], trainable=True)
q_embed = embedding(query)
d_embed = embedding(doc)

# Translation layer build translation matrix M.
mm = Dot(axes=[2, 2], normalize=True)([q_embed, d_embed])

# Kernel - Pooling
KM = []
kernel_num = 21

for i in range(kernel_num):
    mu = 1. / (kernel_num - 1) + (2. * i) / (kernel_num - 1) - 1.0
    sigma = 0.1
    if mu > 1.0:
        sigma = 0.001
        mu = 1.0
    mm_exp = Kernel_layer(mu, sigma)(mm)

    mm_doc_sum = Lambda(lambda x: K.tf.reduce_sum(x, 2))(mm_exp)

    mm_log = Activation(K.tf.log1p)(mm_doc_sum)

    mm_sum = Lambda(lambda x: K.tf.reduce_sum(x, 1))(mm_log)
    KM.append(mm_sum)
# Learning to Rank
# KMM = BatchNormalization(axis=1,center=True,scale=True,beta_regularizer=l2(0.01),gamma_regularizer=l2(0.01))(KM)
Phi = Lambda(lambda x: K.tf.stack(x, 1))(KM)

out_ = Dense(1, kernel_initializer=RandomUniform(minval=-0.014, maxval=0.014), bias_initializer='zeros')(Phi)

model = Model(inputs=[query, doc], outputs=[out_])

<<<<<<< HEAD
model.compile(optimizer=Adam(lr=0.001), loss=rank_hinge_loss)
# model.compile(optimizer=Adam(lr=0.001), loss=rank_lsep_loss)
# model.compile(optimizer=SGD(lr=0.001, momentum=0.5), loss=rank_lsep_loss)
# 0.5 0.9 0.99分别代表2倍 10倍 100倍SGD的速度
# history = model.fit(input_data1, Y, batch_size=100, epochs=200)
num_iters = 300
lossy = []
generator = PairGenerator()
for i_e in range(num_iters):
    genfun = generator.get_batch_generator()
    print('[%s]\t[Train:%s] ' % (time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())), 'train'), end='')
    history = model.fit_generator(
        genfun,
        steps_per_epoch=10,
=======
model.compile(optimizer=Adam(lr=0.001), loss=rank_hinge_loss, metrics=[MAP])
# model.compile(optimizer=SGD(lr=0.001, momentum=0.5), loss=rank_lsep_loss)
# 0.5 0.9 0.99分别代表2倍 10倍 100倍SGD的速度
num_iters = 2000
train_loss = []
valid_loss = []
train_map = []
valid_map = []
rel_train = './data/WikiQA/relation_train.txt'
rel_valid = './data/WikiQA/relation_valid.txt'
pair_generator = PairGenerator(rel_train)
list_generator = PairGenerator(rel_valid)

for i_e in range(num_iters):
    genfun = pair_generator.get_batch_generator()
    genfun_v = list_generator.get_batch_generator()
    print('[%s]\t[Train:%s] ' % (time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())), 'train'), end='')
    his = model.fit_generator(
        genfun,
        steps_per_epoch=10,
        validation_data=genfun_v,
        validation_steps=10,
>>>>>>> 6636718fe1739e29021c42bc19be0f43de359b94
        epochs=1,
        shuffle=False,
        verbose=1
    )  # callbacks=[eval_map])
<<<<<<< HEAD
    print('Iter:%d\tloss=%.6f' % (i_e, history.history['loss'][0]), end='\n')
    if (i_e + 1) % 100 == 0:
        model.save_weights('weights/weights_hinge_%s' % (i_e + 1))
    print(history.history['loss'])

    # y_pre = model.predict(input_data2, batch_size=10)
    #
    # y_pred = np.squeeze(y_pre)


    # with open('y_pre.txt', 'w') as f:
    #     for i in range(len(YY)):
    #         f.write(str(y_pred[i]))
    #         f.write('\n')

    lossy.append(history.history['loss'])
np_lossy = np.array(lossy)
import matplotlib.pyplot as plt
import numpy as np

x = range(len(np_lossy))
plt.plot(x, np_lossy)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('iters')
plt.show()

# np.savetxt('loss.txt', np_lossy)

=======
    print('Iter:%d\ttrain_loss=%.6f\ttrain_map=%.6f' % (i_e, his.history['loss'][0], his.history['m_a_p'][0]), end='\n')
    print('Iter:%d\tvalid_loss=%.6f\tvalid_map=%.6f' % (i_e, his.history['val_loss'][0], his.history['val_m_a_p'][0]),
          end='\n')
    if (i_e + 1) % 100 == 0:
        model.save_weights('weights/Model_test2_%s' % (i_e + 1))

    train_loss.append(his.history['loss'])
    valid_loss.append(his.history['val_loss'])
    train_map.append(his.history['m_a_p'])
    valid_map.append(his.history['val_m_a_p'])

    if i_e % 100 == 0:
        np_train_loss = np.array(train_loss)
        np_valid_loss = np.array(valid_loss)
        np_train_map = np.array(train_map)
        np_valid_map = np.array(valid_map)

        x = range(len(np_train_loss))
        plt.plot(x, np_train_loss, color='r', label='train_loss')
        plt.plot(x, np_valid_loss, color='g', label='valid_loss')
        plt.plot(x, np_train_map, color='b', label='train_map')
        plt.plot(x, np_valid_map, color='k', label='valid_map')
        plt.title('model loss and map')
        plt.ylabel('loss map')
        plt.xlabel('iters')
        plt.savefig('loss-map-%d.png' % i_e)

with open('train_loss.txt', 'w') as f:
    for i in range(len(train_loss)):
        f.write(train_loss[i])
        f.write('\n')
with open('valid_loss.txt', 'w') as f:
    for i in range(len(valid_loss)):
        f.write(valid_loss[i])
        f.write('\n')
with open('train_map.txt', 'w') as f:
    for i in range(len(train_map)):
        f.write(train_map[i])
        f.write('\n')
with open('valid_loss.txt', 'w') as f:
    for i in range(len(valid_map)):
        f.write(valid_map[i])
        f.write('\n')
>>>>>>> 6636718fe1739e29021c42bc19be0f43de359b94
