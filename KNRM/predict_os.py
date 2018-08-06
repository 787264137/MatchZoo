from __future__ import absolute_import
from __future__ import print_function

import math
import random
import sys
import time
import traceback
from collections import OrderedDict
from keras.layers import *
from keras.models import Model
import io
import jieba
import os
import knrm_model
np.random.seed(49999)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class RankLosses(object):
    def rank_hinge_loss(self, kwargs=None):
        margin = 1.
        if isinstance(kwargs, dict) and 'margin' in kwargs:
            margin = kwargs['margin']

        def _margin_loss(y_true, y_pred):
            # output_shape = K.int_shape(y_pred)
            y_pos = Lambda(lambda a: a[::2, :], output_shape=(1,))(y_pred)
            y_neg = Lambda(lambda a: a[1::2, :], output_shape=(1,))(y_pred)
            loss = K.maximum(0., margin + y_neg - y_pos)
            return K.mean(loss)

        return _margin_loss


class Metrics(object):
    def _to_list(self, x):
        if isinstance(x, list):
            return x
        return [x]

    def map(self, y_true, y_pred, rel_threshold=0):
        s = 0.
        y_true = self._to_list(np.squeeze(y_true).tolist())
        y_pred = self._to_list(np.squeeze(y_pred).tolist())
        c = list(zip(y_true, y_pred))
        random.shuffle(c)
        c = sorted(c, key=lambda x: x[1], reverse=True)
        ipos = 0
        for j, (g, p) in enumerate(c):
            if g > rel_threshold:  # g = 0 的表示不相关的文档
                ipos += 1.
                s += ipos / (j + 1.)
        if ipos == 0:
            s = 0.
        else:
            s /= ipos
        return s

    def ndcg(self, k=10):
        def top_k(y_true, y_pred, rel_threshold=0.):
            if k <= 0.:
                return 0.
            s = 0.
            y_true = self._to_list(np.squeeze(y_true).tolist())
            y_pred = self._to_list(np.squeeze(y_pred).tolist())
            c = list(zip(y_true, y_pred))
            random.shuffle(c)
            c_g = sorted(c, key=lambda x: x[0], reverse=True)
            c_p = sorted(c, key=lambda x: x[1], reverse=True)
            idcg = 0.
            ndcg = 0.
            for i, (g, p) in enumerate(c_g):
                if i >= k:
                    break
                if g > rel_threshold:
                    idcg += (math.pow(2., g) - 1.) / math.log(2. + i)
            for i, (g, p) in enumerate(c_p):
                if i >= k:
                    break
                if g > rel_threshold:
                    ndcg += (math.pow(2., g) - 1.) / math.log(2. + i)
            if idcg == 0.:
                return 0.
            else:
                return ndcg / idcg

        return top_k


# Read Embedding File
def read_embedding(filename):
    embed = {}
    for line in open(filename):
        line = line.strip().split()
        embed[int(line[0])] = list(map(float, line[1:]))
    print('[%s]\n\tEmbedding size: %d' % (filename, len(embed)), end='\n')
    return embed


def read_data(filename):
    data = {}
    for line in open(filename):
        line = line.strip().split()
        tid = line[0]
        data[tid] = list(map(int, line[1:]))
    return data


def read_relation(filename, verbose=True):
    data = []
    for line in open(filename):
        line = line.strip().split()
        data.append((int(line[0]), line[1], line[2]))
    if verbose:
        print('[%s]\n\tInstance size: %s' % (filename, len(data)), end='\n')
    return data


def load_word_dict(word_map_file, english=True):
    """ file -> {word: index} """
    word_dict = {}
    for line in io.open(word_map_file, encoding='utf8'):
        if english:
            line = line.split(' ')
        else:
            line = line.split(':')
        try:
            word_dict[line[0]] = int(line[1])
        except:
            print(line)
            continue
    return word_dict


# Convert Embedding Dict 2 numpy array
def convert_embed_2_numpy(embed_dict, max_size=0, embed=None):
    feat_size = len(embed_dict[list(embed_dict.keys())[0]])
    if embed is None:
        embed = np.zeros((max_size, feat_size), dtype=np.float32)

    if len(embed_dict) > len(embed):
        raise Exception("vocab_size %d is larger than embed_size %d, change the vocab_size in the config!"
                        % (len(embed_dict), len(embed)))

    for k, value in embed_dict.items():
        embed[k - 1] = np.array(value)
    print('Generate numpy embed:', str(embed.shape), end='\n')
    return embed


def import_object(import_str, *args, **kwargs):
    return import_class(import_str)(*args, **kwargs)


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' %
                          (class_str,
                           traceback.format_exception(*sys.exc_info())))


config = {
    "text2_corpus": "./data/WikiQA/corpus_preprocessed.txt",
    "class_num": 2,
    "embed_size": 300,
    "vocab_size": 407870,
    "train_embed": True,
    "text1_maxlen": 10,
    "text2_maxlen": 40,
    'embed_path': "./data/WikiQA/embed_glove_d300_norm",
    "batch_list": 10,
    "batch_size": 100,

    "model_py": "knrm_model.KNRM",

    "kernel_num": 21,
    "sigma": 0.1,
    "exact_sigma": 0.001,
    "dropout_rate": 0.0,

    "object_name": "rank_hinge_loss",
    "object_params": {"margin": 1.0},
}


class PairGeneratorPredict(object):
    def __init__(self, config):
        self.__name = 'PairGeneratorPredict'
        self.config = config
        self.samples_index = []
        self.data1 = self.config['data1']
        self.data2 = self.config['data2']
        self.rel = self.read_samples_index()
        self.batch_size = self.config['batch_size']
        self.data1_maxlen = self.config['text1_maxlen']
        self.data2_maxlen = self.config['text2_maxlen']
        self.fill_word = self.config['vocab_size'] - 1
        self.class_num = self.config['class_num']
        self.point = 0
        self.total_rel_num = len(self.rel)
        self.check_list = ['text2_corpus', 'text1_maxlen', 'text2_maxlen', 'batch_size',
                           'vocab_size']
        if not self.check():
            raise TypeError('[PairGeneratorPredict] parameter check wrong.')

    def check(self):
        for e in self.check_list:
            if e not in self.config:
                print('[%s] Error %s not in config' % (self.__name, e), end='\n')
                return False
        return True

    def get_batch_generator(self):
        while True:
            sample = self.get_batch()
            if not sample:
                break
            X1, X1_len, X2, X2_len, Y, ID_pairs = sample
            yield ({'query': X1, 'query_len': X1_len, 'doc': X2, 'doc_len': X2_len, 'ID': ID_pairs}, Y)

    def get_batch(self):
        curr_batch_size = self.batch_size
        if self.total_rel_num - self.point < self.batch_size:
            curr_batch_size = self.total_rel_num - self.point
        X1 = np.zeros((curr_batch_size, self.data1_maxlen), dtype=np.int32)
        X1_len = np.zeros((curr_batch_size,), dtype=np.int32)
        X2 = np.zeros((curr_batch_size, self.data2_maxlen), dtype=np.int32)
        X2_len = np.zeros((curr_batch_size,), dtype=np.int32)
        # Y = np.zeros((curr_batch_size,), dtype=np.int32)
        Y = np.zeros((curr_batch_size, self.class_num), dtype=np.int32)
        ID_pairs = []

        X1[:] = self.fill_word
        X2[:] = self.fill_word
        for i in range(curr_batch_size):
            label, d1, d2 = self.rel[self.point]
            self.point += 1
            if d2 not in self.data2.keys():
                continue
            d1_len = min(self.data1_maxlen, len(self.data1[d1]))
            d2_len = min(self.data2_maxlen, len(self.data2[d2]))
            X1[i, :d1_len], X1_len[i] = self.data1[d1][:d1_len], d1_len
            X2[i, :d2_len], X2_len[i] = self.data2[d2][:d2_len], d2_len
            ID_pairs.append((d1, d2))
            # Y[i, label] = 1.
        return X1, X1_len, X2, X2_len, Y, ID_pairs

    def reset(self):
        self.point = 0

    def read_samples_index(self):
        self.samples_index = []
        if not self.samples_index:
            with open('./data/WikiQA/documents_index.txt', 'r', encoding='utf-8') as p:
                self.samples_index = p.read().split()
        rel = [(0, '0', qid2) for qid2 in self.samples_index]
        return rel


def predict_one_sentence(embed,config, weights_file,input_str_ids, corpus):

    config['embed'] = embed
    config['data1'][str(0)] = input_str_ids

    # Load Model
    m_start = time.time()
    model = knrm_model.KNRM(config).build()
    m_end = time.time()
    print('predict===loading model costs:%f.' % (m_end - m_start), end='\n')

    w_start = time.time()
    model.load_weights(weights_file)
    w_end = time.time()
    print('predict===loading weights costs:%f.' % (w_end - w_start), end='\n')

    c_start = time.time()
    predict_gen = PairGeneratorPredict(config=config)
    genfun = predict_gen.get_batch_generator()
    res_scores = []

    for input_data, y_true in genfun:
        if len(y_true) != 0:
            y_pred = model.predict(input_data, batch_size=len(y_true))

            for i, (qid1, qid2) in enumerate(input_data['ID']):
                res_scores.append((qid2, y_pred[i]))
        else:
            break

    recommendation = OrderedDict()
    top10_dinfo = sorted(res_scores, key=lambda d: d[1][0], reverse=True)[:10]

    for qid2, score in top10_dinfo:
        recommendation['语料索引：{}，得分：{:.6f}，详细内容'.format(qid2, score[0])] = corpus[qid2]
    c_end = time.time()

    sys.stdout.flush()
    print('predict===calculating scores costs:%f.' % (c_end - c_start), end='\n')
    return recommendation


def main():
    print('#############Pre Load Start#####################')
    wd_start = time.time()
    word_dict = load_word_dict('./data/WikiQA/word_dict.txt', english=False)
    wd_end = time.time()
    print('loading word_dict costs:%f' % (wd_end - wd_start))

    sw_start = time.time()
    with open('./data/WikiQA/chinese_stopwords', 'r', encoding='utf-8') as f:
        chinese_stopwords = f.read().split()
    sw_end = time.time()
    print('loading stopwords costs:%f' % (sw_end - sw_start))

    cp_start = time.time()
    corpus = {}
    with open('./data/WikiQA/corpus.txt', 'r', encoding='utf-8') as p:
        lines = p.readlines()
        for line in lines:
            x = line.strip().split(' ', maxsplit=1)
            if len(x) == 2:
                corpus[x[0]] = x[1]
            else:
                corpus[x[0]] = '空'
    cp_end = time.time()
    print('loading corpus costs:%f' % (cp_end - cp_start))

    e_start = time.time()
    embed = np.float32(np.random.uniform(-0.2, 0.2, [config['vocab_size'], config['embed_size']]))
    e_end = time.time()
    print('predict===loading embeding costs:%f.' % (e_end - e_start), end='\n')

    d_start = time.time()
    config['data1'] = {}
    config['data2'] = read_data(config['text2_corpus'])
    d_end = time.time()
    print('predict===loading dataset costs:%f.' % (d_end - d_start), end='\n')

    weights_file = 'examples/wikiqa/weights/knrm.wikiqa.weights.200'

    print('#############Pre Load End#####################')
    while True:
        print('Please input your sentence:')
        input_str = input()
        if input_str == '' or input_str.isspace():
            print('See you next time!')
            break
        else:
            input_process_start = time.time()
            input_str = input_str.strip()
            input_str = input_str.replace(' ', '')
            input_str = list(jieba.cut(input_str))
            input_str = [w for w in input_str if w not in chinese_stopwords]  # 过滤掉停用词
            print("输入：{}".format(input_str))
            input_str_ids = [word_dict[w] for w in input_str if w in word_dict.keys()]
            if len(input_str_ids) < 8:
                input_str_ids = input_str_ids * 8
            input_process_end = time.time()
            print('inputing preprocess costs:%f' % (input_process_end - input_process_start))

            predict_start = time.time()
            recommendation = predict_one_sentence(embed,config, weights_file,input_str_ids, corpus)
            predict_end = time.time()
            print('predicting costs:%f' % (predict_end - predict_start))

            for key, values in recommendation.items():
                print(key + ':' + values)


if __name__ == '__main__':
    main()
