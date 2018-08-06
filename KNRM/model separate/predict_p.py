from __future__ import absolute_import
from __future__ import print_function

import sys
import time
import traceback
from collections import OrderedDict
from keras.layers import *
import io
import jieba
import os
from knrm_p import KNRM

np.random.seed(49999)


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


def read_samples_index(filename):
    with open(filename, 'r', encoding='utf-8') as p:
        samples_index = p.read().split()
    rel = [qid2 for qid2 in samples_index]
    return rel


config = {
    "text2_corpus": "./data/WikiQA/corpus_preprocessed.txt",
    "class_num": 2,
    "embed_size": 300,
    "text1_maxlen": 10,
    "text2_maxlen": 40,
    "vocab_size": 407870,
    "batch_size": 100,

    "object_name": "rank_hinge_loss",
    "object_params": {"margin": 1.0},

    'kernel_num': 21,
    'words_dir': 'phi/'
}


class PairGeneratorPredict(object):
    def __init__(self, config, files):
        self.__name = 'PairGeneratorPredict'
        self.config = config
        self.files = files
        self.words_dir = self.config['words_dir']
        self.rel = self.config['sample_index']
        self.phis = self.read_phis(files)
        self.total_phi_num = len(self.rel)
        self.batch_size = self.config['batch_size']
        self.phi_len = self.config['kernel_num']
        self.fill_word = self.config['vocab_size'] - 1
        self.class_num = self.config['class_num']
        self.point = 0

    def reset(self):
        self.point = 0

    def read_phis(self, files):

        re_start = time.time()
        phis = []
        for filename in files:
            phi = {}
            with open(self.words_dir + filename, 'r', encoding='utf-8') as f:
                while True:
                    line = f.readline().strip()
                    if not line: break
                    itms = line.split(' ')
                    did, p = itms[0], [np.float32(x) for x in itms[1:]]
                    phi[did] = p
            phis.append(phi)
        re_end = time.time()
        print('读取本地的特征文件所花费的时间:%d' % (re_end - re_start))
        return phis

    def get_batch_generator(self):
        while True:
            Xs = []
            IDs = []

            for phi in self.phis:
                sample = self.get_batch(phi)
                if not sample:
                    break
                X, ID = sample
                IDs.append(ID)
                Xs.append(X)
            if len(self.files) == 1:
                Xs = Xs[0]
            else:
                Xs = np.array(Xs)
                # print(Xs.shape)
                matrix = np.zeros(shape=Xs[0].shape)
                for X in Xs[0:]:
                    # print(X)
                    # print('---------')
                    if X.shape == matrix.shape:
                        matrix += X
                Xs = matrix
            # print(Xs[0].shape)
            # print(type(Xs))
            yield {'Xs': Xs, 'IDs': IDs}

    def get_batch(self, phi):
        curr_batch_size = self.batch_size
        if self.total_phi_num - self.point < self.batch_size:
            curr_batch_size = self.total_phi_num - self.point
        X = np.zeros((curr_batch_size, self.phi_len), dtype=np.float32)
        X[:] = self.fill_word
        ID = []
        for i in range(curr_batch_size):
            did = self.rel[self.point]
            self.point += 1
            d_len = self.phi_len
            X[i, :d_len] = phi[did][:d_len]
            ID.append(did)
        return X, ID


def predict_one_sentence(model, config, input_str_ids, corpus):
    c1_start = time.time()
    files = []
    for word_id in input_str_ids:
        for filename in os.listdir(config['words_dir']):
            # filename = '诸葛亮_7502_phi.txt'
            if str(word_id) == filename.split('.')[0].split('_')[1]:
                files.append(filename)
    if len(files) != len(input_str_ids):
        print('word file not found')
    print(files)
    c1_end = time.time()
    print('predict=C1==calculating scores costs:%f.' % (c1_end - c1_start), end='\n')

    generator = PairGeneratorPredict(config, files)  # 17s
    c2_start = time.time()
    genfun = generator.get_batch_generator()
    c2_end = time.time()
    print('predict=C2==calculating scores costs:%f.' % (c2_end - c2_start), end='\n')

    c3_start = time.time()
    res_scores = []
    for input_data in genfun:
        if len(input_data['Xs']) != 0:
            y_pred = model.predict(input_data, batch_size=len(input_data['Xs']))
            for i, did in enumerate(input_data['IDs'][0]):
                res_scores.append((did, y_pred[i]))
        else:
            break
    generator.reset()

    c3_end = time.time()
    print('predict=C3==calculating scores costs:%f.' % (c3_end - c3_start), end='\n')

    # costs: 0.086258
    c4_start = time.time()
    recommendation = OrderedDict()
    top10_dinfo = sorted(res_scores, key=lambda d: d[1][0], reverse=True)[:10]

    for qid2, score in top10_dinfo:
        recommendation['语料索引：{}，得分：{:.6f}，详细内容'.format(qid2, score[0])] = corpus[qid2]
    # -----------------
    c4_end = time.time()
    print('predict=C4==calculating scores costs:%f.' % (c4_end - c4_start), end='\n')
    return recommendation


def main():
    print('#############Pre Load Start#####################')
    wd_start = time.time()
    word_dict = load_word_dict('data/word_dict.txt', english=False)
    wd_end = time.time()
    print('loading word_dict costs:%f' % (wd_end - wd_start))

    sw_start = time.time()
    with open('data/chinese_stopwords', 'r', encoding='utf-8') as f:
        chinese_stopwords = f.read().split()
    sw_end = time.time()
    print('loading stopwords costs:%f' % (sw_end - sw_start))

    cp_start = time.time()
    corpus = {}
    with open('data/corpus.txt', 'r', encoding='utf-8') as p:
        lines = p.readlines()
        for line in lines:
            x = line.strip().split(' ', maxsplit=1)
            if len(x) == 2:
                corpus[x[0]] = x[1]
            else:
                corpus[x[0]] = '空'
    cp_end = time.time()
    print('loading corpus costs:%f' % (cp_end - cp_start))

    m_start = time.time()
    weights_file = '../examples/wikiqa/weights/knrm.wikiqa.weights.150'
    model = KNRM(config).build()
    model.load_weights(weights_file, by_name=True)
    m_end = time.time()
    print('loading model costs:%f' % (m_end - m_start))

    embed = np.float32(np.random.uniform(-0.2, 0.2, [config['vocab_size'], config['embed_size']]))
    _PAD_ = config['vocab_size'] - 1
    embed[_PAD_] = np.zeros((config['embed_size'],), dtype=np.float32)

    config['embed'] = embed
    config['sample_index'] = read_samples_index('data/documents_index.txt')
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
            print(input_str_ids)
            # if len(input_str_ids) < 8:
            #     input_str_ids = input_str_ids * 8

            input_process_end = time.time()
            print('inputing preprocess costs:%f' % (input_process_end - input_process_start))

            predict_start = time.time()
            recommendation = predict_one_sentence(model, config, input_str_ids, corpus)
            predict_end = time.time()
            print('predicting costs:%f' % (predict_end - predict_start))

            for key, values in recommendation.items():
                print(key + ':' + values)


if __name__ == '__main__':
    main()


