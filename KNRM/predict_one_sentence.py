from __future__ import absolute_import
from __future__ import print_function

import math
import random
import sys
import time
import traceback
from collections import OrderedDict
import psutil
import six
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.generic_utils import deserialize_keras_object
import tensorflow as tf
import io
import jieba
import os
from itertools import permutations


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

    def rank_crossentropy_loss(self, kwargs=None):
        neg_num = 1
        if isinstance(kwargs, dict) and 'neg_num' in kwargs:
            neg_num = kwargs['neg_num']

        def _cross_entropy_loss(y_true, y_pred):
            y_pos_logits = Lambda(lambda a: a[::(neg_num + 1), :], output_shape=(1,))(y_pred)
            y_pos_labels = Lambda(lambda a: a[::(neg_num + 1), :], output_shape=(1,))(y_true)
            logits_list, labels_list = [y_pos_logits], [y_pos_labels]
            for i in range(neg_num):
                y_neg_logits = Lambda(lambda a: a[(i + 1)::(neg_num + 1), :], output_shape=(1,))(y_pred)
                y_neg_labels = Lambda(lambda a: a[(i + 1)::(neg_num + 1), :], output_shape=(1,))(y_true)
                logits_list.append(y_neg_logits)
                labels_list.append(y_neg_labels)
            logits = tf.concat(logits_list, axis=1)
            labels = tf.concat(labels_list, axis=1)
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

        return _cross_entropy_loss


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

    def mrr(self, y_true, y_pred, rel_threshold=0.):
        k = 10
        s = 0.
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

    def precision(self, k=10):
        def top_k(y_true, y_pred, rel_threshold=0.):
            if k <= 0:
                return 0.
            s = 0.
            y_true = self._to_list(np.squeeze(y_true).tolist())
            y_pred = self._to_list(np.squeeze(y_pred).tolist())
            c = list(zip(y_true, y_pred))
            random.shuffle(c)
            c = sorted(c, key=lambda x: x[1], reverse=True)
            ipos = 0
            prec = 0.
            for i, (g, p) in enumerate(c):
                if i >= k:
                    break
                if g > rel_threshold:
                    prec += 1
            prec /= k
            return prec

        return top_k

    # compute recall@k
    # the input is all documents under a single query
    def recall(self, k=10):
        def top_k(y_true, y_pred, rel_threshold=0.):
            if k <= 0:
                return 0.
            s = 0.
            y_true = self._to_list(
                np.squeeze(y_true).tolist())  # y_true: the ground truth scores for documents under a query
            y_pred = self._to_list(
                np.squeeze(y_pred).tolist())  # y_pred: the predicted scores for documents under a query
            pos_count = sum(
                i > rel_threshold for i in y_true)  # total number of positive documents under this query
            c = list(zip(y_true, y_pred))
            random.shuffle(c)
            c = sorted(c, key=lambda x: x[1], reverse=True)
            ipos = 0
            recall = 0.
            for i, (g, p) in enumerate(c):
                if i >= k:
                    break
                if g > rel_threshold:
                    recall += 1
            recall /= pos_count
            return recall

        return top_k

    def mse(self, y_true, y_pred, rel_threshold=0.):
        s = 0.
        y_true = self._to_list(np.squeeze(y_true).tolist())
        y_pred = self._to_list(np.squeeze(y_pred).tolist())
        return np.mean(np.square(y_pred - y_true), axis=-1)

    def accuracy(self, y_true, y_pred):
        y_true = self._to_list(np.squeeze(y_true).tolist())
        y_pred = self._to_list(np.squeeze(y_pred).tolist())
        y_true_idx = np.argmax(y_true, axis=1)
        y_pred_idx = np.argmax(y_pred, axis=1)
        assert y_true_idx.shape == y_pred_idx.shape
        return 1.0 * np.sum(y_true_idx == y_pred_idx) / len(y_true)


# Read Embedding File
def read_embedding(filename):
    embed = {}
    for line in open(filename):
        line = line.strip().split()
        embed[int(line[0])] = list(map(float, line[1:]))
    print('[%s]\n\tEmbedding size: %d' % (filename, len(embed)), end='\n')
    return embed


def read_data(filename, word_dict=None):
    data = {}
    for line in open(filename):
        line = line.strip().split()
        tid = line[0]
        if word_dict is None:
            data[tid] = list(map(int, line[1:]))
        else:
            data[tid] = []
            for w in line[2:]:
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


def load_model(config):
    global_conf = config["global"]
    model_type = global_conf['model_type']
    if model_type == 'JSON':
        mo = Model.from_config(config['model'])
    elif model_type == 'PY':
        model_config = config['model']['setting']
        model_config.update(config['inputs']['share'])
        sys.path.insert(0, config['model']['model_path'])

        model = import_object(config['model']['model_py'], model_config)
        mo = model.build()
    return mo


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
    "net_name": "predict_one_sentence",
    "global": {
        "model_type": "PY",
        "weights_file": "examples/wikiqa/weights/knrm.wikiqa.weights",
        "save_weights_iters": 100,
        "num_iters": 100,
        "display_interval": 10,
        "test_weights_iters": 100,
        "optimizer": "adam",
        "learning_rate": 0.001
    },
    "inputs": {
        "share": {
            "text1_corpus": "./data/WikiQA/corpus_preprocessed.txt",
            "text2_corpus": "./data/WikiQA/corpus_preprocessed.txt",
            # "corpus_len": 50699,
            "use_dpool": False,
            "embed_size": 300,
            # "embed_path": "./data/WikiQA/embed_glove_d300_norm",
            "vocab_size": 407870,
            "train_embed": True,
            "target_mode": "ranking",
            "text1_maxlen": 10,
            "text2_maxlen": 40
        },
        "predict": {
            "input_type": "ListGenerator",
            "phase": "PREDICT",
            "batch_list": 10,
            "batch_size": 100,
            "relation_file": "./data/WikiQA/relation_test.txt"
        }
    },
    "outputs": {
        "predict": {
            "save_format": "TREC",
            "save_path": "predict.test.knrm_ranking.wikiqa.txt"
        }
    },
    "model": {
        "model_path": "",
        "model_py": "knrm_model.KNRM",
        "setting": {
            "kernel_num": 21,
            "sigma": 0.1,
            "exact_sigma": 0.001,
            "dropout_rate": 0.0
        }
    },
    "losses": [
        {
            "object_name": "rank_hinge_loss",
            "object_params": {"margin": 1.0}
        }
    ],
    "metrics": ["ndcg@3", "ndcg@5", "map"]
}


class PairGeneratorPredict(object):
    def __init__(self, config):
        self.__name = 'PairGeneratorPredict'
        self.samples_index = []
        self.config = config
        self.data1 = config['data1']
        self.data2 = config['data2']
        self.rel = self.read_samples_index()
        self.batch_size = config['batch_size']
        self.data1_maxlen = config['text1_maxlen']
        self.data2_maxlen = config['text2_maxlen']
        self.fill_word = config['vocab_size'] - 1
        self.target_mode = config['target_mode']
        self.class_num = 5
        self.point = 0
        self.total_rel_num = len(self.rel)
        self.check_list = ['text1_corpus', 'text2_corpus', 'text1_maxlen', 'text2_maxlen', 'batch_size',
                           'vocab_size']
        if not self.check():
            raise TypeError('[ListGenerator] parameter check wrong.')

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
            with open('C:\\Users\DELL\MatchZoo\data\WikiQA\pos_samples_index.txt', 'r', encoding='utf-8') as p:
                self.samples_index = p.read().split()
        rel = [(0, '0', qid2) for qid2 in self.samples_index]
        return rel

    def read_samples_index_1(self):

        self.samples_index = []
        with open('./data/WikiQA/corpus_preprocessed.txt', 'r', encoding='utf-8') as p:
            data = p.readlines()
        for line in data:
            self.samples_index.append(line.split('\t', maxsplit=1)[0])
        rel = [(0, '0', qid2) for qid2 in self.samples_index]
        return rel


def predict_one_sentence(config, input_str_ids, corpus):
    # initial data generator
    input_conf = config['inputs']
    share_input_conf = input_conf['share']

    # collect embedding
    if 'embed_path' in share_input_conf:
        embed_dict = read_embedding(filename=share_input_conf['embed_path'])
        _PAD_ = share_input_conf['vocab_size'] - 1
        embed_dict[_PAD_] = np.zeros((share_input_conf['embed_size'],), dtype=np.float32)
        embed = np.float32(
            np.random.uniform(-0.02, 0.02, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
        share_input_conf['embed'] = convert_embed_2_numpy(embed_dict, embed=embed)
    else:
        embed = np.float32(
            np.random.uniform(-0.2, 0.2, [share_input_conf['vocab_size'], share_input_conf['embed_size']]))
        share_input_conf['embed'] = embed
    print('[Embedding] Embedding Load Done.', end='\n')

    # list all input tags and construct tags config
    input_predict_conf = OrderedDict()
    for tag in input_conf.keys():
        if 'phase' not in input_conf[tag]:
            continue
        if input_conf[tag]['phase'] == 'PREDICT':
            input_predict_conf[tag] = {}
            input_predict_conf[tag].update(share_input_conf)
            input_predict_conf[tag].update(input_conf[tag])
    print('[Input] Process Input Tags. %s in PREDICT.' % (input_predict_conf.keys()), end='\n')

    # collect dataset identification
    dataset = {}
    for tag in input_conf:
        if tag == 'share' or input_conf[tag]['phase'] == 'PREDICT':
            if 'text1_corpus' in input_conf[tag]:
                datapath = input_conf[tag]['text1_corpus']
                if datapath not in dataset:
                    dataset[datapath], _ = read_data(datapath)
            if 'text2_corpus' in input_conf[tag]:
                datapath = input_conf[tag]['text2_corpus']
                if datapath not in dataset:
                    dataset[datapath], _ = read_data(datapath)
    print('[Dataset] %s Dataset Load Done.' % len(dataset), end='\n')

    ######## Load Model ########
    global_conf = config["global"]
    weights_file = 'examples/wikiqa/weights/knrm.wikiqa.weights.300'

    # weights_file = str(global_conf['weights_file']) + '.' + str(global_conf['test_weights_iters'])
    # weights_file = 'knrm.wights.300_2'
    if not os.path.exists(weights_file):
        weights_file = os.path.join('..', weights_file)  # 加入到工程中
    model = load_model(config)
    model.load_weights(weights_file)

    eval_metrics = OrderedDict()
    eval_metrics["ndcg@3"] = Metrics().ndcg(3)
    eval_metrics["ndcg@5"] = Metrics().ndcg(5)
    eval_metrics["map"] = Metrics().map

    predict_gen = OrderedDict()
    for tag, conf in input_predict_conf.items():
        conf['data1'] = dataset[conf['text1_corpus']]
        conf['data2'] = dataset[conf['text2_corpus']]
        # 在语料数据的最后加上当前语句编号
        conf['data1'][str(0)] = input_str_ids
        predict_gen[tag] = PairGeneratorPredict(config=conf)

    print('[%s]\t[Predict] @ %s ' % (time.strftime('%m-%d-%Y %H:%M:%S', time.localtime(time.time())), 'Predict'),
          end='\n')

    for tag, generator in predict_gen.items():
        genfun = generator.get_batch_generator()
        res_scores = []
        label = {}
        for input_data, y_true in genfun:
            if len(y_true) != 0:
                y_pred = model.predict(input_data, batch_size=len(y_true))

                for i, (qid1, qid2) in enumerate(input_data['ID']):
                    res_scores.append((qid2, y_pred[i]))
                    # label[qid2] = y_pred[i][0]
                    print(qid2, y_pred[i])
            else:
                break
        # with open('learned_label.txt', 'w') as f:
        #     for k, v in label.items():
        #         f.write(k)
        #         f.write(':')
        #         f.write(str(v))
        #         f.write('\n')

        generator.reset()
        recommendation = OrderedDict()
        top10_dinfo = sorted(res_scores, key=lambda d: d[1][0], reverse=True)[:10]

        # # evaluation
        # top10_true_y = []
        # top10_pred_y = []
        # for i in range(len(top10_dinfo)):
        #     top10_true_y.append(top10_dinfo[i])
        #     top10_pred_y.append(top10_dinfo[i][0])
        # print(top10_dinfo[2])
        # print(top10_dinfo[1][0])
        #
        # res = dict([[k, 0.] for k in eval_metrics.keys()])  # 初始化为{"ndcg@3":0.0, "ndcg@5":0.0, "map":0.0}
        # for k, eval_func in eval_metrics.items():
        #     res[k] += eval_func(y_true=np.array(top10_true_y), y_pred=top10_pred_y)
        # num_valid = len(top10_dinfo) - 1
        # print('[Predict] results: ', '\t'.join(['%s=%f' % (k, v / num_valid) for k, v in res.items()]), end='\n')

        for qid2, score in top10_dinfo:
            # print(qid2, score)
            recommendation['语料索引：{}，得分：{:.6f}，详细内容'.format(qid2, score[0])] = corpus[qid2]
        sys.stdout.flush()
    return recommendation


#  # 测试结果，不是输入向量少两个的原因
# corpus = {}
# with open('./data/WikiQA/corpus.txt', 'r', encoding='utf-8') as p:
#     lines = p.readlines()
#     for line in lines:
#         x = line.strip().split(' ', maxsplit=1)
#         if len(x) == 2:
#             corpus[x[0]] = x[1]
#         else:
#             corpus[x[0]] = '空'
# input_str_ids = [5523, 1135, 37104, 25073, 11369, 315, 0, 4366, 7559, 0]
# recommendation = predict_one_sentence(config, input_str_ids, corpus)
# for key, values in recommendation.items():
#     print(key + ':' + values)


def main():
    word_dict = load_word_dict('./data/WikiQA/word_dict.txt', english=False)

    with open('./data/WikiQA/chinese_stopwords', 'r', encoding='utf-8') as f:
        chinese_stopwords = f.read().split()
    corpus = {}
    with open('./data/WikiQA/corpus.txt', 'r', encoding='utf-8') as p:
        lines = p.readlines()
        for line in lines:
            x = line.strip().split(' ', maxsplit=1)
            if len(x) == 2:
                corpus[x[0]] = x[1]
            else:
                corpus[x[0]] = '空'
    while (1):
        print('Please input your sentence:')
        input_str = input()
        if input_str == '' or input_str.isspace():
            print('See you next time!')
            break
        else:
            input_str = input_str.strip()
            input_str = input_str.replace(' ', '')
            input_str = list(jieba.cut(input_str))
            input_str = [w for w in input_str if w not in chinese_stopwords]  # 过滤掉停用词
            print("输入：{}".format(input_str))
            input_str_ids = [word_dict[w] if w in word_dict.keys() else word_dict['<unk>'] for w in input_str]
            if len(input_str_ids) < 8:
                input_str_ids = input_str_ids * 8
            recommendation = predict_one_sentence(config, input_str_ids, corpus)

            for key, values in recommendation.items():
                print(key + ':' + values)


if __name__ == '__main__':
    main()
