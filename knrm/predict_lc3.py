import sys
import traceback
from keras.layers import *
import io
import json
import knrm_lc as knrm_lc
import jieba
import tensorflow as tf
import os

config = tf.ConfigProto()
tf.ConfigProto(allow_soft_placement=True)
tf.ConfigProto(log_device_placement=True)
sess = tf.Session(config=config)

np.random.seed(49999)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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


def get_query(query_path):
    querys_index = []
    querys = []
    for line in open(query_path, 'r', encoding='utf-8').readlines():
        index, query = check(line)
        if int(index) not in querys_index:
            querys_index.append(index)
            querys.append(query)
    return querys_index, querys


def check(line):
    subs = line.split(' ')
    if len(subs) == 2:
        return subs[0], subs[1]
    else:
        return 0, 0


def getwords_query(querys, stopwords):
    words = []
    jieba.load_userdict('data/word_dict_userdict.txt')
    for q in querys:
        line_cut = list(jieba.cut(q))
        line_d_stopwords = [w for w in line_cut if w not in stopwords]
        line_chinese = [w for w in line_d_stopwords if is_chinese(w)]
        for word in line_chinese:
            if word not in words:
                words.append(word)
    return words


def get_stopwords(path):
    with open(path, 'r', encoding='utf-8') as f:
        chinese_stopwords = f.read().split()
    return chinese_stopwords


def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


def get_words_dict(word_dict_path):
    with open(word_dict_path, 'r', encoding='utf-8') as f:
        words_dict = {}
        while True:
            line = f.readline().strip()
            if not line: break
            word, word_digit = line.split(':')
            words_dict[word] = word_digit
    return words_dict


def convert_to_digit(words, words_dict):
    words_others = []
    words_digit = []
    words_chn = []
    for word in words:
        if word in words_dict.keys():
            words_digit.append(words_dict[word])
            words_chn.append(word)
        else:
            words_others.append(word)
    return words_chn, words_digit, words_others


config = {
    "text2_corpus": "./data/corpus_preprocessed.txt",
    "class_num": 2,
    "embed_size": 300,
    "vocab_size": 407870,
    "train_embed": True,
    "text1_maxlen": 10,
    "text2_maxlen": 40,
    # 'embed_path': "./data/embed_glove_d300_norm",
    "batch_list": 10,
    "batch_size": 10,

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
            with open('./data/documents_index.txt', 'r', encoding='utf-8') as p:
                self.samples_index = p.read().split()
        rel = [(0, '0', qid2) for qid2 in self.samples_index]
        return rel


def predict_lc(model, embed, config, words_digit,words_chn, filename):
    config['data1'] = {}
    config['data2'] = read_data(config['text2_corpus'])
    config['embed'] = embed
    phi = {}
    for i in range(len(words_digit)):
        print(i)
        print(words_digit[i],words_chn[i])
        config['data1'][str(0)] = words_digit[i]
        # 由于模型只是进行核映射和log等运算，因此不需要权重文件
        predict_gen = PairGeneratorPredict(config=config)
        genfun = predict_gen.get_batch_generator()
        res_scores = {}
        for input_data, y_true in genfun:
            if len(y_true) != 0:
                y_pred = model.predict(input_data, batch_size=len(y_true))
                for j, (qid1, qid2) in enumerate(input_data['ID']):
                    res_scores[qid2] = y_pred[j]
            else:
                break
        phi[words_digit[i]] = res_scores
        predict_gen.reset()
    js_obj = json.dumps(phi)
    file_object = open(filename, 'w')
    file_object.write(js_obj)
    file_object.close()
    return 'local calculation completed!'


if __name__ == '__main__':
    embed = np.float32(np.random.uniform(-0.2, 0.2, [config['vocab_size'], config['embed_size']]))
    _PAD_ = config['vocab_size'] - 1
    embed[_PAD_] = np.zeros((config['embed_size'],), dtype=np.float32)

    chinese_stopwords = get_stopwords('data/chinese_stopwords')
    query_indexes, querys = get_query('data/querys_ch.txt')
    words = getwords_query(querys, chinese_stopwords)
    words_dict = get_words_dict('data/word_dict.txt')
    words_chn, words_digit, words_others = convert_to_digit(words, words_dict)

    config['embed'] = embed
    model = knrm_lc.KNRM(config).build()

    filename = 'phi_6000_11000.json'
    start_point = 6000
    stop_point = 11000

    predict_lc(model, embed, config, words_digit[start_point:stop_point],words_chn[start_point:stop_point], filename)
