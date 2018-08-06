import codecs
import os
import random
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import SnowballStemmer
import jieba
import numpy as np
import six
import sys


class Preprocess(object):
    _valid_lang = ['en', 'cn']
    _stemmer = SnowballStemmer('english')

    def __init__(self,
                 word_seg_config={},
                 doc_filter_config={},
                 word_stem_config={},
                 word_lower_config={},
                 word_filter_config={},
                 word_index_config={}
                 ):
        # set default configuration
        self._word_seg_config = {'enable': True, 'lang': 'en'}
        self._doc_filter_config = {'enable': True, 'min_len': 0, 'max_len': six.MAXSIZE}
        self._word_stem_config = {'enable': True}
        self._word_lower_config = {'enable': True}
        self._word_filter_config = {'enable': True, 'stop_words': nltk_stopwords.words('english'),
                                    'min_freq': 1, 'max_freq': six.MAXSIZE, 'words_useless': None}
        self._word_index_config = {'word_dict': None}

        self._word_seg_config.update(word_seg_config)
        self._doc_filter_config.update(doc_filter_config)
        self._word_stem_config.update(word_stem_config)
        self._word_lower_config.update(word_lower_config)
        self._word_filter_config.update(word_filter_config)
        self._word_index_config.update(word_index_config)

        self._word_dict = self._word_index_config['word_dict']
        self._words_stats = dict()

    def run(self, file_path):
        print('load...')
        dids, docs = Preprocess.load(file_path)

        if self._word_seg_config['enable']:
            print('word_seg...')
            docs = Preprocess.word_seg(docs, self._word_seg_config)

        if self._doc_filter_config['enable']:
            print('doc_filter...')
            dids, docs = Preprocess.doc_filter(dids, docs, self._doc_filter_config)

        if self._word_stem_config['enable']:
            print('word_stem...')
            docs = Preprocess.word_stem(docs)

        if self._word_lower_config['enable']:
            print('word_lower...')
            docs = Preprocess.word_lower(docs)

        self._words_stats = Preprocess.cal_words_stat(docs)

        if self._word_filter_config['enable']:
            print('word_filter...')
            docs, self._words_useless = Preprocess.word_filter(docs, self._word_filter_config, self._words_stats)

        print('word_index...')
        docs, self._word_dict = Preprocess.word_index(docs, self._word_index_config)

        return dids, docs

    @staticmethod
    def parse(line):
        subs = line.split(' ', 1)
        if 1 == len(subs):
            return subs[0], ''
        else:
            return subs[0], subs[1]

    @staticmethod
    def load(file_path):
        dids = list()
        docs = list()
        f = codecs.open(file_path, 'r', encoding='utf8')
        for line in tqdm(f):
            line = line.strip()
            if '' != line:
                did, doc = Preprocess.parse(line)
                dids.append(did)
                docs.append(doc)
        f.close()
        return dids, docs

    @staticmethod
    def word_seg_en(docs):
        docs = [word_tokenize(sent) for sent in tqdm(docs)]
        # show the progress of word segmentation with tqdm
        '''docs_seg = []
        print('docs size', len(docs))
        for i in tqdm(range(len(docs))):
            docs_seg.append(word_tokenize(docs[i]))'''
        return docs

    @staticmethod
    def word_seg_cn(docs):
        docs = [list(jieba.cut(sent)) for sent in docs]
        return docs

    @staticmethod
    def word_seg(docs, config):
        assert config['lang'].lower() in Preprocess._valid_lang, 'Wrong language type: %s' % config['lang']
        docs = getattr(Preprocess, '%s_%s' % (sys._getframe().f_code.co_name, config['lang']))(docs)
        return docs

    @staticmethod
    def cal_words_stat(docs):
        words_stats = {}
        docs_num = len(docs)
        for ws in docs:
            for w in ws:
                if w not in words_stats:
                    words_stats[w] = {}
                    words_stats[w]['cf'] = 0
                    words_stats[w]['df'] = 0
                    words_stats[w]['idf'] = 0
                words_stats[w]['cf'] += 1
            for w in set(ws):
                words_stats[w]['df'] += 1
        for w, winfo in words_stats.items():
            words_stats[w]['idf'] = np.log((1. + docs_num) / (1. + winfo['df']))
        return words_stats

    @staticmethod
    def word_filter(docs, config, words_stats):
        if config['words_useless'] is None:
            config['words_useless'] = set()
            # filter with stop_words
            config['words_useless'].update(config['stop_words'])
            # filter with min_freq and max_freq
            for w, winfo in words_stats.items():
                # filter too frequent words or rare words
                if config['min_freq'] > winfo['df'] or config['max_freq'] < winfo['df']:
                    config['words_useless'].add(w)
        # filter with useless words
        docs = [[w for w in ws if w not in config['words_useless']] for ws in tqdm(docs)]
        return docs, config['words_useless']

    @staticmethod
    def doc_filter(dids, docs, config):
        new_docs = list()
        new_dids = list()
        for i in tqdm(range(len(docs))):
            if config['min_len'] <= len(docs[i]) <= config['max_len']:
                new_docs.append(docs[i])
                new_dids.append(dids[i])
        return new_dids, new_docs

    @staticmethod
    def word_stem(docs):
        docs = [[Preprocess._stemmer.stem(w) for w in ws] for ws in tqdm(docs)]
        return docs

    @staticmethod
    def word_lower(docs):
        docs = [[w.lower() for w in ws] for ws in tqdm(docs)]
        return docs

    @staticmethod
    def build_word_dict(docs):
        word_dict = dict()
        for ws in docs:
            for w in ws:
                word_dict.setdefault(w, len(word_dict))
        return word_dict

    @staticmethod
    def word_index(docs, config):
        if config['word_dict'] is None:
            config['word_dict'] = Preprocess.build_word_dict(docs)
        docs = [[config['word_dict'][w] for w in ws if w in config['word_dict']] for ws in tqdm(docs)]
        return docs, config['word_dict']

    @staticmethod
    def save_lines(file_path, lines):
        f = codecs.open(file_path, 'w', encoding='utf8')
        for line in lines:
            line = line
            f.write(line + "\n")
        f.close()

    @staticmethod
    def load_lines(file_path):
        f = codecs.open(file_path, 'r', encoding='utf8')
        lines = f.readlines()
        f.close()
        return lines

    @staticmethod
    def save_dict(file_path, dic, sort=False):
        if sort:
            dic = sorted(dic.items(), key=lambda d: d[1], reverse=False)
            lines = ['%s %s' % (k, v) for k, v in dic]
        else:
            lines = ['%s %s' % (k, v) for k, v in dic.items()]
        Preprocess.save_lines(file_path, lines)

    @staticmethod
    def load_dict(file_path):
        lines = Preprocess.load_lines(file_path)
        dic = dict()
        for line in lines:
            k, v = line.split()
            dic[k] = v
        return dic

    def save_words_useless(self, words_useless_fp):
        Preprocess.save_lines(words_useless_fp, self._words_useless)

    def load_words_useless(self, words_useless_fp):
        self._words_useless = set(Preprocess.load_lines(words_useless_fp))

    def save_word_dict(self, word_dict_fp, sort=False):
        Preprocess.save_dict(word_dict_fp, self._word_dict, sort)

    def load_word_dict(self, word_dict_fp):
        self._word_dict = Preprocess.load_dict(word_dict_fp)

    def save_words_stats(self, words_stats_fp, sort=False):
        if sort:
            word_dic = sorted(self._word_dict.items(), key=lambda d: d[1], reverse=False)
            lines = ['%s %d %d %f' % (wid, self._words_stats[w]['cf'], self._words_stats[w]['df'],
                                      self._words_stats[w]['idf']) for w, wid in word_dic]
        else:
            lines = ['%s %d %d %f' % (wid, self._words_stats[w]['cf'], self._words_stats[w]['df'],
                                      self._words_stats[w]['idf']) for w, wid in self._word_dict.items()]
        Preprocess.save_lines(words_stats_fp, lines)

    def load_words_stats(self, words_stats_fp):
        lines = Preprocess.load_lines(words_stats_fp)
        for line in lines:
            wid, cf, df, idf = line.split()
            self._words_stats[wid] = {}
            self._words_stats[wid]['cf'] = int(cf)
            self._words_stats[wid]['df'] = int(df)
            self._words_stats[wid]['idf'] = float(idf)


def parse_line_for_quora(line, delimiter='","'):
    subs = line.split(delimiter)
    # print('subs: ', len(subs))
    # if subs[1]=="qid1":
    #     return
    if 6 != len(subs):
        # print( "line__not satisfied",line)
        # raise ValueError('format of data file wrong, should be \'label,text1,text2\'.')
        return 0, 0, 0, 0, 0
    else:
        return subs[1], subs[2], subs[3], subs[4], subs[5][0]


def run_with_one_corpus_for_quora(file_path):
    corpus = {}
    rels = []
    querys = []
    documents = []
    f = codecs.open(file_path, 'r', encoding='utf8')
    next(f)
    for line in f:
        # print("", i)
        # print("", i)
        # line = line.decode('utf8')
        line = line.strip()
        qid1, qid2, q1, q2, label = parse_line_for_quora(line, "\t")
        if q1 != 0:
            corpus[qid1] = q1
            corpus[qid2] = q2
            rels.append((label, qid1, qid2))
            querys.append((qid1, q1))
            documents.append((qid2, q2))
    f.close()
    return corpus, rels, documents, querys


def save_corpus(file_path, corpus):
    f = codecs.open(file_path, 'w', encoding='utf8')
    for qid, text in corpus.items():
        f.write('%s %s\n' % (qid, text))
    f.close()


def save_list(file_path, lst_name):
    f = codecs.open(file_path, 'w', encoding='utf8')
    for itm in lst_name:
        f.write('%s %s\n' % itm)
    f.close()


def merge_corpus(train_corpus, valid_corpus, test_corpus):
    # cat train valid test > corpus.txt
    # cat corpus_train.txt corpus_valid.txt corpus_test.txt > corpus.txt
    os.system('cat ' + train_corpus + ' ' + valid_corpus + ' ' + test_corpus + '  > corpus.txt')


def save_relation(file_path, relations):
    f = open(file_path, 'w')
    for rel in relations:
        f.write('%s %s %s\n' % rel)
    f.close()


def split_train_valid_test(relations, ratio=(0.8, 0.1, 0.1)):
    random.shuffle(relations)
    total_rel = len(relations)
    num_train = int(total_rel * ratio[0])
    num_valid = int(total_rel * ratio[1])
    valid_end = num_train + num_valid
    rel_train = relations[: num_train]
    rel_valid = relations[num_train: valid_end]
    rel_test = relations[valid_end:]
    return rel_train, rel_valid, rel_test


infile = 'quora_duplicate_questions.txt'
corpus, rels, documents, querys = run_with_one_corpus_for_quora(infile)

save_list('querys_ch.txt', querys)
save_list('documents_ch.txt', documents)

print('total corpus : %d ...' % (len(corpus)))
print('total relations : %d ...' % (len(rels)))
save_corpus('corpus.txt', corpus)
rel_train, rel_valid, rel_test = split_train_valid_test(rels, [0.8, 0.1, 0.1])
save_relation('relation_train.txt', rel_train)
save_relation('relation_valid.txt', rel_valid)
save_relation('relation_test.txt', rel_test)
print('Preparation finished ...')

# preprocessor = Preprocess(word_stem_config={'enable': False}, word_filter_config={'min_freq': 5})#处理英文
preprocessor = Preprocess(word_seg_config={'lang': 'cn'}, word_stem_config={'enable': False},
                          word_filter_config={'min_freq': 5,
                                              'stop_words': nltk_stopwords.words('chinese_stopwords')})
dids, docs = preprocessor.run('corpus.txt')

lines = ['%s:%s' % (k, v) for k, v in preprocessor._word_dict.items()]
preprocessor.save_lines('word_dict.txt', lines)

fout = open('corpus_preprocessed.txt', 'w')
for inum, did in enumerate(dids):
    fout.write('%s\t%s\n' % (did, ' '.join(map(str, docs[inum]))))
fout.close()

fout = open('corpus_for_w2v.txt', 'w')
for inum, did in enumerate(dids):
    fout.write('%s\n' % (' '.join(map(str, docs[inum]))))
fout.close()
print('preprocess finished ...')
