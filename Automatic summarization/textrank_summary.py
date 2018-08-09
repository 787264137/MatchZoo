from snownlp import seg
from snownlp import bm25
import re


def get_sentences(doc):
    line_break = re.compile('[\r\n]')
    delimiter = re.compile('[。？！]')
    sentences = []
    for line in line_break.split(doc):
        line = line.strip()
        if not line:
            continue
        for sent in delimiter.split(line):
            sent = sent.strip()
            if not sent:
                continue
            sentences.append(sent)
    return sentences


def filter_stop(words, stopwords):
    return list(filter(lambda x: x not in stopwords, words))


def get_ch_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        chinese_stopwords = f.read().split()
    return chinese_stopwords


def get_doc_for_rank(sents):
    doc = []
    for sent in sents:
        words = seg.seg(sent)
        words = filter_stop(words, stopwords)
        doc.append(words)
    return doc


class TextRank(object):
    def __init__(self, docs):
        self.docs = docs
        self.bm25 = bm25.BM25(docs)
        self.D = len(docs)
        self.d = 0.85
        self.weight = []
        self.weight_sum = []
        self.vertex = []
        self.max_iter = 200
        self.min_diff = 0.001
        self.top = []
        # self.weight_sum每一句与其它句相似度和
        # self.weight每一句与其它句的相似度

    def text_rank(self):
        for cnt, doc in enumerate(self.docs):
            scores = self.bm25.simall(doc)
            self.weight.append(scores)
            self.weight_sum.append(sum(scores) - scores[cnt])
            self.vertex.append(1.0)
        for _ in range(self.max_iter):
            m = []
            max_diff = 0
            for i in range(self.D):
                m.append(1 - self.d)
                for j in range(self.D):
                    if j == i or self.weight_sum[j] == 0:
                        continue
                    # TextRank的公式
                    m[-1] += (self.d * self.weight[j][i]
                              / self.weight_sum[j] * self.vertex[j])
                if abs(m[-1] - self.vertex[i]) > max_diff:
                    max_diff = abs(m[-1] - self.vertex[i])
            self.vertex = m
            if max_diff <= self.min_diff:
                break
        self.top = list(enumerate(self.vertex))
        self.top = sorted(self.top, key=lambda x: x[1], reverse=True)

    def top_index(self, limit):
        return list(map(lambda x: x[0], self.top))[:limit]

    def top(self, limit):
        return list(map(lambda x: self.docs[x[0]], self.top))


if __name__ == '__main__':
    with open("data/news3.txt", "r", encoding='utf-8') as myFile:
        text = myFile.read().replace('\n', '')
    sents = get_sentences(text)
    stopwords = get_ch_stopwords('data/chinese_stopwords')

    doc = get_doc_for_rank(sents)
    rank = TextRank(doc)
    rank.text_rank()
    index = rank.top_index(2)
    print('Extracted key sentences:\n')
    with open("data/textrank_summary3.txt", "w", encoding='utf-8') as f:
        for i in range(len(index)):
            print(sents[index[i]])
            f.write(sents[index[i]] + '\n')
        f.close()
