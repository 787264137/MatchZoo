import jieba
from heapq import nlargest
from collections import defaultdict
import re
from jieba.analyse import textrank


def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


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


def get_ch_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        chinese_stopwords = f.read().split()
    return chinese_stopwords


def summarize(text, n):

    freq = dict(textrank(text, topK=15, withWeight=True))
    print(freq)

    sents = get_sentences(text)
    assert n <= len(sents)

    word_sent = [jieba.lcut(s) for s in sents]
    ranking = defaultdict(int)
    for i, word in enumerate(word_sent):
        for w in word:
            if w in freq:
                ranking[i] += freq[w]
    sents_idx = rank(ranking, n)
    return [sents[j] for j in sents_idx]


def rank(ranking, n):
    return nlargest(n, ranking, key=ranking.get)


if __name__ == '__main__':
    with open("data/news3.txt", "r", encoding='utf-8') as myFile:
        text = myFile.read().replace('\n', '')
    stopwords = get_ch_stopwords('data/chinese_stopwords')
    res = summarize(text, 2)
    f = open("data/keyword2_summary3.txt", "w", encoding='utf-8')
    print('Extracted key sentences:\n')
    for i in range(len(res)):
        print(res[i])
        f.write(res[i] + '\n')
    f.close()
