import jieba
from heapq import nlargest
from collections import defaultdict
import re


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


def compute_frequencies(word_sent):
    max_cut = 0.9
    min_cut = 0.1
    freq = defaultdict(int)
    for s in word_sent:
        for word in s:
            if word not in stopwords and is_chinese(word):
                freq[word] += 1
    m = float(max(freq.values()))
    for w in list(freq.keys()):
        freq[w] /= m
        if freq[w] >= max_cut or freq[w] <= min_cut:
            del freq[w]
    # {key:单词，value：重要性}
    return freq


def summarize(text, n):
    sents = get_sentences(text)
    assert n <= len(sents)

    word_sent = [jieba.lcut(s) for s in sents]
    freq = compute_frequencies(word_sent)
    print(freq)
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
    with open("data/news1.txt", "r", encoding='utf-8') as myFile:
        text = myFile.read().replace('\n', '')
    stopwords = get_ch_stopwords('data/chinese_stopwords')
    res = summarize(text, 2)
    f = open("data/keyword_summary1.txt", "w", encoding='utf-8')
    print('Extracted key sentences:\n')
    for i in range(len(res)):
        print(res[i])
        f.write(res[i] + '\n')
    f.close()
