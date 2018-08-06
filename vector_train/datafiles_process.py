import jieba
import os
import json


def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


def data_preprocess(srcfile, stopwords_file, dstfile):
    with open(srcfile, encoding='utf-8') as f:
        lines = f.read()
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        chinese_stopwords = f.read().split()

    with open(dstfile, 'w', encoding='utf-8') as f:
        i = 0
        for line in lines.split('###'):
            line_cut = list(jieba.cut(line))
            line_d_stopwords = [w for w in line_cut if w not in chinese_stopwords]
            line_chinese = [w for w in line_d_stopwords if is_chinese(w)]
            i += 1
            print(i)
            print(line_chinese)
            for word in line_chinese:
                f.write(word)
                f.write(' ')
            f.write('\n')


srcfile = '../data/corpus_gongwen_80000.txt'
stopwords_file = '../data/chinese_stopwords'
dstfile = '../data/corpus_processed_80000.txt'
data_preprocess(srcfile,stopwords_file,dstfile)