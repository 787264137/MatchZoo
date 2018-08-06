from collections import Counter
import numpy as np
import gensim
import pandas as pd 
import jieba
import datetime
from jieba.analyse import *
jieba.load_userdict("./自定义词典1.txt")     
starttime=datetime.datetime.now()
#model = gensim.models.word2vec.Word2Vec.load('./model_file/word2vec.model')
model = gensim.models.word2vec.Word2Vec.load('./model_file/word2vec_wx')
#model = gensim.models.word2vec.Word2Vec.load('./model_file/wiki.zh.text.model')
#stoplist = {}.fromkeys([ line.strip() for line in open("./stopwords",encoding='utf-8') ])
#stopwords = {}.fromkeys(['我们','指出'])
def stopwordslist(filepath):  
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]  
    return stopwords  

def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False
 
def seg_sentence(sentence):  
    sentence_seged = jieba.cut(sentence.strip())  
    stopwords = stopwordslist('./stopwords.txt')  # 这里加载停用词的路径  
    outstr = [] 
    for word in sentence_seged:
        if is_chinese(word):  
            if word not in stopwords:  
                if word != '\t':  
                    outstr.append(word)  
    return outstr  

def predict_proba(oword, iword):
    iword_vec = model[iword]
    oword = model.wv.vocab[oword]
    oword_l = model.syn1[oword.point].T
    dot = np.dot(iword_vec, oword_l)
    lprob = -sum(np.logaddexp(0, -dot) + oword.code*dot) 
    return lprob

def keywords(s):
    s = [w for w in s if w in model]
    ws = {w:sum([predict_proba(u, w) for u in s]) for w in s}
    return Counter(ws).most_common()

data=open('./test_data2/1.txt')
a=data.readlines()
n=len(a)
s=''
for i in range(n):
    s+=a[i]

#print(pd.Series(keywords(jieba.cut(s,cut_all=False))))
print(pd.Series(keywords(seg_sentence(s)))[0:9])
#print(type(keywords(seg_sentence(s)))[0:7])
#data=keywords(seg_sentence(s))[0:7]
endtime=datetime.datetime.now()
print('time:',(endtime-starttime).seconds)

print("TF-IDF:")
data=seg_sentence(s)
data=','.join(data)
for keyword, weight in extract_tags(data, topK=7, withWeight=True):
    print('%s %s' % (keyword, weight))