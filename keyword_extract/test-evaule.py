from collections import Counter
import numpy as np
import os
import json
import gensim
import pandas as pd 
import jieba
import datetime
from jieba.analyse import *
jieba.load_userdict("./自定义词典1.txt")     
starttime=datetime.datetime.now()
#model = gensim.models.word2vec.Word2Vec.load('./model_file/word2vec.model')
model = gensim.models.word2vec.Word2Vec.load('./model_file/word2vec_wx')
modell = gensim.models.Word2Vec.load('./model_file/word2vec_wx')
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


def TFidf(data):
    b=''
    for keyword, weight in extract_tags(data, topK=3, withWeight=True):
        b+=keyword+'\t'
    return b
        

'''def read_file(path):
    print("1")
    file_name=os.listdir(path)
    m=len(file_name)
    print(m)
    for file in file_name:
        a=''
        file_path=path+'/'+file
        f=open(file_path,'r+',encoding='utf-8')
        data=f.readlines()
        data=','.join(data)
        result=keywords(seg_sentence(data))[0:5]
        for i in range(4):
            a+=result[i][0]+'\t'
        result_tfidf=TFidf(data)
        f.write('\n'+'word2vec+softmax:'+'\n'+a+'\n'+'TF-IDF:'+'\n'+result_tfidf+'\n')
        f.close()
        m-=1
        print("还剩下:",m)'''
def accuracy(result_word2vec,result_tfidf,y_keyword):
    #word_count=0
    #tfidf_cont=0
    wordlist=[]
    for i in range(3):
        wordlist.append(result_word2vec[i][0])
    result_tfidf=result_tfidf.split("\t")
    y_keyword=y_keyword.split(",")

    word_count= len([l for l in wordlist if l in y_keyword])
    tfidf_cont= len([l for l in result_tfidf if l in y_keyword])
# [4,5]
    '''for i in y_keyword:
        for j in result_word2vec:
            if i==j:
                word_count+=1
                break'''
            #elif modell.n_similarity(i,j)>=float(0.7):
            #    word_count+=1
            #    break
    print("word2vec:",wordlist)
    print(type(wordlist))
    print("TFidf:",result_tfidf)
    print(type(result_tfidf))
    print('y_keyword:',y_keyword)
    print(type(y_keyword))
    print("word2vec预测对的个数",word_count)
    '''for i in y_keyword:
        for j in result_tfidf:
            if i==j:
                tfidf_cont+=1
                break'''
            #elif modell.n_similarity(i,j)>=float(0.7):
            #    tfidf_cont+=1
            #    break
    print("tf预测对的个数",tfidf_cont)
    return word_count,tfidf_cont


def read_file(path):
    accuracy_word2=0
    accuracy_tfidf=0
    file_name=os.listdir(path)
    m=len(file_name)
    for file in file_name:
        file_path=path+'/'+file
        f=open(file_path,'r+',encoding='utf-8')
        data=json.load(f)
        result_word2vec=keywords(seg_sentence(data["body"]))[0:3]
        result_tfidf=TFidf(data["body"])
        y_keyword=data['keywords']
        a,b=accuracy(result_word2vec,result_tfidf,y_keyword)
        accuracy_word2+=a
        accuracy_tfidf+=b
        m-=1
        print("还剩下:",m)
    print("word2vec accuracy:",float(accuracy_word2) / float(900))
    print("word2vec accuracy:",float(accuracy_tfidf) / float(900))

#print(pd.Series(keywords(jieba.cut(s,cut_all=False))))
#data=','.join(data)
read_file("./test_data2/news")