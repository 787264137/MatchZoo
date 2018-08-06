
import codecs
from gensim.models import Word2Vec
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import jieba.posseg as psg
from sklearn import cross_validation
import collections
import math
import os
jieba.load_userdict("./自定义词典1.txt")     
f=open('./word_dic.txt','w',encoding='utf-8')

def process(filename):
    data=open(file_name,encoding='utf-8'):
    lines=data.readlines()
    return lines

def iter_files(rootDir):
    #遍历根目录
    for root,dirs,files in os.walk(rootDir):
        for file in files:
            file_name = os.path.join(root,file)
            data=process(file_name)
            wordslit=cutWords(data)
            f.write(wordslit+'\n')

        for dirname in dirs:
            #递归调用自身,只改变目录名称
            iter_files(dirname)

#iter_files("./data")

def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False
def cutWords(eachTxt):
    stopList = []
    for stopWord in codecs.open('F:/getKeyWords/stopwords.txt', 'r', 'utf-8'):
        stopList.append(stopWord.strip())
    words = psg.cut(eachTxt)
    wordsList = []
    for w in words:
        flag = True
        for i in range(len(w.word)):
            if not is_chinese(w.word[i]):
                flag = False
                break
        if flag and len(w.word) > 1:
            wordsList.append(w.word)
    return wordsList

if __name__=='__main__':
    iter_files("./data")
    f.close()
