
import codecs
from gensim.models import Word2Vec
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import jieba.posseg as psg
from sklearn import cross_validation
import collections
import math
import os
import jieba
jieba.load_userdict("./自定义词典1.txt")     
f=open('./word_dic.txt','w',encoding='utf-8')

def process(filename):
    data=open(filename,'r',encoding='utf-8')
    lines=data.readlines()
    return lines[0]

'''def iter_files(rootDir):
    #遍历根目录
    word=''
    i=1
    for root,dirs,files in os.walk(rootDir):
        for file in files:
            file_name = os.path.join(root,file)
            data=process(file_name)
            wordslit=cutWords(data)
            for temp in wordslit:
                word=temp+' '
            f.write(word+'\n')
            print("已经写入：%d 条数据"，i)
            i+=1
        for dirname in dirs:
            #递归调用自身,只改变目录名称
            iter_files(dirname)'''
def iter_files(rootDir):
    #遍历根目录
    namelist=[]
    for root,dirs,files in os.walk(rootDir):
        for file in files:
            file_name = os.path.join(root,file)
            namelist.append(file_name)
        for dirname in dirs:
            #递归调用自身,只改变目录名称
            iter_files(dirname)
    return namelist

#iter_files("./data")

def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False
def cutWords(eachTxt):
    stopList = []
    for stopWord in codecs.open('./stopwords', 'r', 'utf-8'):
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
    info=''
    i=1
    total_list=[]
    file_name_list=iter_files("./data")
    file_name_list=set(file_name_list)
    print('文件总数：%d',len(file_name_list))
    for filename in file_name_list:
        data=process(filename)
        wordslit= cutWords(data)
        for temp in wordslit:
            info=info+temp+' '
        print("已经处理完:%d个文件 ",i)
        i+=1
    f.write(info)
    f.close()

   
