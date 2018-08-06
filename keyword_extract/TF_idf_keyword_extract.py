from jieba.analyse import *
import jieba
import datetime
from jieba.analyse import *
jieba.load_userdict("./政治.txt") 
def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False
def stopwordslist(filepath):  
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]  
    return stopwords  
f=open("1.txt",encoding='utf-8')
data=f.readlines()
n=len(data)
f2=open("2.txt",'w',encoding='utf-8')
for index,i in enumerate(data):

	#print(index)
	#print(i)
	a=str(index)+'\t'
	info=i.strip()
	#f2=open("2.txt",'w',encoding='utf-8')
	for keyword,weight in textrank(i,topK=15,withWeight=True):
		a+=keyword+'\t'
	f2.write(a+'\n')
	print(n)
	n=n-1
f2.close()


