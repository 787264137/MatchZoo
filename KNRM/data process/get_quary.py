import jieba 
def stopwordslist(filepath):  
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]  
    return stopwords  

def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False

#f=open("quora_duplicate_questions.txt",encoding='utf-8')
f=open("new_quora1.txt",encoding='utf-8')
data=f.readlines()
n=len(data)
temp='1'
q_list=['乐活周末#人们把陕西叫做三秦，三秦指的是什么？']
for i in range(1,n):
	info=data[i].split("\t")
	#print(len(info))
	if info[1]==temp :
		continue
	else :
		q_list.append(info[3])
		temp=info[1]
print(len(q_list))
f2=''.join(q_list)

dic_all=[]
f1=open("word_dict.txt",encoding='utf-8')
dic_f=f1.readlines()
print(len(dic_f))
for dic in dic_f:
	dic_1=dic.split(":")[0]
	if dic_1 in f2:
		#print("1")
		dic_all.append(dic)

#print(len(dic_all))
print(len(set(dic_all)))


'''dic=jieba.cut(f)
dic_all=''
for j in dic:
	dic_all+=','+j'''

'''def seg_sentence(sentence):  
    sentence_seged = jieba.cut(sentence.strip())  
    stopwords = stopwordslist('./stopwords.txt')  # 这里加载停用词的路径  
    outstr = [] 
    for word in sentence_seged:
        if is_chinese(word):  
            if word not in stopwords:  
                if word != '\t':  
                    outstr.append(word) 
    print(len(set(outstr))) 
    return outstr


dic_all= '\t'.join(set(seg_sentence(f)))'''
dic_all=''.join(dic_all)
#print(len(dic_all))
f1=open('result_quary_dic1.txt','w',encoding='utf-8')
f1.write(dic_all)

