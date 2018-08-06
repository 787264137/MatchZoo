import os
a=open('./自定义词典.txt',encoding='utf-8')
data=a.readlines()
with open('./自定义词典1.txt','w',encoding='utf-8'):
	for line in data:
		info =line.strip().split()
		if len(info)==2:
			f.write(info[1]+'\n')
