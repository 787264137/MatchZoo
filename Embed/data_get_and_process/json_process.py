import jieba
import json
import sys

print(sys.maxunicode)


def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


with open('data/docs_gongwen', 'r', encoding='utf-8') as fp:
    fp.__next__()  # 第一行数据有问题
    lines = fp.readlines()
    i = 0
    for line in lines:
        i += 1
        print(i)
        file_dict = json.loads(line)
        file_body = file_dict['snapshot_content']
        print(file_dict)




with open('data/corpus_gongwen_20000.txt', encoding='utf-8') as f:
    lines = f.read()
with open('data/chinese_stopwords', 'r', encoding='utf-8') as f:
    chinese_stopwords = f.read().split()

sentences = []
j = 0
for line in lines.split('###'):
    print(j)
    j += 1
    line_cut = list(jieba.cut(line))
    line_d_stopwords = [w for w in line_cut if w not in chinese_stopwords]
    line_chinese = [w for w in line_d_stopwords if is_chinese(w)]
    sentences.append(line_chinese)

with open('data/corpus_processed_20000.txt', 'w', encoding='utf-8') as f:
    i = 0
    for sentence in sentences:
        print(i)
        i += 1
        for word in sentence:
            f.write(word)
            f.write(' ')
        f.write('\n')
