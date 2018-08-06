import jieba
import json



def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


with open('../data/docs_gongwen.txt', 'r', encoding='utf-8') as fp:
    lines = fp.readlines()


with open('../data/chinese_stopwords', 'r', encoding='utf-8') as f:
    chinese_stopwords = f.read().split()
with open('../data/corpus_processed_90000.txt', 'w', encoding='utf-8') as f:
    i = 0
    for line in lines:
        i += 1
        print(i)
        file_dict = json.loads(line)
        file_body = file_dict['body']
        line_cut = list(jieba.cut(file_body))
        line_d_stopwords = [w for w in line_cut if w not in chinese_stopwords]
        line_chinese = [w for w in line_d_stopwords if is_chinese(w)]
        print(line_chinese)
        for word in line_chinese:
            f.write(word)
            f.write(' ')
        f.write('\n')

