import jieba
import json


def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


with open('../data/chinese_stopwords', 'r', encoding='utf-8') as f:
    chinese_stopwords = f.read().split()

# ##添加自动一词典
jieba.load_userdict('../data/special_words_cut.txt')

with open('../data/corpus_processed_90000.txt', 'w', encoding='utf-8') as f:
    with open('../data/docs_gongwen.txt', 'r', encoding='utf-8') as fp:
        i = 0
        j = 0
        while True:
            line = fp.readline()
            i += 1
            print('sentences%d' % i)
            if not line: break
            file_dict = json.loads(line)
            file_body = file_dict['body']
            line_cut = list(jieba.cut(file_body))
            line_d_stopwords = [w for w in line_cut if w not in chinese_stopwords]
            line_chinese = [w for w in line_d_stopwords if is_chinese(w)]
            print(line_chinese)
            for word in line_chinese:
                j += 1
                f.write(word)
                f.write(' ')
            f.write('\n')
            print('words:%d' % j)

# words:84290757
# sentences94334
