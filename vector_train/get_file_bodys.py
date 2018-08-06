import jieba
import json


with open('../data/gongwen_body.txt', 'w', encoding='utf-8') as f:
    with open('../data/docs_gongwen.txt', 'r', encoding='utf-8') as fp:
        i = 0
        while True:
            line = fp.readline().strip()
            i += 1
            print('sentences%d' % i)
            if not line: break
            file_dict = json.loads(line)
            file_body = file_dict['body'].strip()
            f.write('###')
            f.write(file_body)
            f.write('\n')


# words:84290757
# sentences94334
