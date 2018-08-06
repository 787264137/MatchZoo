import os
import json


def files_to_txt(dirname, txtname):
    with open(txtname, 'w', encoding='utf-8') as f:
        i = 0
        for fname in os.listdir(dirname):
            file_data = open(os.path.join(dirname, fname), 'r', encoding='utf-8').read()
            file_dict = json.loads(file_data)
            file_body = file_dict['snapshot_content']
            i += 1
            print('=========================')
            print(file_dict['snapshot_content'])
            print(i)
            f.write('###')
            f.write(file_body)


dirname = '../data/cnki'
txtname = '../data/corpus_gongwen_80000.txt'
files_to_txt(dirname, txtname)
