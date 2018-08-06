import sys

if sys.version_info.major < 3:
    reload(sys)  # Reload does the trick!
    sys.setdefaultencoding('utf-8')
    import codecs
    open = codecs.open

import os, re
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
os.environ['CLASSPATH'] = os.path.join(BASE_DIR, "AiCmnTika.jar")

from jnius import autoclass
import html2text

Tika = autoclass('org.apache.tika.Tika')
Metadata = autoclass('org.apache.tika.metadata.Metadata')
ByteArrayInputStream = autoclass('java.io.ByteArrayInputStream')

def parse_from_buffer(file_buffer):
    try:
        tika = Tika()
        metadata = Metadata()
        bais = ByteArrayInputStream(file_buffer)
        text = tika.parseToString(bais, metadata)
        if sys.version_info.major < 3:
            text = text.decode('utf-8')
        if len(text) >= 5 and text[0:5] == '<html':
            text = html2text.html2text(text)
        text = re.sub(r'^\s|\s+(?=\s)|\s$', r'', text)
        return text.strip()
    except:
        return None
    finally:
        try:
            bais.close()
        except:
            pass

def parse_from_file(file):
    with open(file, 'rb') as rf:
        file_buffer = rf.read()
    return parse_from_buffer(file_buffer)

if __name__ == '__main__':
    # print(parse_from_file(r'C:\Users\soso\Documents\WeChat Files\sysu_sicily\Files\【2018校招】+【黄国俊】+【18826136267】+【来自职场校长】.pdf'))
    # print(parse_from_file(r'F:\数据集\简历数据集\计算机行业简历分类数据集\Android开发\Android软件开发工程师-陈晓东-华南师范大学-软件工程.pdf'))
    # with open('test.txt', 'w') as wf:
    #     wf.write(parse_from_file('/home/zli/Downloads/CV_folder/CV_test_1.txt'))
    print(parse_from_file(sys.argv[1]))