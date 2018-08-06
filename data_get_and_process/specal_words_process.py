import jieba
import re


def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


def has_digit(uchar):
    p = r"\d+"
    m = re.findall(p, uchar)
    if not m:
        return False
    else:
        return True


def has_alpha(uchar):
    p = r'[A-Za-z]'
    m = re.findall(p, uchar)
    if not m:
        return False
    else:
        return True


def has_plus(uchar):
    p = r'\+'
    m = re.findall(p, uchar)
    if not m:
        return False
    else:
        return True


with open('../data/chinese_stopwords', 'r', encoding='utf-8') as f:
    chinese_stopwords = f.read().split()

jieba.load_userdict('../data/special_words.txt')

with open('../data/special_words.txt', 'r', encoding='utf-8') as f:
    with open('../data/special_words_cut.txt', 'w', encoding='utf-8') as fw:
        while True:
            line = f.readline().strip()
            if not line: break
            line_words = list(jieba.cut(line))
            line_words = [w for w in line_words if w not in chinese_stopwords]  # 过滤掉停用词
            line_words = [w for w in line_words if is_chinese(w)]
            line_words = [w for w in line_words if not has_digit(w)]
            line_words = [w for w in line_words if not has_alpha(w)]
            line_words = [w for w in line_words if not has_plus(w)]
            print(line_words)
            if line_words:
                fw.write('\n'.join(line_words))
                fw.write('\n')
