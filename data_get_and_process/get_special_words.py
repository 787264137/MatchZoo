import re


def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False

with open('../data/gongwen_body.txt', 'r', encoding='utf-8') as f:
    lines = f.read().split('###')
special_words = []
for line in lines:
    pattern1 = re.compile(r'《(.{,20}?)》')
    matchObj1 = pattern1.findall(line)
    print(matchObj1)
    special_words.extend(matchObj1)
    pattern2 = re.compile(r'“(.{,8}?)”')
    matchObj2 = pattern2.findall(line)
    print(matchObj2)
    special_words.extend(matchObj2)

words_dict = {}
for word in special_words:
    if word not in words_dict.keys():
        words_dict[word] = 1
    else:
        words_dict[word] += 1

with open('../data/special_words.txt', 'w', encoding='utf-8') as f:
    for word, frequence in words_dict.items():
        # f.write('%s %d' % (word, frequence))
        f.write('%s' % (word))
        f.write('\n')
