import jieba


def check(line):
    subs = line.split(' ')
    if len(subs) == 2:
        return subs[0], subs[1]
    else:
        return 0, 0


def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


def get_query(query_path):
    querys_index = []
    querys = []
    for line in open(query_path, 'r', encoding='utf-8').readlines():
        index, query = check(line)
        if int(index) not in querys_index:
            querys_index.append(index)
            querys.append(query)
    return querys_index, querys


def getwords_query(querys, stopwords):
    words = []
    jieba.load_userdict('word_dict_userdict.txt')
    for q in querys:
        line_cut = list(jieba.cut(q))
        line_d_stopwords = [w for w in line_cut if w not in stopwords]
        line_chinese = [w for w in line_d_stopwords if is_chinese(w)]
        for word in line_chinese:
            if word not in words:
                words.append(word)
    return words


def getdocs_corpus_digit(query_indexes, corpus_digit_path):
    with open(corpus_digit_path, 'r', encoding='utf-8') as f:
        docs_digit = []
        did = []
        i = 0
        while True:
            line = f.readline().strip()
            if not line: break
            subs = line.split('\t', maxsplit=1)
            if len(subs) != 2:
                i += 1
                print(line)
                continue
            cid, content = subs[0], subs[1]
            if cid not in query_indexes:
                did.append(cid)
                docs_digit.append(content)
        print("%d rows have errors" % i)
    return did, docs_digit


def get_words_dict(word_dict_path):
    with open(word_dict_path, 'r', encoding='utf-8') as f:
        words_dict = {}
        while True:
            line = f.readline().strip()
            if not line: break
            word, word_digit = line.split(':')
            words_dict[word] = word_digit
    return words_dict


def get_stopwords(path):
    with open(path, 'r', encoding='utf-8') as f:
        chinese_stopwords = f.read().split()
    return chinese_stopwords


def convert_to_digit(words, words_dict):
    words_others = []
    words_digit = []
    words_chn = []
    for word in words:
        if word in words_dict.keys():
            words_digit.append(words_dict[word])
            words_chn.append(word)
        else:
            words_others.append(word)
    return words_chn, words_digit, words_others


# 得到query中的words的digit表示
chinese_stopwords = get_stopwords('chinese_stopwords')
query_indexes, querys = get_query('querys_ch.txt')
words = getwords_query(querys, chinese_stopwords)
words_dict = get_words_dict('word_dict.txt')
words_chn, words_digit, words_others = convert_to_digit(words, words_dict)

print(len(words), len(words_digit), len(words_chn), len(words_others))

# 得到doc的digit表示
corpus_digit_path = 'corpus_preprocessed.txt'
did, docs_digit = getdocs_corpus_digit(query_indexes, corpus_digit_path)
print(len(did), len(docs_digit))
# 99421 99421
print(docs_digit[0])
for i in range(len(words_digit)):
    with open('D://words_docs1//%s-%s.txt' % (words_chn[i], words_digit[i]), 'w', encoding='utf-8') as f:
        f.write('%s\t%s\n' % (words_chn[i], words_digit[i]))
        for j in range(len(did)):
            print('words:%d,docs:%d' % (i, j))
            f.write('%s\t%s\n' % (did[j], docs_digit[j]))
