import gensim


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        i = 0
        for line in open(self.dirname, 'r', encoding='utf-8'):
            i += 1
            print(i)
            sentence = line.rstrip().split(' ')
            print(sentence)
            yield sentence


sentences = MySentences('../data/corpus_processed_80000.txt')  # a memory-friendly iterator
model = gensim.models.Word2Vec(sentences, size=300, workers=5)
model.wv.save_word2vec_format('../data/embed_d300_80000')  # 保存的是utf-8

# model = gensim.models.KeyedVectors.load_word2vec_format('../data/embed_d300_20000_iter')

print(model.most_similar(positive=['中国', '中心'], negative=['危险'], topn=5))
print(model.doesnt_match("中国 美国 加拿大 内容".split()))
print(model.similarity('国家', '发生'))
print(model.most_similar(['内容', '标题']))
print(model.most_similar('项目'))
print(model.most_similar('安排'))
print(model.most_similar('深入'))
print(model.most_similar('粤'))
print(model.most_similar('国家'))
print(model.most_similar('中心'))
print(model.most_similar('危险品'))
print(model.most_similar('泄露'))
print(model.most_similar('发生'))
print(model.most_similar('二级'))
print(model.most_similar('内容'))
print(model.most_similar('说'))
print(model.most_similar('看'))
vec_zf = model['中国']
print(len(vec_zf))

# class MySentences(object):
#     def __init__(self, dirname):
#         self.dirname = dirname
#
#     def __iter__(self):
#         for fname in os.listdir(self.dirname):
#             for line in open(os.path.join(self.dirname, fname)):
#                 yield line.split()
