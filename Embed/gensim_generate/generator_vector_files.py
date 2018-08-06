import gensim
import os


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname), encoding='utf-8'):
                sentence = line.rstrip().split(' ')
                print(sentence)
                yield sentence


sentences = MySentences('../data/files')  # a memory-friendly iterator
model = gensim.models.Word2Vec(sentences, size=300, workers=5)
model.wv.save_word2vec_format('../data/embed_d300_3')  # 保存的是utf-8

# model = gensim.models.KeyedVectors.load_word2vec_format('embed_d300_20000_iter')

# model.most_similar(positive=['中国', '中心'], negative=['危险'], topn=1)
# model.doesnt_match("中国 美国 加拿大 内容".split())
# model.similarity('国家', '发生')
# model.most_similar(['内容', '标题'])
# print(model.most_similar('项目'))
# print(model.most_similar('国家'))
# print(model.most_similar('中心'))
# print(model.most_similar('危险品'))
# print(model.most_similar('泄露'))
# print(model.most_similar('发生'))
# print(model.most_similar('二级'))
# print(model.most_similar('内容'))
# vec_zf = model['中国']
# len(vec_zf)
