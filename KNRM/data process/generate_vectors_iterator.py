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


sentences = MySentences('corpus_for_w2v.txt')
model = gensim.models.Word2Vec(sentences, size=300, workers=5)
model.wv.save_word2vec_format('embed_d300_100000')

model1 = gensim.models.KeyedVectors.load_word2vec_format('KNMR/embed_d300_100000')
model1.most_similar('19')