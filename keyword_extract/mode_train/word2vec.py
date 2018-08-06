import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#import pymongo
import hashlib

#db = pymongo.MongoClient('172.16.0.101').weixin.text_articles_words
md5 = lambda s: hashlib.md5(s).hexdigest()
class sentences:
    def __iter__(self):
        texts_set = set()
        for a in db.find(no_cursor_timeout=True):
            if md5(a['text'].encode('utf-8')) in texts_set:
                continue
            else:
                texts_set.add(md5(a['text'].encode('utf-8')))
                yield a['words']
        print u'最终计算了%s篇文章'%len(texts_set)

word2vec = gensim.models.word2vec.Word2Vec(sentences(), size=256, window=10, min_count=64, sg=1, hs=1, iter=10, workers=25)
word2vec.save('word2vec_wx')