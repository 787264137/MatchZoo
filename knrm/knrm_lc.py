from keras.initializers import RandomUniform
from keras.layers import *
from keras.models import Model


class KNRM(object):
    def __init__(self, config):
        self._name = 'KNRM'
        self.config = config

    def build(self):
        # 定义RBF核函数
        def Kernel_layer(mu, sigma):
            def kernel(x):
                return K.tf.exp(-0.5 * (x - mu) * (x - mu) / sigma / sigma)
            return Activation(kernel)

        # 输入
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))

        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']],
                              trainable=self.config['train_embed'])
        q_embed = embedding(query)
        d_embed = embedding(doc)

        # 将q和d的向量分别点乘
        mm = Dot(axes=[2, 2], normalize=True)([q_embed, d_embed])  # axes = 0,矩阵。1,行。2,单个 元素
        KM = []
        for i in range(self.config['kernel_num']):
            mu = 1. / (self.config['kernel_num'] - 1) + (2. * i) / (self.config['kernel_num'] - 1) - 1.0
            sigma = self.config['sigma']
            if mu > 1.0:
                sigma = self.config['exact_sigma']
                mu = 1.0
            # 将相似度矩阵mm中的每一个值经过高斯核的激活函数，映射成另一个矩阵mm_exp
            mm_exp = Kernel_layer(mu, sigma)(mm)
            # 对于矩阵mm_exp中的每一个，q
            mm_doc_sum = Lambda(lambda x: K.tf.reduce_sum(x, 2))(mm_exp)

            mm_log = Activation(K.tf.log1p)(mm_doc_sum)

            mm_sum = Lambda(lambda x: K.tf.reduce_sum(x, 1))(mm_log)
            KM.append(mm_sum)

        Phi = Lambda(lambda x: K.tf.stack(x, 1))(KM)

        model = Model(inputs=[query, doc], outputs=[Phi])
        return model