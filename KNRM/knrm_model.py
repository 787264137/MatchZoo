# -*- coding=utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import sys
import traceback

import psutil
from keras.initializers import RandomUniform
from keras.layers import *
from keras.models import Model


class Utility(object):
    def show_layer_info(self, layer_name, layer_out):
        print('[layer]: %s\t[shape]: %s \n%s' % (layer_name, str(layer_out.get_shape().as_list()), self.show_memory_use()))

    def show_memory_use(self):
        used_memory_percent = psutil.virtual_memory().percent
        strinfo = '{}% memory has been used'.format(used_memory_percent)
        return strinfo

    def import_class(self, import_str):
        mod_str, _sep, class_str = import_str.rpartition('.')
        __import__(mod_str)
        try:
            return getattr(sys.modules[mod_str], class_str)
        except AttributeError:
            raise ImportError('Class %s cannot be found (%s)' %
                              (class_str,
                               traceback.format_exception(*sys.exc_info())))

    def import_object(self, import_str, *args, **kwargs):
        return self.import_class(import_str)(*args, **kwargs)

    def import_module(self, import_str):
        __import__(import_str)
        return sys.modules[import_str]


class BasicModel(object):
    def __init__(self, config):
        self._name = 'BasicModel'
        self.config = {}
        self.check_list = []
        # self.setup(config)
        # self.check()

    def set_default(self, k, v):
        if k not in self.config:
            self.config[k] = v

    def setup(self, config):
        pass

    def check(self):
        for e in self.check_list:
            if e not in self.config:
                print(e, end='\n')
                print('[Model] Error %s not in config' % e, end='\n')
                return False
        return True

    def build(self):
        pass

    def check_list(self, check_list):
        self.check_list = check_list


class KNRM(BasicModel):
    def __init__(self, config):
        super(KNRM, self).__init__(config)
        self._name = 'KNRM'
        self.check_list = [ 'text1_maxlen', 'kernel_num','sigma','exact_sigma',
                            'embed', 'embed_size', 'vocab_size']
        self.setup(config)
        if not self.check():
            raise TypeError('[KNRM] parameter check wrong')
        print('[KNRM] init done')

    def setup(self, config):
        self.set_default('kernel_num', 11)
        self.set_default('sigma', 0.1)
        self.set_default('exact_sigma', 0.001)
        if not isinstance(config, dict):
            raise TypeError('parameter config should be dict:', config)
        self.config.update(config)

    def build(self):
        # 定义RBF核函数
        def Kernel_layer(mu,sigma):
            def kernel(x):
                return K.tf.exp(-0.5 * (x - mu) * (x - mu) / sigma / sigma)
            return Activation(kernel)

        # 输入
        query = Input(name='query', shape=(self.config['text1_maxlen'],))
        Utility().show_layer_info('Input', query)
        doc = Input(name='doc', shape=(self.config['text2_maxlen'],))
        Utility().show_layer_info('Input', doc)

        # Embedding层把query中每个词映射成词向量
        embedding = Embedding(self.config['vocab_size'], self.config['embed_size'], weights=[self.config['embed']], trainable=self.config['train_embed'])
        q_embed = embedding(query)
        Utility().show_layer_info('Embedding', q_embed)
        d_embed = embedding(doc)
        Utility().show_layer_info('Embedding', d_embed)

        # 将q和d的向量分别点乘
        mm = Dot(axes=[2, 2], normalize=True)([q_embed, d_embed])
        Utility().show_layer_info('Dot', mm)
        # Kernel - Pooling
        # kernels to convert word-word interactions in the translation matrix M
        # to query - document ranking features
        KM = []
        for i in range(self.config['kernel_num']):
            mu = 1. / (self.config['kernel_num'] - 1) + (2. * i) / (self.config['kernel_num'] - 1) - 1.0
            sigma = self.config['sigma']
            if mu > 1.0:
                sigma = self.config['exact_sigma']
                mu = 1.0
            # 将相似度矩阵mm中的每一个值经过高斯核的激活函数，映射成另一个矩阵mm_exp
            mm_exp = Kernel_layer(mu, sigma)(mm)
            Utility().show_layer_info('Exponent of mm:', mm_exp)
            # 对于矩阵mm_exp中的每一个，q
            mm_doc_sum = Lambda(lambda x: K.tf.reduce_sum(x,2))(mm_exp)
            Utility().show_layer_info('Sum of document', mm_doc_sum)

            mm_log = Activation(K.tf.log1p)(mm_doc_sum)
            Utility().show_layer_info('Logarithm of sum', mm_log)

            mm_sum = Lambda(lambda x: K.tf.reduce_sum(x, 1))(mm_log)
            Utility().show_layer_info('Sum of all exponent', mm_sum)
            KM.append(mm_sum)
        #
        # Learning to Rank
        Phi = Lambda(lambda x: K.tf.stack(x, 1))(KM)
        Utility().show_layer_info('Stack', Phi)
        if self.config['target_mode'] == 'classification':
            out_ = Dense(2, activation='softmax', kernel_initializer=RandomUniform(minval=-0.014, maxval=0.014), bias_initializer='zeros')(Phi)
        elif self.config['target_mode'] in ['regression', 'ranking']:
            out_ = Dense(1, kernel_initializer=RandomUniform(minval=-0.014, maxval=0.014), bias_initializer='zeros')(Phi)
        Utility().show_layer_info('Dense', out_)

        model = Model(inputs=[query, doc], outputs=[out_])
        return model
