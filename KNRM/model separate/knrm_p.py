from keras.initializers import RandomUniform
from keras.layers import *
from keras.models import Model


class KNRM(object):
    def __init__(self, config):
        self._name = 'KNRM'
        self.config = config

    def build(self):
        phi = Input(name='Xs', shape=(21,))

        # Learning to Rank
        out_ = Dense(1, kernel_initializer=RandomUniform(minval=-0.014, maxval=0.014), bias_initializer='zeros')(phi)
        model = Model(inputs=[phi], outputs=[out_])
        return model
