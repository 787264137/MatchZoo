import numpy as np
import keras.backend as K

mat1 = np.array([[10, 20], [30, 40]])
mat2 = np.array([[10, 20], [30, 40]])
mat1 + mat2  # numpy 中直接计算即可

val1 = K.tf.reduce_sum(mat1 + mat2, axis=0)

mat3 = np.array([[[10, 20, 1], [30, 40, 1]], [[10, 20, 1], [30, 40, 1]], [[10, 20, 1], [30, 40, 1]]])
2 * mat3
print(mat3.shape)
val2 = K.tf.reduce_sum(mat3, axis=0)
sess = K.tf.InteractiveSession()
sess.run(val2)

mat3 = np.array([[[10, 20, 1], [30, 40, 1]], [[10, 20, 1], [30, 40, 1]], [[10, 20, 1], [30, 40, 1]]])
matrix = np.zeros(shape=mat3[0].shape)
for X in mat3[0:]:
    # print(X)
    # print('---------')
    matrix += X
print(matrix)
"""
"""

import numpy as np

with open('D:\phi_\国民党_10193_phi.txt', 'r', encoding='utf-8') as f:
    while True:
        line = f.readline().strip()
        if not line: break
        itms = line.split(' ')
        qid2, phi = itms[0], [np.float32(x) for x in itms[1:]]
        print(qid2)
        print(phi)

"""
"""


def read_phi(filename):
    qid2 = []
    phi = []
    with open(filename, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline().strip()
            if not line: break
            itms = line.split(' ')
            q, p = itms[0], [np.float32(x) for x in itms[1:]]
            qid2.append(q)
            phi.append(p)
    return phi


phi = read_phi('D:\\phi_\\爆发_1542_phi.txt')
phi[2][0:5]
"""
"""
corpus = {}
with open('KNMR/data/corpus.txt', 'r', encoding='utf-8') as p:
    lines = p.readlines()
    for line in lines:
        x = line.strip().split(' ', maxsplit=1)
        if len(x) == 2:
            corpus[x[0]] = x[1]
        else:
            corpus[x[0]] = '空'
