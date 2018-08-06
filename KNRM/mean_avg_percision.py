import tensorflow as tf
import numpy as np
import keras.backend as K


def mean_average_precision(y_true, y_pred):
    def _to_list(x):
        if isinstance(x, list):
            return x
        return [x]

    y_true = _to_list(np.squeeze(y_true).tolist())
    y_pred = _to_list(np.squeeze(y_pred).tolist())
    s = 0.
    c = list(zip(y_true, y_pred))
    c = sorted(c, key=lambda x: x[1], reverse=True)
    ipos = 0
    for j, (g, p) in enumerate(c):
        if g > 0:
            ipos += 1.
            s += ipos / (j + 1.)
    if ipos == 0:
        s = 0.
    else:
        s /= ipos
    return s


def m_a_p(y_true, y_pred):
    MAP = tf.py_func(mean_average_precision, [y_true, y_pred], tf.float64)
    with tf.Session() as sess:
        MM = sess.run(MAP)
    return np.asarray(MM)

x = np.asarray([1, 2, 0, 4, 0])
y = np.asarray([5, 0, 10, 1, 1])

print(type(x))
MAP = mean_average_precision(x, y)
print('=====')
print(type(MAP))
MAP = np.asarray()
print(type(MAP))
print(MAP)
# with tf.Session() as sess:
#     print(sess.run(MAP))
# eval_MAP = mean_average_precision([1, 2, 0, 4, 0], [5, 0, 10, 1, 1])
# print(eval_MAP)

"""
Tensor mean_average_percision
"""

import tensorflow as tf
import numpy as np


def mean_average_precision1(y_true, y_pred):
    def _to_list(x):
        if isinstance(x, list):
            return x
        return [x]
    sess = tf.Session()
    y_true_np = y_true.eval(session=sess)
    y_pred_np = y_pred.eval(session=sess)
    sess.close()
    y_true = _to_list(np.squeeze(y_true_np).tolist())
    y_pred = _to_list(np.squeeze(y_pred_np).tolist())
    s = 0.
    c = list(zip(y_true, y_pred))
    c = sorted(c, key=lambda x: x[1], reverse=True)
    ipos = 0
    for j, (g, p) in enumerate(c):
        if g > 0:
            ipos += 1.
            s += ipos / (j + 1.)
    if ipos == 0:
        s = 0.
    else:
        s /= ipos
    return s


x = tf.convert_to_tensor([1, 2, 0, 4, 0])
y = tf.convert_to_tensor([5, 0, 10, 1, 1])

MAP = mean_average_precision1(x, y)
print(MAP)
# with tf.Session() as sess:
#     print(sess.run(MAP))
# eval_MAP = mean_average_precision([1, 2, 0, 4, 0], [5, 0, 10, 1, 1])
# print(eval_MAP)
# sess = tf.Session()
# y_true_np = y.eval(session = sess)
# sess.close()
# print(y_true_np)

"""
输入numpy.array,输出tensor
"""

# import tensorflow as tf
# import numpy as np
#
#
# def mean_average_precision2(y_true, y_pred):
#     def _to_list(x):
#         if isinstance(x, list):
#             return x
#         return [x]
#     sess = tf.Session()
#     y_true_np = y_true.eval(session=sess)
#     y_pred_np = y_pred.eval(session=sess)
#     sess.close()
#     y_true = _to_list(np.squeeze(y_true_np).tolist())
#     y_pred = _to_list(np.squeeze(y_pred_np).tolist())
#     s = 0.
#     c = list(zip(y_true, y_pred))
#     c = sorted(c, key=lambda x: x[1], reverse=True)
#     ipos = 0
#     for j, (g, p) in enumerate(c):
#         if g > 0:
#             ipos += 1.
#             s += ipos / (j + 1.)
#     if ipos == 0:
#         s = 0.
#     else:
#         s /= ipos
#     return s
#
#
# x = np.asarray([1, 2, 0, 4, 0])
# y = np.asarray([5, 0, 10, 1, 1])
#
# MAP = mean_average_precision2(x, y)
# print(MAP)