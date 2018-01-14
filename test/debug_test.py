# from collections import defaultdict
# import random
#
# t = int(input(''))
#
# for _ in range(t):
#     play_list = input('').split('\t')
#     artist_list = input('').split('\t')
#     dataset = defaultdict(list)
#     _str = ''
#
#     for idx, artist in enumerate(artist_list):
#         dataset[artist].append(play_list[idx])
#
#     for key, value in dataset.items():
#         if len(value) > 1:
#             random.shuffle(value)
#             dataset[key] = value
#
#     max_cnt = max([len(value) for value in dataset.values()])
#     cnt = 0
#
#     for idx in range(max_cnt):
#         for value in dataset.values():
#             if len(value) > idx:
#                 if cnt == 0:
#                     _str += value[idx]
#                 else:
#                     _str += '\t' + value[idx]
#             cnt += 1
#
#     print(_str)

import tensorflow as tf

c1 = tf.constant([1, 3, 5, 7, 9, 0, 2, 4, 6, 8])
c2 = tf.tile(c1, [3])
c3 = tf.random_normal([10, 0, 1])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(c2))
    print(c3.get_shape())
    print(sess.run(c3))