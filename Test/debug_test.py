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

# import tensorflow as tf
#
# x = [[[1,2,3],
#       [4, 5, 6]
#      ],
#      [[6,7,8],
#       [6,7,8]
#      ]
#     ]  # (2, 2, 3)
# y = tf.reduce_mean(x, (1, 2))
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(y))
#     # print(c3.get_shape())
#     # print(sess.run(c3))

# s = 'asdfqwerszz'
#
# def alphabet_counter(text, search):
#     cnt = 0
#     for alphabet in text:
#        if alphabet == search:
#            cnt += 1
#     return cnt
#
# def convert_word(word, cnt):
#     return chr(ord(word) + cnt - 26) if (ord(word) + cnt) > ord('z') else chr(ord(word) + cnt)
#
# for idx, word in enumerate(s):
#     print(convert_word(word, alphabet_counter(s[:idx], word)), end='')
# import time
# word_count = {}
#
# def convert_word_func(word, cnt):
#     cnt = cnt%26
#     return chr(ord(word) + cnt - 26) if (ord(word) + cnt) > ord('z') else chr(ord(word) + cnt)
#
# def file_read():
#     with open('test.txt', mode='rt') as f:
#         return f.read()
#
# def file_write(d):
#     with open('result.txt', mode='wt') as f:
#         f.write(d)
#
# result_text = ''
# stime = time.time()
# for idx, word in enumerate(file_read()):
#     convert_word = ''
#     if word in word_count:
#         convert_word = convert_word_func(word, word_count[word])
#         word_count[word] += 1
#     else:
#         convert_word = word
#         word_count[word] = 1
#     result_text += convert_word
#     # print(convert_word, end='')
# file_write(result_text)
# etime = time.time()
#
# print('\n' + str(etime - stime))

# def generate_data():
#     with open('test.txt', mode='wt') as file:
#         file.write('xyza'*100000)
#
# generate_data()
# import numpy as np
# a = np.array([['16', '20', '5', '3', '16'],
#               ['13', '4', '15', '5', '2'],
#               ['7', '19', '6', '5', '19'],
#               ['19', '14', '13', '2', '4'],
#               ['9', '8', '5', '19', '4']
#              ])
#
# def search_min_max_index(value):
#     min_idx, max_idx = [], []
#
#     min_tmp = [idx.tolist() for idx in np.where(a.astype(np.int32) == a.astype(np.int32).min())]
#     max_tmp = [idx.tolist() for idx in np.where(a.astype(np.int32) == a.astype(np.int32).max())]
#
#     for idx in zip(min_tmp[0], min_tmp[1]):
#         min_idx.append([idx[0], idx[1]])
#
#     for idx in zip(max_tmp[0], max_tmp[1]):
#         max_idx.append([idx[0], idx[1]])
#
#     return min_idx, max_idx
#
# def check_min_max_value(value):
#     tot_tmp = []
#     min_idx, max_idx = search_min_max_index(value)
#
#     for row_idx in range(value.shape[0]):
#         tmp = []
#         for col_idx in range(value.shape[1]):
#             if [row_idx, col_idx] in min_idx:  # min 값에 해당하는 인덱스가 있는 경우
#                 tmp.append(value[row_idx][col_idx] + '(MIN)')
#             elif [row_idx, col_idx] in max_idx:  # max 값에 해당하는 인덱스가 있는 경우
#                 tmp.append(value[row_idx][col_idx] + '(MAX)')
#             else:
#                 tmp.append(value[row_idx][col_idx])
#         tot_tmp.append(tmp)
#
#     return np.array(tot_tmp)
#
# print(check_min_max_value(a))
#
#
# data = np.random.randint(1, 21, size=(5, 5))
# min_number, max_number = np.min(data), np.max(data)
# min_idx = data == min_number
# max_idx = data == max_number
#
# data = data.astype(np.str)
#
# data[min_idx] = str(min_number) + '(MIN)'
# data[max_idx] = str(max_number) + '(MAX)'
# print(data)

# print(np.where([[True, False], [True, True]], [[1, 2], [3, 4]], [[9, 8], [7, 6]]))

# for c, x, y in zip([[True, False], [True, True]], [[1, 2], [3, 4]], [[9, 8], [7, 6]]):
#     print(c, x, y)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def data_gen(t=0):
    cnt = 0
    while cnt < 1000:
        cnt += 1
        t += 0.1
        yield t, np.sin(2*np.pi*t) * np.exp(-t/10.)


def init():
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(0, 10)
    del xdata[:]
    del ydata[:]
    line.set_data(xdata, ydata)
    return line,

fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.grid()
xdata, ydata = [], []


def run(data):
    # update the data
    t, y = data
    xdata.append(t)
    ydata.append(y)
    xmin, xmax = ax.get_xlim()

    if t >= xmax:
        ax.set_xlim(xmin, 2*xmax)
        ax.figure.canvas.draw()
    line.set_data(xdata, ydata)

    return line,

ani = animation.FuncAnimation(fig, run, data_gen, blit=False, interval=10,
                              repeat=False, init_func=init)
plt.show()