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

s = 'asdfqwerszz'
word_count = {}

def convert_word_func(word, cnt):
    return chr(ord(word) + cnt - 26) if (ord(word) + cnt) > ord('z') else chr(ord(word) + cnt)

for idx, word in enumerate(s):
    convert_word = ''
    if word in word_count:
        convert_word = convert_word_func(word, word_count[word])
        word_count[word] += 1
    else:
        convert_word = word
        word_count[word] = 1
    print(convert_word, end='')