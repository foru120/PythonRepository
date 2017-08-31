from collections import defaultdict
import random

t = int(input(''))

for _ in range(t):
    play_list = input('').split('\t')
    artist_list = input('').split('\t')
    dataset = defaultdict(list)
    _str = ''

    for idx, artist in enumerate(artist_list):
        dataset[artist].append(play_list[idx])

    for key, value in dataset.items():
        if len(value) > 1:
            random.shuffle(value)
            dataset[key] = value

    max_cnt = max([len(value) for value in dataset.values()])
    cnt = 0

    for idx in range(max_cnt):
        for value in dataset.values():
            if len(value) > idx:
                if cnt == 0:
                    _str += value[idx]
                else:
                    _str += '\t' + value[idx]
            cnt += 1

    print(_str)