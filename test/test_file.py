from collections import defaultdict

case_num = int(input(''))
play_list = input('').split('\t')
artist_list = input('').split('\t')

dataset = defaultdict(list)

for idx, artist in enumerate(artist_list):
    dataset[artist].append(play_list[idx])

print(dataset)