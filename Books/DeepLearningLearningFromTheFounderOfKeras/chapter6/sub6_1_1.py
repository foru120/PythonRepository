#-*-coding: utf-8-*-
#todo p.246 ~ p.249
#todo code 6-1 ~ code 6-4
#todo 6.1.1 단어와 문자의 원-핫 인코딩

# 단어 수준의 원-핫 인코딩하기
import numpy as np
samples = ['The cat sat on the mat.', 'The dog ate my homework.']

token_index = {}
for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index) +1

max_length = 10
results = np.zeros(shape=(len(samples),
                          max_length,
                          max(token_index.values()) + 1))

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1.

print(results.shape)

# 문자 수준 원-핫 인코딩하기
import string

samples = ['The cat sat on the mat.', 'The dog ate my homework.']
characters = string.printable
token_index = dict(zip(characters, range(1, len(characters) + 1)))

max_length = 50
results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))

for i, sample in enumerate(samples):
    for j, character in enumerate(sample):
        index = token_index.get(character)
        results[i, j, index] = 1.

# 케라스를 사용한 단어 수준의 원-핫 인코딩하기
from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples)  # 단어 인덱스를 구축

sequences = tokenizer.texts_to_sequences(samples)  # 문자열을 정수 인덱스의 리스트로 변환

one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')  # 직접 원-핫 이진 벡터 표현을 얻음

word_index = tokenizer.word_index
print('%s개의 고유한 토큰을 찾았습니다.' % len(word_index))

# 해싱 기법을 사용한 단어 수준의 원-핫 인코딩하기
import numpy as np

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

dimensionality = 1000
max_length = 10  # 문장당 최대 단어 개수

results = np.zeros((len(samples), max_length, dimensionality))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = abs(hash(word)) % dimensionality  # hash(): hash 함수를 커쳐 hash value 출력 (음수도 가능)
        results[i, j, index] = 1.