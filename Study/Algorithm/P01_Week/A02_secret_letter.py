import numpy as np
import os, psutil

# 시작 메모리 체크
proc1 = psutil.Process(os.getpid())
mem1 = proc1.memory_info()
before_start = mem1[0]

secret_words = {'000000': 'A', '001111': 'B', '010011': 'C', '011100': 'D',
                '100110': 'E', '101001': 'F', '110101': 'G', '111010': 'H'}

# 문자 검증
def check_words(numbers):
    result = []
    number_list = [numbers[i-6:i] for i in range(6, len(numbers)+1, 6)]
    for i in range(len(number_list)):
        for word in secret_words:
            temp_arr = np.append(np.array([int(x) for x in number_list[i]]), [int(x) for x in word]).reshape(2, 6)
            cnt = (np.sum(temp_arr, axis=0) % 2 != 1).tolist().count(0)
            if 0 <= cnt <= 1:
                result.append(secret_words[word])
                break
        if len(result) != i+1:
            return i+1
    return ''.join(result)

# 문자 개수 입력
while True:
    try:
        n = int(input('문자 개수를 입력하세요 : '))
    except ValueError:
        print('잘못된 숫자를 입력하였습니다.')
        continue
    else:
        if 0 <= n < 10:
            break
        print('잘못된 숫자를 입력하였습니다.')

# 문자 입력
while True:
    numbers = input('문자를 입력하세요 : ')
    if len(numbers) == n*6:
        print(check_words(numbers))
        break
    else:
        print('잘못 입력하였습니다.')
        continue

# 실행 후 맨 밑에서 코드 구동 후 메모리 체크
proc = psutil.Process(os.getpid())
mem = proc.memory_info()
after_start = mem[0]
print('memory use : ', after_start-before_start)