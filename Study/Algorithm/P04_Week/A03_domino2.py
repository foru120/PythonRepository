import math
import time
import psutil
import os

###################################################################################################
## 1. 문제        : 도미노2 (고급)
## 2. 소요 시간   : 0.0 초 (소수점 6자리 반올림)
## 3. 사용 메모리 : 65536 byte
## 4. 만든 사람   : 길용현
###################################################################################################

# 시작 메모리 체크
proc1 = psutil.Process(os.getpid())
mem1 = proc1.memory_info()
before_start = mem1[0]

domino = []
max_value = 0
pos = 0

while True:
    try:
        domino_cnt = int(input('도미노의 개수를 입력하세요.(1~40) : '))
        if 1 <= domino_cnt <= 40:
            break
        else:
            print('도미노의 개수는 1 ~ 40 사이만 가능합니다.\n')
    except ValueError:
        print('올바른 수가 아닙니다. 다시 입력해주세요.\n')

while domino_cnt > 0:
    try:
        domino_val = [int(num) for num in input('도미노의 값을 입력하세요.(x1, x2) : ').split(' ')]

        if (len(domino_val) == 2) and (0 <= domino_val[0] <= 9) and (0 <= domino_val[1] <= 9):
            domino.append(domino_val)
            domino_cnt -= 1
        else:
            raise ValueError
    except ValueError:
        print('올바른 수가 아닙니다. 다시 입력해주세요.\n')

# 시작 시간 체크
stime = time.time()

for values in sorted(domino, key=lambda v: sum(v)):
    max_value += sum(values)*math.pow(10, pos)
    pos += 1

print(int(max_value))

# 종료 시간 체크
etime = time.time()
print('consumption time : ', round(etime-stime, 6))

# 실행 후 맨 밑에서 코드 구동 후 메모리 체크
proc = psutil.Process(os.getpid())
mem = proc.memory_info()
after_start = mem[0]
print('memory use : ', after_start-before_start)