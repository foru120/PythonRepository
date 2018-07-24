###################################################################################################
## 1. 문제        : 소시지 공장 (고급)
## 2. 소요 시간   : 0.045003 초 (소수점 6자리 반올림)
## 3. 사용 메모리 : 143360 byte
## 4. 만든 사람   : 길용현
###################################################################################################
import numpy as np
import time
import psutil
import os

# 시작 메모리 체크
proc1 = psutil.Process(os.getpid())
mem1 = proc1.memory_info()
before_start = mem1[0]

while True:
    try:
        sausage_cnt = int(input('소시지 개수를 입력하세요 : '))

        if 1 <= sausage_cnt <= 5000:
            break
        else:
            raise ValueError
    except ValueError:
        print('잘못된 숫자를 입력하였습니다.\n')

while True:
    try:
        sausage_WH = [int(v) for v in input('소시지의 길이와 너비를 입력하세요 : ').split(' ')]

        if len(sausage_WH) == (sausage_cnt*2):
            sausage_WH = [(sausage_WH[i], sausage_WH[i+1]) for i in range(0, sausage_cnt*2, 2)]
            break
        else:
            raise ValueError
    except ValueError:
        print('잘못된 숫자 or 개수를 입력하였습니다.\n')

# 시작 시간 체크
stime = time.time()

sausage_list = []

for value in sorted(sausage_WH, key=lambda v: (np.mean(v), np.var(v))):
    if len(sausage_list) == 0:
        sausage_list.append({value: value})
        continue

    changed = False

    for i in range(len(sausage_list)-1, -1, -1):
        temp = [(key, max_value) for key, max_value in sausage_list[i].items()][0]
        SL, SW = temp[1]
        if SL <= value[0] and SW <= value[1]:
            sausage_list[i][temp[0]] = value
            changed = True
            break

    if changed == False:
        sausage_list.append({value: value})

print(len(sausage_list))

# 종료 시간 체크
etime = time.time()
print('consumption time : ', round(etime-stime, 6))

# 실행 후 맨 밑에서 코드 구동 후 메모리 체크
proc = psutil.Process(os.getpid())
mem = proc.memory_info()
after_start = mem[0]
print('memory use : ', after_start-before_start)

# 개수 : 50
# 길이, 너비 : 5436 7946 1350 3080 6245 3855 3909 5194 1257 4648 3688 5327 6101 5645 5921 1493 5758 6463 4909 1696 4405 7934 3528 4635 2079 471 1088 281 1258 7404 1610 3838 2362 5294 6553 3472 2415 7632 3596 3776 3074 4059 6481 6051 2482 705 6486 2352 2082 6662 3236 4270 854 4917 985 6312 4291 4248 4638 4670 6395 7345 7176 2863 5249 5926 5491 5346 2646 6287 4743 375 2373 5077 4748 5972 3862 6803 6494 3737 3378 6192 7388 7894 6386 3634 1759 7732 7394 5450 866 712 4294 2344 4906 7958 4789 4410 7904 4556
# 시간 : 12