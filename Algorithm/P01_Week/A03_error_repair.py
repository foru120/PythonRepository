import numpy as np
import os, psutil

# 시작 메모리 체크
proc1 = psutil.Process(os.getpid())
mem1 = proc1.memory_info()
before_start = mem1[0]

# 패리티 검사
def parity_check(arr):
    check_arr = np.append(np.sum(arr, axis=0) % 2 == 0, np.sum(arr, axis=1) % 2 == 0)  # 각 열과 행의 덧셈이 짝수인 원소에 대해 True, False 로 출력 후 해당 배열 Merge
    if np.sum(check_arr) == len(check_arr):
        return True
    return False

# change bit 검사
def change_bit(arr):
    col_bol = (np.sum(arr, axis=0) % 2 != 0).tolist()
    row_bol = (np.sum(arr, axis=1) % 2 != 0).tolist()
    if sum(col_bol) == 1 and sum(row_bol) == 1:
        return (True, (row_bol.index(1)+1, col_bol.index(1)+1))
    else:
        return (False, ())

# 행렬 크기 설정
n = int(input('행렬의 크기를 입력하세요 : '))
arr = np.zeros((n, n))
i = 0

# 행렬 초기화
while True:
    try:
        var_list = [int(var) for var in input('행렬의 값 : ').split(' ')]
        if len(var_list) != n:
            raise ValueError()
    except ValueError:
        print('잘못 입력하였습니다.')
        continue

    arr[i] = var_list
    i += 1

    if i == n: break

if parity_check(arr):
    print('OK')
else:
    change_var = change_bit(arr)
    if change_var[0]:
        print('Change bit ' + str(change_var[1]))
    else:
        print('Corrupt')

# 실행 후 맨 밑에서 코드 구동 후 메모리 체크
proc = psutil.Process(os.getpid())
mem = proc.memory_info()
after_start = mem[0]
print('memory use : ', after_start-before_start)