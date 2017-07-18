import psutil, time, os

###################################################################################################
## 1. 문제        : 좋은 수열 (고급)
## 2. 소요 시간   : 0.0 초 (소수점 6자리 반올림)
## 3. 사용 메모리 : 28672 byte
## 4. 만든 사람   : 길용현
###################################################################################################

# 시작 메모리 체크
proc1 = psutil.Process(os.getpid())
mem1 = proc1.memory_info()
before_start = mem1[0]

number = []

while True:
    digit = int(input('자리수를 입력하세요 : '))
    if 1 <= digit <= 80:
        break
    else:
        print('1~80 이하의 자리수만 가능합니다.')

# 시작 시간 체크
stime = time.time()

def check_sequence(number, d):
    num_len = len(number)-1
    if num_len >= (d*2-1):
        if number[num_len-(2*d-1):num_len-(d-1)] == number[num_len-(d-1):]:
            return False
        else:
            return check_sequence(number, d+1)
    return True

index = 0  # 현재 자리수

while True:
    # 가장 작은 수 부터 넣기 위해 1~3까지 순차적으로 입력 후 체크
    for i in range(1, 4):
        number.append(i)
        if check_sequence(number, 1):
            break
        else:
            del number[index]

    # 해당 자리수의 모든 값이 나쁜 수열을 만족하면 그 전 자리수의 값을 증가
    if len(number) != index+1:
        index -= 1
        for i in range(len(number)-1, -1, -1):
            bol = False
            while True:
                if number[i] != 3:  # 그 전 자리수의 값이 3이 아니면
                    number[i] += 1
                    if check_sequence(number, 1):
                        bol = True
                        break
                else:  # 그 전 자리수의 값이 3이면
                    del number[i]
                    index -= 1
                    break
            if bol is True:
                break

    index += 1

    if len(number) == digit:
        break

print(''.join([str(num) for num in number]))

# 종료 시간 체크
etime = time.time()
print('consumption time : ', round(etime-stime, 6))

# 실행 후 맨 밑에서 코드 구동 후 메모리 체크
proc = psutil.Process(os.getpid())
mem = proc.memory_info()
after_start = mem[0]
print('memory use : ', after_start-before_start)