import sys
try:
    f = open('D:/02.Python/ch13/hellopython.txt', 'r')
except FileNotFoundError:
    print('no file')
    sys.exit(0) #인수 0은 정상 종료를 뜻함

print('next code...')