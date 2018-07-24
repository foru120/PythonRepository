# Error2.py
import sys

try:
    f = open("c:/hellopython.txt", 'r')
except FileNotFoundError:
    print("no file")
    sys.exit(0)  # 코드 종료(인수 0은 정상 종료를 의미)

print("next code...")  # 예외 발생 시 처리되지 않는 코드`
