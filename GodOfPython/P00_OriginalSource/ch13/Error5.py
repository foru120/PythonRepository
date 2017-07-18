# Error5.py
import sys

try:
    f = open("C:/hellopython.txt", 'r')
except FileNotFoundError:
    print("no file")
else:
    print(f.read())  # 파일이 열렸을 때만 실행되는 코드
    f.close()  # 파일이 열렸을 때만 실행되는 코드

print("next code...")
