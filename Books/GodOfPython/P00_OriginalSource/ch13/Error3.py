# Error3.py

try:
    f = open("c:/hellopython.txt", 'r')
except FileNotFoundError:
    print("no file")
    raise SystemExit  # 프로그램 종료

print("next code...")
