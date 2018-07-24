# Error1.py

try:
    f = open("C:/hellopython.txt", 'r')
except FileNotFoundError:
    print("no file")

print("next code...")  # FileNotFoundError 예외가 발생해도 처리됨
