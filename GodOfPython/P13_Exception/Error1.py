try:
    f = open('D:/02.Python/ch13/hellopython.txt', 'r')
except FileNotFoundError:
    print('no file')

print('next code...')