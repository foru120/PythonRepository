def makeacall():
    print('Make a Call')

def photo():
    print('Take a photo')

while True:
    choice = input('what do you want?')

    if choice=='0':
        break

    if choice=='1':
        makeacall()
    elif choice=='2':
        photo()
    elif choice=='3':
        print('나중에 구현될 기능')

print('프로그램이 종료되었습니다.')