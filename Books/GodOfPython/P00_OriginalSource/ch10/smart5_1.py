# smart5_1.py
from smtpkg5 import *  # '*'사용


def smart_on():
    while True:
        choice = input('what do you want? :')
        if choice == '0':
            break
        if choice == '1':
            camera.camera.photo()
        elif choice == '2':
            phone.phone.makeacall()
        elif choice == '3':
            print('나중에 구현될 기능')
    print("프로그램이 종료되었습니다.")

if __name__ == '__main__':
    smart_on()