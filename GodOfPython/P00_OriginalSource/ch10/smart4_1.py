# smart4_1.py
import smtpkg4.camera  # camera 패키지까지만 import (smart3.py와 코드를 비교해보자.)
import smtpkg4.phone  # " "


def smart_on():
    while True:
        choice = input('what do you want? :')
        if choice == '0':
            break
        if choice == '1':
            smtpkg4.camera.camera.photo()
        elif choice == '2':
            smtpkg4.phone.phone.makeacall()
        elif choice == '3':
            print('나중에 구현될 기능')
    print("프로그램이 종료되었습니다.")

if __name__ == '__main__':
    smart_on()
