# smart4.py
from smtpkg4.camera import *  # 수정
from smtpkg4.phone import *  # 수정


def smart_on():
    while True:
        choice = input('what do you want? :')
        if choice == '0':
            break
        if choice == '1':  # 에러 발생 위치
            camera.photo()
        elif choice == '2':
            phone.makeacall()
        elif choice == '3':
            print('나중에 구현될 기능')
    print("프로그램이 종료되었습니다.")

if __name__ == '__main__':
    smart_on()
