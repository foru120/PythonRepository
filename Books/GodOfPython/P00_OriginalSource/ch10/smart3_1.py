# smart3_1.py
from smtpkg3.camera.camera import photo  # 바뀐 부분
from smtpkg3.phone.phone import makeacall  # 바뀐 부분


def smart_on():
    while True:
        choice = input('what do you want? :')
        if choice == '0':
            break
        if choice == '1':
            photo()  # 바뀐 부분
        elif choice == '2':
            makeacall()  # 바뀐 부분
        elif choice == '3':
            print('나중에 구현될 기능')
    print("프로그램이 종료되었습니다.")

if __name__ == '__main__':
    smart_on()
