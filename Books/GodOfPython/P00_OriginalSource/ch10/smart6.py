# C:\gop\ch10\smart6.py
import smtpkg5  # 최상위 패키지만 불러들임


def smart_on():
    while True:
        choice = input('what do you want? :')
        if choice == '0':
            break
        if choice == '1':
            smtpkg5.camera.camera.photo()  # 내부 패키지에 접근 가능
        elif choice == '2':
            smtpkg5.phone.phone.makeacall()
        elif choice == '3':
            print('나중에 구현될 기능')
    print("프로그램이 종료되었습니다.")

if __name__ == '__main__':
    smart_on()
