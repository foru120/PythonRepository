# smart2.py
import smtpkg2.camera
import smtpkg2.phone


def smart_on():
    while True:
        choice = input('what do you want? :')
        if choice == '0':
            break
        if choice == '1':
            smtpkg2.camera.photo()
        elif choice == '2':
            smtpkg2.phone.makeacall()
        elif choice == '3':
            print('나중에 구현될 기능')
    print("프로그램이 종료되었습니다.")

if __name__ == '__main__':
    smart_on()
