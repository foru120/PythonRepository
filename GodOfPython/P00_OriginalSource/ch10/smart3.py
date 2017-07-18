# smart3.py
import smtpkg3.camera.camera
import smtpkg3.phone.phone


def smart_on():
    while True:
        choice = input('what do you want? :')
        if choice == '0':
            break
        if choice == '1':
            smtpkg3.camera.camera.photo()
        elif choice == '2':
            smtpkg3.phone.phone.makeacall()
        elif choice == '3':
            print('나중에 구현될 기능')
    print("프로그램이 종료되었습니다.")

if __name__ == '__main__':
    smart_on()
