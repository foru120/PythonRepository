from GodOfPython.P10_ModulePackage.smtpkg4.camera import * # *으로는 모듈을 읽어올 수 없다
from GodOfPython.P10_ModulePackage.smtpkg4.phone import *

def smart_on():
    while True:
        choice = input('what do you want?')

        if choice=='0':
            break

        if choice=='1':
            camera.photo()
        elif choice=='2':
            phone.makeacall()
        elif choice=='3':
            print('나중에 구현될 기능')

    print('프로그램이 종료되었습니다.')

if __name__=='__main__':
    smart_on()