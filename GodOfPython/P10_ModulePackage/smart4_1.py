import GodOfPython.P10_ModulePackage.smtpkg4.camera #패키지까지만 입력하면 모듈을 읽어들일 수 없다
import GodOfPython.P10_ModulePackage.smtpkg4.phone

def smart_on():
    while True:
        choice = input('what do you want?')

        if choice=='0':
            break

        if choice=='1':
            GodOfPython.P10_ModulePackage.camera.photo()
        elif choice=='2':
            GodOfPython.P10_ModulePackage.phone.makeacall()
        elif choice=='3':
            print('나중에 구현될 기능')

    print('프로그램이 종료되었습니다.')

if __name__=='__main__':
    smart_on()