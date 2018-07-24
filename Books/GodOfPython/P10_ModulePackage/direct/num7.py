import GodOfPython.P10_ModulePackage.testpkg

def test():
    while True:
        choice = input('what do you want?')

        if choice=='0':
            break

        if choice=='1':
            testpkg.call.call.call()
        elif choice=='2':
            testpkg.message.message.message()
        elif choice=='3':
            testpkg.photo.photo.photo()
        elif choice=='4':
            print('다른 기능을 준비중입니다.')

    print('프로그램이 종료되었습니다.')

if __name__=='__main__':
    test()