import GodOfPython.P10_ModulePackage.camera_base
import GodOfPython.P10_ModulePackage.phone_base
print('------')

while True:
    choice = input('what do you want?')
    if choice=='0':
        break
    if choice=='1':
        GodOfPython.P10_ModulePackage.camera_base.photo()
    elif choice=='2':
        GodOfPython.P10_ModulePackage.phone_base.makeacall()
    elif choice=='3':
        print('나중에 구현될 기능')
print('프로그램이 종료되었습니다.')