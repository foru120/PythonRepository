# smart_mod.py
import camera_mod
import phone_mod


def smart_on():
    while True:
        choice = input('what do you want? :')
        if choice == '0':
            break
        if choice == '1':
            camera_mod.photo()
        elif choice == '2':
            phone_mod.makeacall()
        elif choice == '3':
            print('나중에 구현될 기능')
    print("프로그램이 종료되었습니다.")

if __name__ == '__main__':
    smart_on()  # import 될 때는 실행되지 않는 코드
