# smart_base_2.py
import camera_base
import phone_base
print("-----")
while True:
    choice = input('what do you want? :')
    if choice == '0':
        break

    if choice == '1':
        camera_base.photo()
    elif choice == '2':
        phone_base.makeacall()
    elif choice == '3':
        print('나중에 구현될 기능')
print("프로그램이 종료되었습니다.")
