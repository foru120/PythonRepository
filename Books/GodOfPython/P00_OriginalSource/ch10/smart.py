# smart.py
import camera
import phone
print("smart.py's module name is", __name__)  # 추가(모듈의 이름을 출력)
print("-----")
while True:
    choice = input('what do you want? :')
    if choice == '0':
        break
    if choice == '1':
        camera.photo()
    elif choice == '2':
        phone.makeacall()
    elif choice == '3':
        print('나중에 구현될 기능')
print("프로그램이 종료되었습니다.")
