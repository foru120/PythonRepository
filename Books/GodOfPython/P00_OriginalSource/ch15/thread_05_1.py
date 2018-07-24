# thread_05_1.py
import threading

Q = 100000
thread_list = []


def drink(max):
    global Q
    for i in range(0, max):
        Q -= 1

for i in range(0, 2):
    thread_inst = threading.Thread(target=drink, args=(50000,))
    thread_list.append(thread_inst)  # 생성된 쓰레드를 thread_list에 저장
    thread_inst.start()  # 쓰레드 실행

for thread in thread_list:
    thread.join()  # 리스트 thread_list에 있는 쓰레드가 종료될 때까지 대기

print(Q)  # Q의 값 출력
