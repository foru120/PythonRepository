# thread_05.py
import threading

Q = 1000


def drink(max):
    global Q
    for i in range(0, max):
        Q -= 1

A = threading.Thread(target=drink, args=(400,))
B = threading.Thread(target=drink, args=(550,))
A.start()
B.start()

A.join()  # A 쓰레드 종료 전까지 대기
B.join()  # B 쓰레드 종료 전까지 대기

print(Q)
