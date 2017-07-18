# thread_05_2.py
import threading

Q = 100000
thread_list = []
mylock = threading.Lock()  # Lock객체 생성


def drink(max):
    global Q
    for i in range(0, max):
        mylock.acquire()  # 락(Lock)을 획득
        Q -= 1
        mylock.release()  # 락(Lock)을 반납

for i in range(0, 2):
    thread_inst = threading.Thread(target=drink, args=(50000,))
    thread_list.append(thread_inst)
    thread_inst.start()

for thread in thread_list:
    thread.join()

print(Q)
