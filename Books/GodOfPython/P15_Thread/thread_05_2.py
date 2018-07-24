import threading

Q=10000000
thread_list = []
mylock = threading.Lock() 

def drink(max):
    global Q

    for i in range(0, max):
        mylock.acquire() # 락 획득
        Q-=1
        mylock.release() # 락 해제

for i in range(0, 2):
    thread_inst = threading.Thread(target=drink, args=(5000000,))
    thread_list.append(thread_inst)
    thread_inst.start()

for thread in thread_list:
    thread.join()

print(Q)