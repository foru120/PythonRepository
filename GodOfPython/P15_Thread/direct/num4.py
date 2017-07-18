import threading
import time
import random

total = []
locking = threading.Lock()

def produce(cnt):
    global total

    while cnt>0:
        locking.acquire()
        total.append(random.randint(1,101))
        locking.release()
        cnt-=1

thread_A = threading.Thread(target=produce, args=(20,))
thread_B = threading.Thread(target=produce, args=(30,))
thread_A.start()
thread_B.start()

thread_A.join()
thread_B.join()

sum = 0
for num in total:
    sum += num

print(sum, len(total))