import threading
import time
import random

total = []
locking = threading.Lock()

def produce():
    global total
    num = 1
    while num <= 50:
        locking.acquire()
        total.append(random.randint(0,101))
        locking.release()
        num+=1

thread_A = threading.Thread(target=produce, args=())
thread_B = threading.Thread(target=produce, args=())
thread_A.start()
thread_B.start()

thread_A.join()
thread_B.join()

sum = 0

for num in total:
    sum += num

print(sum, len(total))