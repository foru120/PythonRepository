import threading, queue
import time
import random

data_from_client = {}

def client(name, inputdata, sec):
    time.sleep(sec)
    data_from_client[name] = inputdata

def result():
    #time.sleep(5)
    A.join()
    B.join()
    print('A : ', data_from_client['A'])
    print('B : ', data_from_client['B'])

A = threading.Thread(target=client, args=('A', random.randint(0,2), random.randint(1,4)))
A.start()

B = threading.Thread(target=client, args=('B', random.randint(0,2), random.randint(1,4)))
B.start()

C = threading.Thread(target=result, args=())
C.start()