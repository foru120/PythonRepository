# thread_04.py
import threading
import queue
import time
import random

data_from_client = {}  # 두 클라이언트의 가위바위보의 데이터를 넣기 위한 사전이다.
#lockA = threading.Lock()


def client(name, inputdata, sec):
    time.sleep(sec)  # 클라이언트로부터 데이터를 전송받을 때 걸리는 시간 대신 sleep 사용
    data_from_client[name] = inputdata  # 전송받은 데이터를 사전에 저장


def result():
    #time.sleep(5) #동기화를 위한 지연 5초
    print("A :", data_from_client["A"]) #KeyError 발생
    print("B :", data_from_client["B"])

A = threading.Thread(target=client, args=("A", random.randint(0, 2), random.
                                          randint(1, 4)))
A.start()
B = threading.Thread(target=client, args=("B", random.randint(0, 2), random.
                                          randint(1, 4)))
B.start()
C = threading.Thread(target=result, args=())  # 가위바위보 결과를 처리하는 쓰레드 생성

C.start()
