# thread_01.py
# 실행을 멈추려면 ctrl+break
import threading
import time


def client_thread(clientname, sec):
    while True:
        print("{} - 지연 {} ".format(clientname, sec))
        time.sleep(sec)

clientA = threading.Thread(target=client_thread, args=("clientA", 0.1))
clientB = threading.Thread(target=client_thread, args=("clientB", 0.1))
clientC = threading.Thread(target=client_thread, args=("clientC", 2))
clientD = threading.Thread(target=client_thread, args=("clientD", 0.1))
clientE = threading.Thread(target=client_thread, args=("clientE", 0.1))
clientF = threading.Thread(target=client_thread, args=("clientF", 0.1))
clientG = threading.Thread(target=client_thread, args=("clientG", 0.1))
clientH = threading.Thread(target=client_thread, args=("clientH", 1))

clientA.start()
clientB.start()
clientC.start()
clientD.start()
clientE.start()
clientF.start()
clientG.start()
clientH.start()
