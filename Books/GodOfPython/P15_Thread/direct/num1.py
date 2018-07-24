import threading
import time

def client_thread(word, sec):
    while True:
        print(word)
        time.sleep(sec)

thread_A = threading.Thread(target=client_thread, args=('A', 1))
thread_B = threading.Thread(target=client_thread, args=('B', 1.5))
thread_C = threading.Thread(target=client_thread, args=('C', 2))

thread_A.start()
thread_B.start()
thread_C.start()