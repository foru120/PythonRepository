import threading
import time

class client_thread():
    def __init__(self, word, sec):
        self.word = word
        self.sec = sec

    def __call__(self):
        while True:
            print(self.word)
            time.sleep(self.sec)

client_A = threading.Thread(target=client_thread('A', 1))
client_B = threading.Thread(target=client_thread('B', 1.5))
client_C = threading.Thread(target=client_thread('C', 2))

client_A.start()
client_B.start()
client_C.start()