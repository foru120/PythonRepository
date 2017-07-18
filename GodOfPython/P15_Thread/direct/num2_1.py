import threading
import time

class client_thread(threading.Thread):
    def __init__(self, word, sec):
        threading.Thread.__init__(self)
        self.word = word
        self.sec = sec

    def run(self):
        while True:
            print(self.word)
            time.sleep(self.sec)

client_A = client_thread('A', 1)
client_B = client_thread('B', 1.5)
client_C = client_thread('C', 2)

client_A.start()
client_B.start()
client_C.start()