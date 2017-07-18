import threading
import time

class client_thread(threading.Thread):
    def __init__(self, clientname, sec):
        threading.Thread.__init__(self)
        self.clientname = clientname
        self.sec = sec

    def run(self):
        while True:
            print('{} - 지연 {} '.format(self.clientname, self.sec))
            time.sleep(self.sec)

clientA = client_thread('clientA', 0.1)
clientB = client_thread('clientB', 0.1)

clientA.start()
clientB.start()