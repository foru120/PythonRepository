import threading
import time

class client_thread():
    def __init__(self, clientname, sec):
        self.clientname = clientname
        self.sec = sec

    def __call__(self):
        while True:
            print('{} - 지연 {} '.format(self.clientname, self.sec))
            time.sleep(self.sec)

clientA = threading.Thread(target=client_thread('clientA', 0.1))
clientB = threading.Thread(target=client_thread('clientB', 0.1))
clientC = threading.Thread(target=client_thread('clientC', 2))
clientD = threading.Thread(target=client_thread('clientD', 0.1))
clientE = threading.Thread(target=client_thread('clientE', 0.1))
clientF = threading.Thread(target=client_thread('clientF', 0.1))
clientG = threading.Thread(target=client_thread('clientG', 0.1))
clientH = threading.Thread(target=client_thread('clientH', 1))

clientA.start()
clientB.start()
clientC.start()
clientD.start()
clientE.start()
clientF.start()
clientG.start()
clientH.start()