#thread_03.py
import threading
import time

class client_thread(threading.Thread):
    def __init__(self, clientname, sec):
        threading.Thread.__init__(self)
        self.clientname = clientname
        self.sec = sec
    def run(self):
        while True:
            print("{} - 지연 {} ".format(self.clientname, self.sec))
            time.sleep(self.sec)

clientA = client_thread("clientA", 0.1)
clientB = client_thread("clientB", 0.1)
clientC = client_thread("clientC", 2)
clientD = client_thread("clientD", 0.1)
clientE = client_thread("clientE", 0.1)
clientF = client_thread("clientF", 0.1)
clientG = client_thread("clientG", 0.1)
clientH = client_thread("clientH", 1)

clientA.start()
clientB.start()
clientC.start()
clientD.start()
clientE.start()
clientF.start()
clientG.start()
clientH.start()
