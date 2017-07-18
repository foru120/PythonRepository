#ThreadClient.py
from socket import *
import threading

sevip = '127.0.0.1'
sevport = 62581
address = (sevip, sevport)

mysock = socket(AF_INET, SOCK_STREAM)
print("connecting to server {} on port {}...".format(sevip, sevport))
mysock.connect(address)
print("connection complete")
print("If you want to leave chat, just type !quit\n")

def receive():
    global mysock
    while True:
        data = mysock.recv(1024)        
        print(data.decode("UTF-8"), " *from Sever")
        
    mysock.close()

thread_recv = threading.Thread(target = receive, args = ())    #쓰레드 생성
thread_recv.start()                                         #쓰레드 시작

while True:
    try:
        data = input("")
    except KeyboardInterrupt:
        break
    if data =='!quit':                  #!quit를 입력하면 while루프를 끝낸다.
        break
    mysock.send(bytes(data,"UTF-8"))
mysock.close()

print("disconnected")
