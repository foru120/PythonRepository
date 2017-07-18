#ThreadServer.py
from socket import *
import threading            #쓰레드를 사용하기 위해

sevip = '127.0.0.1'
sevport = 62581
address = (sevip, sevport)

sevsock = socket(AF_INET, SOCK_STREAM)
sevsock.bind(address)
sevsock.listen()
print("waiting for connection...")
clisock , cliaddr = sevsock.accept()
print("connection from {}".format(cliaddr))
print("If you want to leave chat, just type !quit\n")

#쓰레드에서 실행될 코드를 담은 함수를 정의
def receive():
    global clisock
    while True:
        data = clisock.recv(1024)  
        print(data.decode("UTF-8"), " *from Client")
    clisock.close()
    
thread_recv = threading.Thread(target = receive, args = ())    #쓰레드 생성
thread_recv.start()                                         #쓰레드 시작

while True:
    try:
        data = input("")
    except KeyboardInterrupt:
        break
    if data =='!quit' or '':            #!quit를 입력하면 while루프를 끝낸다.
        clisock.close()
        break
    clisock.send(bytes(data,"UTF-8"))       
sevsock.close()

print("disconnected")

