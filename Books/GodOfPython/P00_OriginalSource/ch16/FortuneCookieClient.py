# Fortune Cookie Client
from socket import *
sevip = '127.0.0.1'  # 실습환경에 따라서 바뀔 수 있다.
sevport = 62580
address = (sevip, sevport)
mysock = socket(AF_INET, SOCK_STREAM)
mysock.connect(address)

data = mysock.recv(1024)  # 연결 성공 메세지 받기
print(data.decode("UTF-8"))

data = mysock.recv(1024)  # Fortune Cookie 받기
print(data.decode("UTF-8"))

mysock.close()
