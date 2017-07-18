from socket import *
sevip = '127.0.0.1'
sevport = 62580
address = (sevip, sevport)
mysock = socket(AF_INET, SOCK_STREAM)
mysock.connect(address)

data = mysock.recv(1024)
print(data.decode('utf-8'))

data = mysock.recv(1024)
print(data.decode('utf-8'))

mysock.close()