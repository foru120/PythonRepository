# import socket
# print(socket.gethostbyname('www.google.com'))

from socket import *
mysock = socket(AF_INET, SOCK_STREAM)
print(mysock, mysock.fileno())