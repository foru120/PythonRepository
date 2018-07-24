from socket import *
myip = '127.0.0.1'
myport = 62580
address = (myip, myport)
sevsocket = socket(AF_INET, SOCK_STREAM)
sevsocket.bind(address)
sevsocket.listen()