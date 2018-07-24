from socket import *
import threading

sevip = '127.0.0.1'
sevport = 62581
address = (sevip, sevport)

sevsock = socket(AF_INET, SOCK_STREAM)
sevsock.bind(address)
sevsock.listen()

print('waiting for connection...')
clisock, cliaddr = sevsock.accept()
print('connection from {}'.format(cliaddr))
print('If you want to leave chat, just type !quit\n')

def receive():
    global clisock

    while True:
        data = clisock.recv(1024)
        print(data.decode('utf-8'), ' *from Client')
    clisock.close()

thread_recv = threading.Thread(target=receive, args=())
thread_recv.start()

while True:
    try:
        data = input('')
    except KeyboardInterrupt:
        break

    if data == '!quit':
        clisock.close()
        break

    clisock.send(bytes(data, 'utf-8'))

sevsock.close()

print('disconnected')