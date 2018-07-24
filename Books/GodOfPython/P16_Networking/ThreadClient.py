from socket import *
import threading

sevip = '127.0.0.1'
sevport = 62581
address = (sevip, sevport)

mysock = socket(AF_INET, SOCK_STREAM)
print('connecting to server {} on port {}...'.format(sevip, sevport))
mysock.connect(address)
print('connection complete')
print('If you want to leave chat, just type !quit\n')

def receive():
    global mysock

    while True:
        data = mysock.recv(1024)
        print(data.decode('utf-8'), ' *from Server')

    mysock.close()

thread_recv = threading.Thread(target=receive, args=())
thread_recv.start()

while True:
    try:
        data = input('')
    except KeyboardInterrupt:
        break

    if data == '!quit':
        break

    mysock.send(bytes(data, 'utf-8'))

mysock.close()

print('disconnected')