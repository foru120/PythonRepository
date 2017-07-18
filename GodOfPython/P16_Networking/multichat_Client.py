from socket import *
import threading

sevip = '127.0.0.1'
sevport = 62580
address = (sevip, sevport)

mysock = socket(AF_INET, SOCK_STREAM)
print('connecting to server {} on port {}...'.format(sevip, sevport))
mysock.connect(address)
print('connection complete')
print('If you want to leave chat, just type !quit\n')

def receive():
    global mysock

    while True:
        try:
            data = mysock.recv(1024)
        except ConnectionError:
            print('서버와의 접속이 끊겼습니다. Enter키를 누르세요')
            break

        if not data:
            print('서버로부터 정상적으로 로그아웃했습니다.')
            break

        print(data.decode('utf-8'))

    print('소켓의 읽기 버퍼를 닫습니다.')
    mysock.shutdown(SHUT_RD)

def mainthread():
    global mysock
    thread_recv = threading.Thread(target=receive, args=())
    thread_recv.start()

    while True:
        try:
            data = input('')
        except KeyboardInterrupt:
            continue

        if data == '!quit':
            print('서버와의 접속을 끊는 중입니다.(!quit)')
            break

        try:
            mysock.send(bytes(data, 'utf-8'))
        except ConnectionError:
            break

    print('소켓의 쓰기 버퍼를 닫습니다.')
    mysock.shutdown(SHUT_WR)
    thread_recv.join()

thread_main = threading.Thread(target=mainthread, args=())
thread_main.start()

thread_main.join()

mysock.close()
print('소켓을 닫습니다.')

print('클라이언트 프로그램이 정성적으로 종료되었습니다.')