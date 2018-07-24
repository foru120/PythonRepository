from socket import *
import threading

mylock = threading.Lock()

sevsock = socket(AF_INET, SOCK_STREAM)
sevsock.bind(('127.0.0.1', 62580))
sevsock.listen()

print('Start Chat - Server')
print('waiting for connection...\n')

cli_list = []
cli_ids = []

def receive(clisock):
    global cli_list

    while True:
        try:
            data = clisock.recv(1024)
        except ConnectionError: #비정상 종료
            print('{}와 연결이 끊겼습니다. #code1'.format(clisock.fileno()))
            break

        if not data: #정상 종료
            print('{}이 연결 종료 요청을 합니다. #code0'.format(clisock.fileno()))
            clisock.send(bytes('서버에서 클라이언트 정보를 삭제하는 중입니다.', 'utf-8'))
            break

        data_with_ID = bytes(str(clisock.fileno()), 'utf-8') + b':' + data

        for sock in cli_list:
            if sock != clisock:
                sock.send(data_with_ID)

    mylock.acquire()
    cli_ids.remove(clisock.fileno())
    cli_list.remove(clisock)
    mylock.release()

    print('현재 연결된 사용자 : {}\n'.format(cli_ids), end='')
    clisock.close()

    print('클라이언트 소켓을 정상적으로 닫았습니다')
    print('---------------------------------------')
    return 0

def connection():
    global cli_list
    global cli_ids

    while True:
        clisock, cliaddr = sevsock.accept()
        mylock.acquire()
        cli_list.append(clisock)
        cli_ids.append(clisock.fileno())
        mylock.release()
        print('{}가 접속하였습니다.'.format(clisock.fileno()))
        print('{}가 접속하였습니다.'.format(cliaddr))
        print('현재 연결된 사용자 : {}\n'.format(cli_ids))
        thread_recv = threading.Thread(target=receive, args=(clisock,))
        thread_recv.start()

thread_connection = threading.Thread(target=connection, args=())
thread_connection.start()

thread_connection.join()

sevsock.close()