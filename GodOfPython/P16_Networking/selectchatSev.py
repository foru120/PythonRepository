from socket import *
from select import *

sevip = ''
sevport = 62580

sevsock = socket(AF_INET, SOCK_STREAM)
sevsock.bind((sevip, sevport))
sevsock.listen()

clients_list = []
clients_ids = []

def sock_close(sock):
    clients_ids.remove(sock.fileno())
    clients_list.remove(sock)

    print('현재 연결된 사용자 : {}\n'.format(clients_ids), end='')
    sock.close()

    print('클라이언트 소켓을 정상적으로 닫았습니다')
    print('---------------------------------------')

while True:
    rlist, wlist, xlist = select([sevsock]+clients_list, [], [], 1) #입력버퍼, 출력버퍼, 에러

    if rlist:
        for sock in rlist:
            if sock == sevsock: #접속 요청시
                clisock, addr = rlist[0].accept()
                clients_list.append(clisock)
                clients_ids.append(clisock.fileno())
                print('{}가 접속하였습니다.'.format(clisock.fileno()))
                print('{}가 접속하였습니다.'.format(addr))
                print('현재 연결된 사용자 : {}\n'.format(clients_ids))
            else: #데이터 요청시
                try:
                    data = sock.recv(1024)
                except ConnectionError:
                    print('{}와 연결이 끊겼습니다. #code1'.format(sock.fileno()))
                    sock_close(sock)

                if not data:
                    print('{}이 연결 종료 요청을 합니다. #code0'.format(sock.fileno()))
                    sock.send(bytes('서버에서 클라이언트 정보를 삭제하는 중입니다.', 'utf-8'))
                    sock_close(sock)
                else:
                    for other_sock in clients_list:
                        if sock == other_sock:
                            continue
                        data_with_ID = bytes(str(sock.fileno()), 'utf-8') + b' : ' + data
                        other_sock.send(data_with_ID)