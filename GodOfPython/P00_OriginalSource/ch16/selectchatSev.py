#selectchatSev.py
from socket import *
from select import *

sevip = ''
sevport = 62580

sevsock = socket(AF_INET, SOCK_STREAM)
sevsock.bind((sevip, sevport))
sevsock.listen()

clients_list = []

while True:
    rlist, wlist, xlist = select([sevsock] + clients_list, [], [], 1)
    if rlist:
        for sock in rlist:                         #변화가 있는 입력 버퍼 검사
            if sock == sevsock:                    #연결 요청이라면
                clisock, addr = rlist[0].accept()  #연결 요청 수락
                clients_list.append(clisock)       #새로운 소켓 생성
            else:                                      #연결 요청이 아니라면
                data = sock.recv(1024)                 #데이터 수신
                if not data:                               #상대가 연결을 끊은 경우
                    print("클라이언트가 연결종료합니다.")  
                    sock.close()                           #해당 소켓 닫기
                    clients_list.remove(sock)              #리스트에서 제거
                else:                                      #데이터 전송이라면 
                    for other_sock in clients_list:        #연결된 모든 클라이언트 소켓을 검사
                        if sock == other_sock:             #데이터를 보낸 클라이언트만 빼고
                            continue
                        other_sock.send(data)              #데이터를 연결된 모든 클라이언트에게 보냄
