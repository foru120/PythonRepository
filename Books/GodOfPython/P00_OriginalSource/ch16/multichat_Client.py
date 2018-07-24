# multichat_Client.py
from socket import *
import threading

sevip = '127.0.0.1'  # 또는 서버의 IP주소
sevport = 62580
address = (sevip, sevport)

mysock = socket(AF_INET, SOCK_STREAM)
print("connecting to server {} on port {}...".format(sevip, sevport))
mysock.connect(address)
print("connection complete")
print("If you want to leave chat, just type !quit\n")


def receive():
    global mysock
    while True:
        try:
            data = mysock.recv(1024)
        except ConnectionError:  # 서버 강제 종료
            print("서버와의 접속이 끊겼습니다. Enter키 를 누르세요")
            break

        if not data:  # 서버 정상 종료
            print("서버로부터 정상적으로 로그아웃 했습니다.")
            break

        print(data.decode("UTF-8"))  # 전송받은 데이터 출력
    print("소켓의 읽기버퍼를 닫습니다.")
    mysock.shutdown(SHUT_RD)


def mainthread():
    global mysock
    thread_recv = threading.Thread(target=receive, args=())
    thread_recv.start()
    while True:
        try:
            data = input("")  # 전송할 데이터 입력
        except KeyboardInterrupt:
            continue

        if data == '!quit':  # 접속 종료 시도
            print("서버와의 접속을 끊는 중 입니다.(!quit)")
            break

        try:
            mysock.send(bytes(data, "UTF-8"))  # 데이터 전송
        except ConnectionError:
            break
    print("소켓의 쓰기버퍼를 닫습니다.")
    mysock.shutdown(SHUT_WR)
    thread_recv.join()  # 자식 쓰레드가 종료되기 전까지 기다림

thread_main = threading.Thread(target=mainthread, args=())

thread_main.start()  # 메인 쓰레드 시작

#######################채팅 서버에 연결됨############################

thread_main.join()  # 메인 쓰레드가 종료되기 전까지 기다림

mysock.close()
print("소켓을 닫습니다.")

print("클라이언트 프로그램이 정상적으로 종료되었습니다.")
