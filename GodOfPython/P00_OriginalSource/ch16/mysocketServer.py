# mysocketServer.py
from socketserver import ThreadingTCPServer, StreamRequestHandler
import sys

servip = ''
servport = 62580

addr = (servip, servport)


class ClientList():
    clist = []

    def addlist(self, sock):
        ClientList.clist.append(sock)

    def removelist(self, sock):
        ClientList.clist.remove(sock)


class RequestHandler(StreamRequestHandler, ClientList):

    def handle(self):
        self.addlist(self.request)
        print("클라이언트가 접속했습니다.")
        while True:

            self.data = self.request.recv(1024)

            if not self.data:
                print("{}이 연결종료 요청을 합니다.".format(self.request.fileno()))
                break

            for sock in self.clist:
                if self.request == sock:
                    continue
                sock.send(bytes(str(self.request.fileno()),
                                'utf-8') + b":" + self.data)

        self.removelist(self.request)
        self.request.close()  # 클라이언트와 연결된 소켓 닫기

if __name__ == '__main__':
    servThreadsock = ThreadingTCPServer((addr), RequestHandler)
    print("waiting for connection...")
    servThreadsock.serve_forever()
