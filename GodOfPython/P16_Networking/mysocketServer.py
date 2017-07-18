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
        print('클라이언트가 접속했습니다.')

        while True:
            try:
                self.data = self.request.recv(1024)
            except ConnectionError:
                print('{}와 연결이 끊겼습니다. #code1'.format(sock.fileno()))
                break

            if not self.data:
                print('{}이 연결종료 요청을 합니다.'.format(self.request.fileno()))
                break

            for sock in self.clist:
                if self.request == sock:
                    continue

                self.request.send(bytes(str(self.request.fileno()), 'utf-8') + b' : ' + self.data)

        self.removelist(self.request)
        self.request.close()

if __name__=='__main__':
    servThreadsock = ThreadingTCPServer((addr), RequestHandler)
    print('waiting for connection...')
    servThreadsock.serve_forever()