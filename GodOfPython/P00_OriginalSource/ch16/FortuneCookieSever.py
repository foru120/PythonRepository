# Fortune Cookie Server

zop = """Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!"""  # 19개의  문장

fortunelist = zop.split('\n')  # 각 문장을 fortunelist(리스트)의 항목으로 만든다

import socket
import random


def choice():  # fortunelist의 항목의 범위 내에서 무작위 숫자 생성
    return random.randint(0, len(fortunelist) - 1)

myip = '127.0.0.1'  # 실습환경에 따라서 바뀔 수 있다.(자신의 IP 또는 루프백 주소)
myport = 62580
address = (myip, myport)

sevsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sevsock.bind(address)
sevsock.listen()

while True:
    print("waiting for connection...")
    clisock, cliaddr = sevsock.accept()
    print("Connection from {}".format(cliaddr))
    clisock.send(b"This is Forthune Cookie Sever. Welcome!")  # 환영 메세지 전송
    clisock.send(bytes(fortunelist[choice()], 'utf-8'))  # 무작위 문장 전송
    clisock.close()  # 연결 끊음
sevsock.close()
