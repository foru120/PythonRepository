#tcphttp.py
from socket import *
import time
mysock = socket(AF_INET, SOCK_STREAM)
mysock.connect(('www.daum.net', 80))           #웹 서버의 주소와 포트번호 80

mysock.send(b"GET / HTTP/1.0\n")               #요청
mysock.send(b"Host : www.daum.net\n\n")        #요청
data = b''
while True:
    part = mysock.recv(1024)     	 
    if not part:			#페이지가 다 로드되었다면 서버에서 접속을 끊는다
        break
    data = data+part            

f = open("C:/gop/ch16/tcphttp.html", 'w', encoding="utf-8")     #파일로 저장 
f.write(data.decode("utf-8"))                                   #    "
f.close()                                                       #    "

print(data.decode("utf-8"))      #화면에 웹 서버로 부터 받은 데이터 출력
