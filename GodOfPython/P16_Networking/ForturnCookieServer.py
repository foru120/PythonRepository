zop = '''Beautiful is better than ugly.
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
Altuough that way not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespace are one honking great idea -- let's do more of those!'''

fortunelist = zop.split('\n')

import socket
import random

def choice():
    return random.randint(0, len(fortunelist)-1)

myip = '127.0.0.1'
myport = 62580
address = (myip, myport)

sevsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sevsock.bind(address)
sevsock.listen()

while True:
    print('waiting for connection...')
    clisock, cliaddr = sevsock.accept()
    print('Connection from {}'.format(cliaddr))
    clisock.send(b'This is Fortune Cookie Server. Welcome!')
    clisock.send(bytes(fortunelist[choice()], 'utf-8'))
    clisock.close()
    
sevsock.close()
