message = 'python'
f = open('D:/02.Python/ch12/direct/utf8.txt', 'wb')
f.write(message.encode('utf-8'))
f.close()
f = open('D:/02.Python/ch12/direct/utf16.txt', 'wb')
f.write(message.encode('utf-16'))
f.close()