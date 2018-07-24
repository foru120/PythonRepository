# f = open('D:/02.Python/ch12/python.txt', 'w')
# f.write('파이썬')
# f.close()

# f = open('D:/02.Python/ch12/TestMode.txt', 'w')  
# f.write('python')
# f.close()

# f = open('D:/02.Python/ch12/TestMode.txt', 'a')
# f.write(' is simple')
# f.close()

# f = open('D:/02.Python/ch12/TestMode.txt', 'r')
# print(f.read())
# f.close()

# f = open('D:/02.Python/ch12/TestMode.txt', 'r+')
# print(f.read())
# f.close()

# f = open('D:/02.Python/ch12/TestMode.txt', 'r+')
# f.write('ABCD')
# f.close()

# f = open('D:/02.Python/ch12/TestMode.txt', 'r+')
# f.write('ABCD')
# print(f.read())
# f.close()

# f = open('D:/02.Python/ch12/TestFile.txt', 'r')
# for line in f:
#     print(line, end='')
# f.close()

# f = open('D:/02.Python/ch12/TestFile.txt', 'r')
# print(f.readlines()) #리스트 형태로 출력
# f.close()

# s = ['Beautiful!\n', 'Explicit!\n', 'Simple!\n', 'Complex!\n', 'Flat\n']
# f = open('D:/02.Python/ch12/List2File.txt', 'w')
# f.writelines(s) #리스트 형태로 입력
# f.close()

# s = ['Beautiful!', 'Explicit!', 'Simple!', 'Complex!', 'Flat']
# f = open('D:/02.Python/ch12/List2File.txt', 'w')
# f.write('\n'.join(s))
# f.close()

# f = open('D:/02.Python/ch12/List2File.txt', 'r')
# for line in f:
#     print(line.strip())
# f.close()

# s = 'Beautiful!/Explicit!/Simple!/Complex!'
# print('\n'.join(s.split('/')))

# f = open('D:/02.Python/ch12/seektell.txt', 'w+')
# f.write('ABCDEFGHIJ')
# f.flush() #쓰기 작업 후 읽기를 할때 flush() 사용
# print(f.tell())
# f.seek(4, 0) 
# print(f.tell())
# print(f.read(1))
# print(f.tell())
# f.seek(0, 1) #읽기 작업 후 쓰기 작업을 할때 seek() 사용
# f.write('***')
# f.tell()
# f.close()

# class myclass():
#     def __enter__(selF):
#         print('enter')

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         print('exit')

# with myclass() as c:
#     pass        

# f = open('D:/02.Python/ch12/linefeed.txt', 'w')    
# f.write('\n')
# f.close()

# f = open('D:/02.Python/ch12/linefeed.txt', 'wb')
# f.write(b'\n')
# f.close()

# import pickle
# f = open('D:/02.Python/ch12/pickle', 'wb')
# mylist = ['text', 'pickle', 1, 2, 3, 4]
# pickle.dump(mylist, f)
# f.close()

# f = open('D:/02.Python/ch12/pickle', 'rb')
# print(pickle.load(f))

# print('\u004E')

# f = open('D:/02.Python/ch12/unicode.txt', 'w')
# f.write('\u007f')
# f.close()

# python = '\ud30c\uc774\uc36c'
# print(python)
# print(python.encode('utf-8'), python.encode('utf-16'))

# f = open('D:/02.Python/ch12/unicode_16.txt', 'wb')
# f.write(python.encode('utf-16'))
# f.close()

# f = open('D:/02.Python/ch12/unicode_8.txt', 'wb')
# f.write(python.encode('utf-8'))
# f.close()

# byte_obj = 'Hello Python, 안녕 파이썬'.encode('utf-8')
# print(byte_obj)
# print(byte_obj.decode('utf-8'))

# print(bytes([237,140,140,236,157,180,236,141,172]))
# print(bytes(range(0,255)))

# python = bytes([237, 140, 140, 236, 157, 180, 236, 141, 172])
# print(python, python.decode('utf-8'))

# f = open('D:/02.Python/ch12/text.txt', 'w')
# f.write('\u0041\n')
# f.write('\ud30c\uc774\uc36c\n')
# f.write('\u2126\n')
# f.close()

# import codecs
# f = codecs.open('D:/02.Python/ch12/text.txt', 'a', 'utf-8')
# f.write('\u266B')
# f.close()