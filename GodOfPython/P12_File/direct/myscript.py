import sys

path = 'D:/02.Python/ch12/direct/'+sys.argv[1]
f = open(path, 'r')

sum=0

for line in f:
    sum += len(line.strip())

print('해당 문서의 총 단어수는 %d 개 입니다.' %(sum))
f.close()    