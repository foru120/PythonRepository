import sys
import copy

# 01. 시퀀스 타입이란?
data = "Hello Python" #문자열
print(data) 
data1 = ["Hello", 77, ["python"]] #리스트(항목의 추가 삭제 가능)
print(data1)
data2 = ("Hello", 77, ["python"]) #튜플(항목의 추가 삭제 불가능)
print(data2)

print(data[0], data[1], data[2])
print(data1[0], data1[1], data1[2])
print(data2[0], data2[1], data2[2])

print(sys.getsizeof((1,2,3,4)))
print(sys.getsizeof([1,2,3,4])) #튜플이 리스트보다 메모리 크기가 더 작다

a="I'm programmer"
print(a[1])

s="first\nsecond" #문자열의 행을 바꾸는 확장열 \n
print(s)
s="first\tsecond" #탭을 의미하는 확장열 \t
print(s)
s="\\and\'and\"" #특별한 문자, 즉 직접 표현할 수 없는 문자를 사용할 경우
print(s)

print('\101') #8진수 -> 아스키 코드 
print('\x41') #16진수 -> 아스키 코드

# 02. 문자열 연산
# 02-01. 기본 연산
a="Hello"
b="python"
print(a+" "+b)
s="phthon"
print(s * 3)
print(s[0], s[1], s[-1], s[-2]) #-1 인덱스는 가장 마지막 인덱스를 뜻함

# 02-02. 분할 연산
s='python'
print(s[0:5]) #0 <= & < 5, 사이의 문자열 출력
print(s[0:]) #0 >= ,사이의 문자열 출력
print(s[:6]) #< 6, 사이의 문자열 출력
print(s[:]) #처음부터 끝까지 출력
print(s[0:-1]) #0<= & < -1(마지막) 사이의 문자열 출력

# 02-03. 확장분할 연산
print(s[0:-1:1]) #s[0:-1] 과 같다
print(s[0:6:2]) #step 이 2씩 증가
print(s[-1::-1]) #거꾸로 출력
print(s[::-1])
print(s[::1])

# 02-04. in 연산
print('p' in s)
print('py' in s)
print('pythom' in s)

# 02-05. 문자열 비교 연산
print('a'<'b'<'c')
print('apple'<'banana')
print('apple'<'Banana')

id_1="apple"
id_2="Banana"
print(id_1<id_2)
print(id_1.lower()<id_2.lower()) #lower() 메소드는 문자열에 있는 모든 대문자를 소문자로 변환한다
print(' '<'a') #문자열중에 공백이 가장 작다

# 02-06. 문자열 포맷팅
x=100
print("x is %s " %x)
x='python'
print("x is %s " %x)
x=3.14
print("x is %s " %x)
print("%d + %d = %d" %(3, 7, 10))

a=33; b=77
print(eval('a+b')) #eval 함수는 안의 표현식의 결과값을 출력
temp = "a+b"
print(eval(temp))
temp = a+b
print(temp)
a=11; b=22
print(temp)

a=1; b=2
print(repr('a+b'))
print(str('a+b'))
print(ascii('a+b')) #repr 함수와 동일하게 동작하지만 유니코드 문자의 경우 이스케이프 시퀀스로 변환
print("%r %s %a" %('a+b', 'a+b', 'a+b'))

# 02-07. 문자열 메소드
print('korea'.capitalize()) #문자열의 첫 문자를 대문자로 변경
print('AbCdEfG'.casefold()) #소문자로 변경하거나 비교를 위한 형태로 변경
s='python'
print(s.center(20, '*')) #문자열을 형식에 맞게 중앙 정렬
s="""Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense."""
print(s.count('better')) #함수로 전달된 문자열과 동일한 부분 문자의 개수를 반환
print(s.count('better', 31, 65))
print(s.count('better', 31))
print(s.count('better', 0, 64))
print('python'.endswith('on')) #문자열이 suffix로 끝나면 True 반환
print('python'.endswith('hon')) 
print('python'.endswith('tho',0,5))
print('{} {} {}'.format('python', 77, 3.14))
print('{1} {2} {0}'.format('python', 77, 3.14)) #format 메소드를 통한 포맷팅
print('fine thank you, and you?'.index('you')) #문자열을 왼쪽부터 검색하여 부분문자열이 최초로 나타내는 위치 반환
print('/'.join('python'))
print('-'.join('python'))
print(' '.join('python'))
print('/'.join(['dog', 'tiger', 'lion', 'monkey'])) #모든 시퀀스 타입에 대해 사용가능하고, 각 요소 사이에 값을 넣어주는 역할
print(' python\n '.strip()) #양쪽 끝부분의 공백이나 \n 제거
print(' python\n '.lstrip())
print(' python\n '.rstrip())
print(''.join('p/y/t/h/o/n'.split('/'))) #특정 문자열을 제거한다. 단, 반환값은 리스트 타입

# 03. 리스트
# 03-01. 리스트의 기본 개념
mylist = ['p','y','t','h','o','n']
for x in range(0, 6):
    print(mylist[x])
print(''.join(mylist[-1::-1])) #문자열로 변환
print(list('python')) #리스트로 변환
print(str(mylist))

mylist = ['p', 'y', 't', 'h','o', 'n', 3.5]
print(mylist)
print(type(mylist[6]))
mylist = ['p', 'y', 't', 'h', 'o', 'n', 3, '.', 5]
print(mylist[6], mylist[7], mylist[8])

mylist = [['h', 'e', 'l', 'l', 'o'], 'python', 3.5]
print(mylist[0][0], mylist[1][0])
mylang = [['hello', 'python'], ['good-bye', 'C']]
print(mylang[0][0][0:4])

# 03-02. 리스트 수정
mylist = ['p', 'y', 't', 'h', 'o', 'm']
print(mylist)
mylist[-1]='n' #리스트의 수정
print(mylist)

mystr = 'python'
print(id(mystr))
print(id(mystr.capitalize()))
print(mystr)

mylist = ['p', 'y', 't', 'h', 'o', 'n']
print(mylist)
print(id(mylist))
mylist[0]='P'
print(mylist)
print(id(mylist))

mylist = [1,2,3,4,5,6,7,8,9,10]
mylist[0:4]=[100,1000]
print(mylist)

mylist = [0,1,2]
mylist[0:2]='Hi! Python'
print(mylist)

mylist = [0,1,2,3,4,5,6,7,8,9,10]
print(mylist[0::2])
mylist[0::2]='python' #확장분할 연산시에는 객체의 수가 동일해야 함
print(mylist)

# 03-03. 리스트 항목 추가
mylist = [1,2,3,4,5]
print(mylist+[100])
mylist.append(100) #append 메소드를 통해 객체를 추가
print(mylist)

mylist = [1,2,3,4,5]
mylist.extend([100,101,102]) #extend 메소드는 넘겨받은 시퀀스 객체들을 기존 리스트에 개별적으로 추가
print(mylist)
 
mylist = [1,2,3,4,5]
mylist.append([100,101,102]) #기존 리스트에 하나의 객체로 추가
print(mylist)

mylist = [1,2,3,4,5]
mylist += [100,101,102]
print(mylist)
mylist += 'python' #iterable 객체에 대해서만 가능
print(mylist)

mystr='hello'
print(id(mystr))
mystr+='python'
print(id(mystr))
mylist=[1,2,3,4,5]
print(id(mylist))
mylist+=[100,101,102]
print(id(mylist))

mylist=[1,2,3,4,5]
mylist.insert(0, -77)
print(mylist)
mylist.insert(1, 0)
print(mylist)
mylist.insert(-1, 100)
print(mylist)
mylist.insert(len(mylist), 1000)
print(mylist)

mylist = [1,2,3,4,5]
mylist.append([5,6,7])
print(mylist)
mylist = [1,2,3,4,5]
mylist.insert(len(mylist), [5,6,7])
print(mylist)

# 03-04. 리스트 항목 삭제
mylist = ['hi', 'python', 2.7]
del mylist[2] #특정 위치 객체 삭제
print(mylist)

mylist = [1,2,3,4,5]
del mylist[0:4]
print(mylist)

mylist = [1,2,3,4,5]
del mylist[0::2]
print(mylist)

mylist = [1,2,3,4,5]
mylist.remove(3) #특정 객체 삭제(동일값 존재시 가장 처음 값 삭제)
print(mylist)

mylist = [1,2,3,4,5]
print(mylist.pop())
print(mylist.pop())
print(mylist.pop())
print(mylist.pop(0)) #특정 인덱스의 값을 삭제
print(mylist)

# 03-05. 리스트의 참조
my_imtb1 = 'python'
my_imtb2 = my_imtb1
print(id(my_imtb1), id(my_imtb2))
print(my_imtb1 is my_imtb2)

my_list1 = [1,2,3,4,5]
my_list2 = my_list1
print(id(my_list1), id(my_list2))
print(my_list1 is my_list2)

my_imtb2 = 'Hi! python'
print(id(my_imtb1), id(my_imtb2))
print(my_imtb1 is my_imtb2)

my_list2 = my_list2+[77]
print(my_list1)
print(my_list2)
print(id(my_list1), id(my_list2))
print(id(my_list1) is id(my_list2))

my_list1 = [1,2,3,4,5]
my_list2 = my_list1
my_list2.append(77)
print(my_list1)
print(my_list2)
print(my_list1 is my_list2)
print(id(my_list1), id(my_list2))
my_list1[-1] = 100
print(my_list2)
print(id(my_list1), id(my_list2))

my_temper = [11,13,15,10]
print(my_temper)
my_temper.insert(0, 9); del my_temper[-1]
print(my_temper)

mylist1 = [1,2,3,4,5]
mylist2 = copy.copy(mylist1) #copy 모듈에 있는 copy 함수를 사용하여 복사(얕은 복사)
print(id(mylist1), id(mylist2))
mylist2 = copy.deepcopy(mylist1) #깊은 복사
print(id(mylist1), id(mylist2))

shallow_list1 = [[1,2,3,4,5], 'python', 2, 3, 4, 5]
shallow_list2 = copy.copy(shallow_list1) #내부에 포함된 객체가 타입에 상관없이 공유(분할연산과 동작 방식은 같으나 분할연산이 성능적으로 4배 가량 더 좋다)
print(id(shallow_list1[0]) == id(shallow_list2[0]))
print(id(shallow_list1[1]) == id(shallow_list2[1]))
print(id(shallow_list1[2]) == id(shallow_list2[2]))

deep_list1 = [[1,2,3,4,5],'python',2,3,4,5]
deep_list2 = copy.deepcopy(deep_list1) #컨테이너 객체중 mutable 객체만 복사하고, 나머지는 공유
print(id(deep_list1[0]) == id(deep_list2[0]))
print(id(deep_list1[1]) == id(deep_list2[1]))
print(id(deep_list1[2]) == id(deep_list2[2]))

my_temper = [11,13,15,10]
print(my_temper)
my_temper.insert(0,9); del my_temper[-1]
print(my_temper)
my_temper = [11,13,15,10]
print(my_temper)
my_0405_temper = copy.deepcopy(my_temper)
my_temper.insert(0,9); del my_temper[-1]
print(my_temper)
print(my_0405_temper)

my_temper = ['temperatur', [11,13,15,10]]
my_0405_temper = copy.copy(my_temper)
print(my_0405_temper)
my_temper[1].insert(0,9); del my_temper[1][-1]
print(my_0405_temper)

# 03-06. 리스트의 메소드
mylist = [4,3,1,5,6,3,7]
mylist.sort();
print(mylist)
mylist = [4,3,1,5,6,3,7]
mylist2 = sorted(mylist)
print(mylist)
print(mylist2)
print('p/y/t/h/o/n'.split('/'))
print('/'.join(['p','y','t','h','o','n']))

# 04. 튜플
# 04-01. 튜플의 생성
tp = (1,2,3,4)
print(tp)
tp2 = ('hello','python',1,2,3,[4,5,6])
print(tp2)

mytuple = ()
print(type(mytuple))
print(len(mytuple))

mytuple = (1,) #한 개 이상의 튜플 생성시 객체 뒤에 , 를 사용해야 한다
print(mytuple)
print(len(mytuple))

mytuple = (1)
print(type(mytuple))
mytuple2 = (1+2)
print(type(mytuple2))
mytuple3 = (1+2,)
print(type(mytuple3))
print(mytuple3)

# 04-02. 튜플의 연산
mytuple = 'p','y','t','h','o','n'
print(mytuple)
print(mytuple[1])
print(mytuple+mytuple)
print(mytuple*2)
print(mytuple[0:4])
print(mytuple[0::2])
print(mytuple[-1::-1])

# 04-03. 이름 있는 튜플
from collections import namedtuple
bookinfo = namedtuple('struct_bookinfo', ['author', 'title']) #namedtuple 함수를 사용해 struct_bookinfo 클래스의 'author', 'title' 속성을 가진 클래스를 생성
mybook = bookinfo('hyun', 'gop') #struct_bookinfo 클래스에 대한 객체 생성
print(mybook.author)
print(mybook[0])
print(mybook.title)
print(mybook[1])

# 04-04. 시퀀스 타입간의 변환
myString = 'python'
print(list(myString))
print(tuple(myString))
myData = [1,2,3,4,5]
print(tuple(myData))
mylist = ['p','y','t','h','o','n']
print(str(mylist), ''.join(mylist))

mylist = [3.14, 2.22, 0.12]
print(str(mylist)) #리스트->문자열
print(list(str(mylist))) #리스트->문자열->리스트
print(eval(str(mylist))) #리스트->문자열->리스트(eval 함수는 표현식 문자열을 표현식으로 처리)
print(eval('3+3'))

mylist = ['3.14','2.22','0.12']
print(''.join(mylist))
mylist = [3.14, 2.22, 0.12]
print(''.join([str(i) for i in mylist])) #제어문을 사용해 문자열로 변환
print('/'.join([str(i) for i in mylist]))
print('3.14/2.22/0.12'.split('/'))

a,b,c,d,e,f = 'python' #언패킹
print(a,b,c,d,e,f)
t = a,b,c,d,e,f #패킹
print(t)

# 직접해 봅시다.
# 1. 아스키 코드표를 참고하여 이스케이프 시퀀스만으로 문자열 'python!' 을 출력하는 코드를 만들어 보자.
print('\x70\x79\x74\x68\x6f\x6e!')

# 2. 1번 문제의 이스케이프 시퀀스로만 이루어진 문자열에는 어떻게 분할 연산이 이루어지는지 실험해보자.

# 3. 파이썬 문서를 참고하여 문자열의 메소드 replace의 사용법에 대해 알아보자. 그리고 예제 코드를 만들어 보자.
print('python very easy!'.replace('y', 'a', 2))

# 4. 본문에서 리스트의 끝에 항목을 추가하는 방법은 여러 가지가 있었다. 이 중에서 + 연산을 사용하는 방법을 제외하고 리스트가 가진
#    메소드를 사용하는 방법으로 리스트 mylist = [[1,2,3,4,5]]에 객체 77을 추가하여 mylist가 [[1,2,3,4,5,77]]이 되도록 하자.
mylist = [[1,2,3,4,5]]
mylist[0].append(77)
print(mylist)
mylist = [[1,2,3,4,5]]
mylist[0].insert(len(mylist[0]), 77)
print(mylist)

# 5. score는 과목, 점수 쌍의 리스트를 항목으로 갖는 리스트다. 아직 score 리스트는 미완성이지만 다음과 같이 추가해 나가고 있다.
#    math의 점수와 동일한 형식으로 english 점수에 90점을 추가하는 연산을 하여라.
score = [['math',89], ['english']]
score[1].append(90)
print(score)

# 6. 문자열 'python'을 리스트, 튜플 타입으로 변환해보자. 또 반대로도 변환해보자.
tempStr = 'python'
print(list(tempStr), tuple(tempStr)) #문자열 -> 리스트, 튜플
print(''.join(list(tempStr)), ''.join(tuple(tempStr)))

# 7. 변수 a에 1을 대입하고 b에 2를 대입한 후 이 둘을 합하여 출력하는 파이썬 코드를 윈도우 커맨드 창에서 직접 실행해보자.
#  - python -c a=1;b=2;print(a+b)

# 8. 첫째 항의 값이 'python', 둘째 항의 값이 '3.4.1'인 네임드 튜플을 만들어 보자. 이 때 첫째 항의 이름을 name, 둘 째 항의 이름을 version으로 한다.
from collections import namedtuple
versionClass = namedtuple('version_class', ['name', 'version'])
version = versionClass('python', '3.4.1')
print(version.name, version.version)

# 9. 윈도우 커맨드 라인에서 print('\xff')를 직접 실행시켜 보자. 에러의 원인을 파악한 후 에러가 발생하지 않도록 하자.
#  - cp949 코덱에서 지원하지 않는 인코딩 타입으로 코덱을 변경해야 한다.
#  - chcp 65001 로 인코딩 타입을 UTF-8로 변환

# 10. 9번에서 에러가 발생하지는 않았지만 원하는 결과가 출력되지 않을 수도 있다. 왜 그런지 설명해보자.
#  - 기존 인코딩 타입에서 지원하는 문자가 아니어서 원하는 결과가 출력되지 않을 수 있다.

# 11. elephant 는 8개의 문자로 이루어진 문자열이다. 이 문자열의 처음부터 시작하여 끝에서 3번째 문자까지만 분할하여 출력해보자.
print('elephant'[0:-2])

# 12. 확장 분할을 사용하여 11번 문제의 결과를 뒤집어 출력해보자.
print('elephant'[-3::-1])