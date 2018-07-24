# 09. 함수의 정의
def func_name():
    print('call func')

func_name()

# 09-01. 인수의 전달 방식
def progression(n, step=1): #함수 생성시 step parameter 에 대해 기본값이 부여되었으므로, 호출시 해당 parameter에 대해 생략 가능
    x=1
    while x <= n:
        print(x)
        x += step

progression(10)

def kwfunc(year, month, day):
    print('오늘은 %d년 %d월 %d일 입니다.' %(year, month, day))

kwfunc(2015, 8, 15) #위치 인수 사용
kwfunc(day=15, year=2015, month=8) #키워드 인수 사용 

def argsfunc(*args): #튜플형 인수
    i=0
    for x in args:
        i+=1
    print('인수의 개수 : %d' %i)
    print(args)
argsfunc(1,2,(3,4,5))
argsfunc(1,[7,55],'test',{'a':1,'b':100})
a=1; b=(3,6,9); c={'x':0, 'y':99}
argsfunc(a,b,c)
argsfunc(*b)

def dictsfunc(**dicts): #사전형 인수
    i=0
    for x in dicts.keys():
        i+=1
    print('인수의 개수 : %d' %i)
    print(dicts)
dictsfunc(a=1,b=2,c=3)
dictsfunc(**{'a':1,'b':2,'c':3}) #사전과 사전을 매핑

def recomandable(a, *args, **dicts): #단순한 형태 -> 복잡한 형태
    print(a, args, dicts)

# 09-02. return
def rtTest():
    x=1
print(rtTest()) #파이썬에서 함수는 기본적으로 None 을 리턴

def returnTest(a,b,c,d):
    print(a+b)
    return b+c
    print(c+d)
print(returnTest(1,2,3,4))

def returnTuple(x,y,z):
    return x,y,z
print(returnTuple(1,2,3))

global_namespace = locals()
print(global_namespace)
print(global_namespace['returnTest'])

locals()['myvar']=100
print(myvar)

print(dir())
print(globals())
print(dir(__builtins__))

# 09-03. 스코핑룰
global_var=77
def function():
    print(global_var)
function()

global_var=77
def myfunc():
    global_var+=1 #함수내에서 전역변수를 사용하고 싶다면 global 키워드로 전역변수를 명시해야함. global 키워드로 명시하지 않을 시 전역변수가 좌변값이 되어서는 안됨.
    print(global_var)
#myfunc()

global_var=77
def myfunc1():
    global global_var #global 키워드를 이용해 전역변수를 명시
    global_var+=1
    print(global_var)
myfunc1()

# 09-04. 중첩 함수
def outter():
    def inner():
        print('inner')
    inner()
outter()

def outter():
    a=1
    def inner1():
        b=2
        def inner2():
            c=3
            print(a,b,c,)
        inner2()
    inner1()
outter()

global_var=77
def outter():
    global_var=100
    def inner():
        global global_var #함수의 중첩내에서도 global 키워드를 통해 전역변수 사용
        global_var+=1
        print(global_var)
    inner()
outter()        

global_var=77
def outter():
    global_var=100
    def inner():
        nonlocal global_var #nonlocal 키워드를 사용해 outter 함수의 지역변수를 사용
        global_var+=1
        print(global_var)
    inner()
outter()

# 09-05. 인수전달 vs global
outData=77
def func(param):
    param = param+23 #지역변수 param 생성
    print(param)
func(outData)
print(outData)

outData=77
def func():
    global outData
    outData = outData+23 #전역변수 outData 사용
    print(outData)
func()
print(outData)

# 09-06. 람다 표현식
print((lambda x:x**2)(3))
lambda_func = lambda x:x**2
print(lambda_func(3))

mylist = [9,1,7,3,4,2,5,6,8]
mylist.sort(key=lambda x:x)
print(mylist)
mylist.sort(key=lambda x:-x)
print(mylist)
mylist.sort(key=lambda x:x%3)
print(mylist)
mylist.sort(key=lambda x:x%2)
print(mylist)

def func(string, option=0):
    if option:
        s=string.upper()
        print(s)
    else:
        s=string.lower()
        print(s)

def func(string, option=0):
    print(option and (lambda s:s.upper())(string) or (lambda s:s.lower())(string)) #and, or 연산 시 뒤의 연산 값이 도출 됨
func('kil yong hyun', 1)

def func():
    pass
n=3
def func():
    if n:
        total=n
    print(total)
func()

total=0
def mysum():
    global total
    total+=1
    print(total)
mysum()
mysum()

var=0
def outter():
    var=77
    def inner():
        nonlocal var
        var+=1
        print(var)
    return inner
clsr1 = outter()
clsr2 = outter()

def outter():
    total=0
    def inner():
        nonlocal total
        total+=1
        print(total)
    return inner
mysum=outter()
mysum()

# 09-07. 장식자
import datetime
def print_hi():
    print('hello python')
print_hi()
def print_hi():
    print('python v 3.5.1')
    print('hello python')
def deco1(func): #인자로 호출하고자 하는 함수를 받아서 수행
    def new_func():
        print('Today', datetime.date.today())
        func()
    return new_func
print_hi1=deco1(print_hi)
print_hi1()

def mydeco(func):
    total=0
    def new_func():
        nonlocal total
        total+=1
        print(total,'번 호출')
        func()
    return new_func
def print_hi():
    print('Hello Python')
func1=mydeco(print_hi)
func2=mydeco(print_hi)
func1()
func1()
func2()

import datetime
def deco1(func):
    def new_func():
        print('Today', datetime.date.today())
        func()
    return new_func
@deco1 #deco1 장식자로 다음 함수를 장식
def print_hi():
    print('Hello python')
print_hi()
@deco1
def print_easy():
    print('Python is easy')
print_easy()
print('')

import datetime
def deco1(func):
    def new_func():
        print('Today', datetime.date.today())
        func()
    return new_func
def deco2(func):
    def new_func():
        print('Python ver 3.5.1')
        func()
    return new_func
@deco1
@deco2
def print_hi():
    print('Hello Python')
print_hi()

import datetime
def deco(func):
    def new_func(name, age):
        print('Today', datetime.date.today())
        func(name, age)
    return new_func
@deco
def print_hi(name, age):
    print('이름:',name,'나이:',age)
print_hi('철수',19)

# 09-08. 제너레이터 함수
def my_gen():
    n=0
    while n<=10:
        yield n #yield 키워드가 사용되면 이 함수는 제너레이터 함수가 됨
        n+=1
a=my_gen()
print(type(a))
print(a.__next__())
print(a.__next__())
print(a.__next__())
#yield는 return처럼 값을 반환하지만 함수를 종료시키지는 않고, __next__ 메소드가 호출될 때까지 잠시 멈춘다.

# 09-09. 코루틴 함수
def co_routine():
    total=0
    while True:
        n=(yield) #(yield) 표현식은 외부로부터 값을 받을수 있다
        total+=n
        print('total =', total)
a=co_routine()
a.__next__()
a.send(1)
a.send(1)
a.send(3)
print(type(a))

def co_routine():
    total=0
    while True:
        n=(yield total) #(yield total) 처럼 외부로부터 값을 받는것과 값을 반환할 수 있다
        total+=n
a=co_routine()
print(a.__next__())
print(a.send(10))

# import time
# def coroutineA():
#     n=0
#     while True:
#         n=(yield n)
#         time.sleep(1)
#         if n%10==3 or n%10==6 or n%10==9:
#             print('A : nothing')
#         else:
#             print('A :', n)
#         n+=1
# n=0
# A=coroutineA()
# A.__next__()
# while True:
#     n=A.send(n)
#     time.sleep(1)
#     if n%10==3 or n%10==6 or n%10==9:
#         print('B : nothing')
#     else:
#         print('B :', n)
#     n+=1

# 직접해봅시다
# 01. 알파벳 문자열을 인수로 넣으면 문자열을 모두 대문자로 바꾸는 함수를 만들어보자.
temp_string='ai4kkkjd'
def string_to_upper(string):
    return string.upper()
temp_string=string_to_upper(temp_string)
print(temp_string)

# 02. 함수 func는 한 개 또는 두 개의 숫자 인수를 받을 수 있다. 하나의 수를 받으면 해당 숫자에 10을 곱해서 반환해준다.
#     단 두 개의 숫자를 인수로 받으면 첫 번째 인수에서 두 번째 인수를 곱해서 반환해준다. func 함수를 설계해보자.
def func(a,b=10):
    return a*b
print(func(1), func(1,3))

# 03. 다음에 정의된 함수 myfunc는 숫자들을 인수로 받아서 10배를 한 후 모두 더한 값을 출력해주는 함수다.
def myfunc(multiple, *numbers):
    total=0
    for i in numbers:
        total+=i*multiple
    print(total)
myfunc(2,3,4,5,2,65,6)

# 04. 두 수를 인수로 받아서 두 수의 합과 차를 동시에 반환하는 함수를 만들어 보자.
def func(a,b):
    sum=a+b
    minus=a-b
    return sum,minus
print(func(5,3))

# 05. 아래 정의된 함수 func를 호출하면 예외가 발생한다.
g=99
def func():
    global g
    g+=1
func()    
print(g)

# 06. 두 전역변수 a,b는 각각 10과 20의 값을 가지고 있다. func 함수를 호출했을 때 전역변수 a와 b의 값이 서로 바뀌도록 해보자.
a=10; b=20
def func():
    global a, b
    temp=a
    a=b
    b=temp
# 답
def func():
    global a,b
    a,b=b,a

func()
print(a, b)
func()
print(a, b)
func()
print(a, b)

# 07. 6번에서 만든 함수가 호출될 때 아무런 것도 안 하는 것처럼 보인다. 따라서 이 함수가 호출될 때 'a와 b의 값이 교환 되었습니다.'라는
#     메시지가 출력되도록 만들어보자. 단 이 함수의 내부를 알 수 없는 상태라 가정하여 재정의가 힘든 상황이라고 한다.
def print_chg(func):
    def inner():
        func()
        print('a와 b의 값이 교환 되었습니다.')
    return inner #장식자는 리턴되는 함수가 있어야 함

@print_chg
def func_chg():
    global a, b
    temp=a
    a=b
    b=temp

func_chg()
print(a, b)

# 08. range 객체와 동일한 기능을 하는 제너레이터 객체를 만들고 이를 이용하여 구구단을 출력하는 함수를 만들어보자.
def func():
    n=1
    while n<=9:
        yield n
        n+=1

def gugudan():        
    for x in func():
        for y in func():            
            print(x,'*',y,'=',x*y)
            
# 답
def mygen(a, b, step=1):
    n = a
    if step > 0:
        while n < b:
            yield n
            n += step
    elif step < 0:
        while n > b:
            yield n
            n += step

def gugudan():
    for i in mygen(1,10):
        print(i, '단')
        for j in mygen(1,10):
            print(i,'*',j,'=',i*j)

gugudan()
# 09. 8번에서 만든 함수에 장식자를 이용해서 구구단의 출력 전에 구구단의 시작을 알리는 문구를 넣고 구구단의 끝에는 끝을 알리는 문구를 넣어라.
def deco(func):
    def inner():
        print('구구단을 시작하겠습니다.')
        func()
        print('구구단이 끝났습니다.')
    return inner

@deco
def gugudan():        
    a=func()
    while True:
        x=a.__next__()
        b=func()
        while True:
            y=b.__next__()
            print(x,'*',y,'=',x*y)
            if y==9:
                break
        if x==9:
            break

gugudan()

# 10. 게임을 만드는 데 메인 루틴에서 주인공에 관련된 처리를 하는 코드를 작성하고 코루틴은 적에 관련된 코드를 작성하려고 한다.
#     현재 구현을 하려는 내용은 주인공이 적에게 데미지를 주면 적의 남은 에너지가 출력되도록 하는 것이다. 이 코드를 구현하여라.
def coroutine(x):
    hp=x
    while hp>0:
        damage=(yield hp)
        hp-=damage
    print('적이 죽었습니다.')
    return 0
enemy=coroutine(100)
enemy.__next__()

while True:
    data = input('attack>>>')
    recv = enemy.send(int(data))

    if not recv:
        print('전투종료')
        break
    print('적의 남은 HP : ', recv)