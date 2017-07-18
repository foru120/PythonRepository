# 08. 제어문
# 08-01. if문
switch = 0
if switch==1:
    print("ON")

switch = 0
if switch==1:
    print("ON")
elif switch==0:
    print("OFF")

number = 99
if number==0 or number==100:
    print('Hello')
elif number<0 or 0<number<100 or 100<number:
    print('good-bye')

if number==0 or number==100:
    print('Hello')
else:
    print('good-bye')

# 08-02. 조건 표현식
switch = 0
print("ON") if switch==1 else print("OFF")
x = 1; y = 77
result = x if x>y else y #if 구문을 표현식으로 사용가능해 변수에 대입할 수 있다
print(result)

button = 2
print('button1') if button==1 else print('button2') if button==2 else print('other button')

# 08-03. for문
my_list = [0,1,2,3,4,5,6,7,8,9,10]
print(sum(my_list))
sum = 0
for x in my_list:
    sum += x
print(sum)

for i in range(0,10):
    print(i)

total = 0
for i in range(0,1001):
    total += i
print(total)

for i in range(0,5):
    print('I love python!')

for x in [1,2,3,4,5,6,7]:
    print(x)

for x in 'python':
    print(x)

for x in ('a','b','c',1,2,3):
    print(x)

my_dict = {'a':1, 'b':2, 'c':3, 'd':4}
for k in my_dict:
    print(k)
for v in my_dict.values():
    print(v)
for k in my_dict:
    print(k, my_dict[k])
for k,v in my_dict.items():
    print(k, v)

for x in range(1,3):
    for y in range(1,4):
        print(x,y)

my_dir1 = ['a.txt','b.dox','c.jpg','d.avi']
my_dir2 = ['e.au','f.kor','g.txt','h.bat']
my_dir3 = ['i.py','j.pyc','k.py']
d = [my_dir1, my_dir2, my_dir3]
for x in d:
    for y in x:
        print(y)

coin_box = [500, 500, 500, 50, 10, 100, 100, 10, 100, 50]
for c in coin_box:
    if c==100:
        print('100원짜리 동전')
        break
    else:
        print('100원짜리 동전 아님')

for c in coin_box:
    if c!=100:
        continue
    else:
        print('100원 있음')
        break

for s in 'python':
    if s=='o':
        break
    print(s)
else: #for 문에서 else문은 반드시 실행되지만 break 문을 사용해서 건너뛸수도 있다
    print('end')

# 08-04. 리스트 생성 표현
a=1
b=2
c=3
temp_list = [a,b,c,a*b,(a,b)]
print(temp_list)
print([x*2 for x in range(1,10)]) #리스트 생성과 for문이 결합된 리스트 내포
my_list=[]
for x in range(1,10):
    my_list.append(x)
print(my_list)

print([x for x in range(1,10) if x%2==0])
info = [1,2,8,22,3,5,20,6,99,22,76]
print(['짝' if x%2==0 else '홀' for x in info])
temp_list = [(x,y) for x in range(1,10) for y in range(1,10)]
print(temp_list)

# 08-05. while 문
count=5
while count:
    print(count)
    count-=1
for x in range(5,0,-1):
    print(x)

# 08-06. 기타 문법
set_comp = {x for x in range(1,10)}
print(set_comp)
dict_comp = {x:x**2 for x in range(1,10)}
print(dict_comp)
tuple_comp = (x for x in range(1,10))
print(tuple_comp)

# 직접해봅시다
# 01. mylist = [1,2,3,4,5,6,7,8,9,10,'python'], mylist를 for문으로 순회하면서 각 항목을 검사하여 짝수면 '짝수', 홀수면 '홀수'를 출력하는 프로그램을 작성하자.
#     이 때 입력받은 데이터가 숫자가 아니라면 '숫자 아님'이라고 출력하자.
mylist = [1,2,3,4,5,6,7,8,9,10,'python']
for x in mylist:
    if type(x)==type(1):
        if x%2==0:
            print('짝수')
        else:
            print('홀수')
    else:
        print('숫자 아님')

for i in mylist:
    if type(i) != int:
        print(i, ': 숫자아님')
    elif i%2==0:
        print(i, ': 짝수')
    else:
        print(i, ': 홀수')

# 02. 1번에서 만든 코드를 조건 표현식을 사용하는 코드로 바꿔보자.
print([('짝수' if x%2==0 else '홀수') if type(x)==type(1) else '숫자 아님' for x in mylist])

# 03. 1부터 1000까지의 홀수의 합을 구하는 방법을 연구해보자.
sum=0
for x in range(1,1001):
    if x%2!=0:
        sum+=x
else:
    print(sum)

# 04. 문자열 'python'의 문자들을 하나씩 출력하되 대문자로 변환하여 출력해보자.
#     단, 문자가 'y'인 경우에는 소문자 그대로 출력하도록 한다.
for x in 'python':
    if x=='y':
        print(x)
    else:
        print(x.upper())

# 05. 사전{'a':1,'b':2,'c':3,'d':4}의 key와 value를 다음과 같이 출력해보자.
dict_list = [x for x in {'a':1,'b':2,'c':3,'d':4}.items()]
for x in dict_list:
    print(dict([x,]))

for k,v in {'a':1,'b':2,'c':3,'d':4}.items():
    print('{',k,':',v,'}')

# 06. 본문의 '심화된 중첩 for문'의 예제의 출력 결과는 파일이 어떤 디렉토리에 속하는지에 대한 구분이 되지 않는다는 것이다.
#     eval 함수를 이용하여 탐색기 형식처럼 디렉토리 구조를 파악할 수 있도록 출력해보자.
my_dir1 = ['a.txt','b.dox','c.jpg','d.avi']
my_dir2 = ['e.au','f.kor','g.txt','h.bat']
my_dir3 = ['i.py','j.pyc','k.py']
dir = ['my_dir1','my_dir2','my_dir3']
for d in dir:
    print(d)
    for f in eval(d): #eval 메소드는 사용하면 표현식을 사용해서 처리 가능
        print(f)

# 07. 어떤 유명한 학원에 입학하기 위해서는 배치고사를 90점 이상 맞아야 한다고 한다.
#     점수가 80점 이상 90점 미만이면 재시험의 기회가 주어진다. 하지만 80점 미만이라면 점수에 상관없이 입학을 할 수가 없다.
#     다음에 학생이름과 배치고사 점수가 주어져 있다. 세 개의 리스트에 입학된 학생, 재시험을 치를 학생, 입학을 못하는 학생별로 각각 저장하는 프로그램을 작성하여라.
list1 = []; list2 = []; list3 = []
for x in [80,96,82,90,74,79,81]:
    if x>=90:
        list1.append(x)
    elif 80<=x<90:
        list2.append(x)
    else:
        list3.append(x)
print(list1, list2, list3)

# 08. 원의 반지름의 길이를 입력하면 원의 둘레와 넓이를 계산해주는 코드를 작성해보자.
#     단 반지름의 길이 대신 'end'를 입력하면 프로그램이 종료되도록 한다.(원주율은 3.14로 한다.)
# r=0
# while True:
#     r=input()
#     if r!='end':
#         r=int(r)
#         print('원의 둘레 :',2*3.14*r,', 원의 넓이 :', 3.14*r**2)
#     else:
#         break

# 09. 리스트 내포를 사용하여 1부터 100까지 숫자 중 2와 3의 공배수 중 4의 배수가 아닌 수들의 리스트를 만들어보자.
print([x for x in range(1,101) if x%2==0 and x%3==0 and x%4!=0])        

# 10. range(1,20)을 순회하여 3으로 나누었을 때 나머지가 1이면 'A'로 2면 'B'로 0이면 'C'로 바꾸어서 리스트로 만들어라.
#     이 때 리스트 내포를 사용하여 코드를 작성하자.
print(['A' if x%3==1 else 'B' if x%3==2 else 'C' for x in range(1,20)])

# 11. 리스트 내포를 이용하여 구구단을 출력하는 코드를 만들어 보자.
gugudan = [(x,y) for x in range(1,10) for y in range(1,10)]
for x in gugudan:
    print(x[0],'*',x[1],'=',x[0]*x[1])

print([str(x) + '*' + str(y) + '=' + str(x*y) for x in range(1,10) for y in range(1,10)])

# 12. 1부터 1000까지의 합을 while 문으로 작성해보자(반복자를 사용하는 방법으로도 이 문제를 해결해보자.)
sum=0
cnt=1
while cnt<=1000:
    sum+=cnt
    cnt+=1
print(sum)

# total=0
# it = iter(range(0,1001))
# while it:
#     total += it.__next__()

# 13. 테트리스라는 게임은 도형을 끼워 맞춰서 빈칸 없이 줄을 채우면 해당 줄이 삭제되는 게임이다.
#     예를들어 다음과 같은 리스트가 있다고 가정해보자.
#     screen_db=[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,1,1],[0,1,1,1,1,1,1,1,0],[1,1,1,1,1,1,1,1,1],[1,0,1,1,1,1,0,1,0],[0,1,1,1,1,1,1,1,0,1]]
#     이 리스트는 테트리스에서 화면에 채워진 도형을 나타낸다. 1은 채워진 칸, 0은 빈칸을 뜻하고 위 리스트를 화면에 출력하면 다음과 같다.
# 13-01. screen_db를 화면에 출력하는 코드를 만들어 보자.
screen_db=[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,1,1],[0,1,1,1,1,1,1,1,0],[1,1,1,1,1,1,1,1,1],[1,0,1,1,1,1,0,1,0],[0,1,1,1,1,1,1,0,1]]
for j in screen_db:
    print('')
    for i in j:
        print(i, end='')
# 13-02. 이 때 for문을 사용하여 1로 꽉 채워진 행을 지우는 코드를 만들어 보도록 하자.
for j in range(0, len(screen_db)):
    for i in screen_db[j]:
        if i != 1:
            break
    else:
        del(screen_db[j])
        screen_db.insert(0, [0,0,0,0,0,0,0,0,0])
print('')

for i in screen_db:
    print('')
    for j in i:
        print(j, end='')
print('')        