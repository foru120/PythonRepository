# 06. 매핑 타입
# 06-01. 사전의 생성
poppul_dict = {'China':1367485388, 'India':1251695584, 'Indonesia':321368864, 'America':513949445, 'Brazil':255993674}
fruit = {'apple':'사과', 'banana':'바나나', 'orange':'오렌지'}
animal = {'monkey':'원숭이', 'bear':'곰', 'cat':'고양이'}
print(fruit, animal)
print(animal['bear'])
print(poppul_dict['Brazil'], poppul_dict['China'])

# 06-02. 사전의 특징
print(poppul_dict) #사전을 생성할 당시의 항목의 순서는 지켜지지 않는다
my_dict = {'a':1, 'a':2}
print(my_dict) #동일한 key가 있다면 마지막에 입력된 값만 남는다
a=77
b=[1,2,3]
c='python'
# dict_test = {a:b, b:c, c:a} 리스트는 mutable 객체이므로 key로 올 수 없다
dict_test = {a:b, tuple(b):c, c:a}
print(dict_test)

# 06-03. 사전의 연산
poppul_dict = {'China':1367485388, 'India':1251695584, 'Indonesia':321368864, 'America':513949445, 'Brazil':255993674}
print(poppul_dict['China'])
poppul_dict['China'] = 13674854000 #value 수정
print(poppul_dict['China'])
poppul_dict['Korea'] = 51529338 #항목 추가
print(poppul_dict)
poppul_dict.update({'Korea':51529777, 'Japan':127103388}) #update 메소드를 통해 항목을 추가하거나 수정 가능
print(poppul_dict)

my_dict = {'c':3, 'd':4, 'a':1, 'b':2}
my_dict.setdefault('a', 100) #setdefault 메소드는 기존에 해당 key 가 존재하면 기존 value를 리턴하고, 존재하지 않으면 해당 항목을 추가
print(my_dict)
my_dict.setdefault('e', 100)
print(my_dict)

del poppul_dict['Japan'] #항목의 삭제
print(poppul_dict)

dict_test = {'first':7, 'second':77, 'third':777}
x = {'first':0, 'end':100}
for my_key in x: 
    if my_key not in dict_test: #in 연산을 통해 사전에 해당 key가 존재하는지 체크
        dict_test.update({my_key:x[my_key]})
print(dict_test)
print('fourth' in dict_test)
print('fourth' not in dict_test)
print('first' in dict_test)
print('first' not in dict_test)

print(poppul_dict)
print(list(poppul_dict)) #list 타입으로 변환시 사전의 key 만 변환
poppul_list = [['China',13674854000], ['Indonesia',321368864], ['Korea', 51529777], ['India',1251695584], ['Japan',127103388], ['Brazil',255993674], ['America', 513949445]]
print(dict(poppul_list)) #리스트의 항목들이 두 개의 객체로 구성되어야 함
test = ['ab', 'cd']
print(dict(test))
print(poppul_dict.items()) #items 메소드는 view object 형식으로 리턴

poppul_dict = {'China':1367485388, 'India':1251695584, 'Indonesia':321368864, 'America':513949445, 'Brazil':255993674, 'Korea':51529777, 'Japan':127103388}
print(poppul_dict.items()) #파이썬 2.x 버전에서는 리스트 객체로 리턴(기존의 사전 객체와 별개의 객체)

py3 = my_dict.items()
print(py3)
my_dict.clear()
print(py3) #파이썬 3.x 버전에서는 기존 사전 객체와 공유

dict_1 = {'a':1, 'b':2}
dict_2 = {'b':3, 'c':4}
view_obj1 = dict_1.items()
view_obj2 = dict_2.items()
print(view_obj1 | view_obj2) #합집합
print(view_obj1 & view_obj2) #교집합
print(view_obj1 - view_obj2) #차집합
print(view_obj1 ^ view_obj2) #합집합 - 교집합
print(dict(view_obj1 | view_obj2))

from timeit import timeit
print(timeit('my_dict.items()', "my_dict = {'a':1, 'b':2, 'c':3, 'd':4}"))

print(list(dict_1.keys()), dict_1.values()) #keys(), values() 를 사용해 key 또는 value 만 얻을 수 있다

# 직접해 봅시다
# 1. 자신의 지인의 연락처를 사전 형식으로 저장해보자. 그리고 색인 연산으로 연락처를 검색해보자(이 때 key는 이름, value는 전화번호)
number = {'홍길동':'010-4456-7865', '고길동':'010-5687-9896', '이지매':'010-7898-1234'}
print(number['홍길동'])

# 2. 1번에서 만든 사전에서 몇몇 연락처를 삭제해보고 다시 추가해보자.
del number['홍길동']; del number['고길동']
number.update({'길용현':'010-5008-6734', '손예진':'010-4568-6698'}) #update 또는 setdefault 메소드 사용
print(number)

# 3. 색인 연산으로 1에서 만든 연락처에 있는 이름과 중복되는 이름을 가진 사람의 연락처를 추가로 입력하려고 기존 연락처가 지워지는 문제가 있다.
#    연락처를 추가는 하되 중복되는 경우는 추가가 안 되도록 하려면 어떤 방법을 사용해야 할까? 이 방법으로 직접 중복되는 이름으로 연락처를 추가해보자.
number.setdefault('이지매', '010-5555-5555')
print(number)

# 4. 3번에서 좀 더 개선된 방법으로 동명이인의 경우를 구분하기 위하여 여러 정보를 가진 key값으로 대체해보자(key 값으로 어떤 타입의 객체를 사용해야 할까?)
number = {('홍길동', 21, '대전'):'010-4456-7865', ('홍길동', 25, '서울'):'010-4578-6666'}
print(number)

# 5. 앞서 만든 주소록 사전을 리스트로 변환해보자. 이 때 key와 value가 모두 포함되도록 하자.
number = {'홍길동':'010-4456-7865', '고길동':'010-5687-9896', '이지매':'010-7898-1234'}
number_list = list(number.items())
print(number_list)

# 6. 파이썬 2.x 버전과 파이썬 3.x 버전 각각에서 사전의 keys 메소드와 values 메소드의 성능 비교를 해보자.
#    그리고 성능의 차이가 무엇 때문인지 생각해보자.
from timeit import timeit
print(timeit('number.keys()', "number = {'홍길동':'010-4456-7865', '고길동':'010-5687-9896', '이지매':'010-7898-1234'}"))
#  - 3.x 버전에서는 view 역할만 하고 직접 리스트를 생성하지 않으므로 2.x 버전보다 빠르다.