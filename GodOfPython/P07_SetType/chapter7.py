# 07. 셋 타입
my_card = [1,10,11,37,58,72,91,99]
my_card.extend([7,55,99])
my_card.sort()
print(my_card)

# 07-01. set의 생성
my_set = {2,4,6,8,10}
print(my_set)

# 07-02. set의 특징
print({1,1,2,3,4,5,6,7,8,9,9,9}) #중복 제거, hashable 객체

# 07-03. set의 연산
a = {1,2,3,4,5}
b = {4,5,6,7,8,9}
print(a|b, a&b, a-b, a^b)

my_card = {1,10,11,37,58,72,91,99}
my_card.add(7)
my_card.add(55)
my_card.add(99) #add 메소드는 단일 항목 추가
print(my_card)
print(my_card.union([8,76,100])) #union 메소드는 iterable 객체 단위로 추가(새로운 set 객체 생성)
print(my_card)

my_set = {1,2,3,4}
my_set.remove(4) #색인 연산이 불가능하므로 del 연산자 사용 불가
print(my_set)
my_set.discard(4) #discard 메소드는 지우고자 하는 항목이 없어도 에러 발생하지 않음
print(my_set)

print(1 in {1,2,3,4}, 5 in {1,2,3,4})

# 07-04. 타입 변환
print(my_card)
print(sorted(my_card))
card_list = list(my_card)
card_list.sort()
print(card_list)

my_card = [1,10,11,37,37,37,58,72,72,91,99]
card_set = set(my_card)
print(card_set)
my_card = list(card_set)
my_card.sort()
print(my_card)

my_set = {1,3,5,7,9}
my_tuple = tuple(my_set)
print(my_tuple)
my_set2 = set(my_tuple)
print(my_set2)

# 직접해봅시다
# 1. 아래 밴다이어그램에서 a영역과 b영역에 들어갈 숫자들을 파이썬의 set 타입의 연산을 이용하여 구하라.
x = {1,2,3,4,5,6,8}
y = {4,5,6,9,10,11}
z = {4,6,8,9,7,10,12}
a = x&y&z
b = y&z-x&y
print(a, b)

# 2. 위 문제에서 제시된 집합에서 x와 y의 교집합 중 z에는 없는 원소를 파이썬 연산을 사용하여 구하여라.
#    그리고 x와 y집합 각각에서 이 공통 원소를 제거해보도록 하자.
a = x&y-z
x = x-a
y = y-a
print(x, y)

# 3. 다음 리스트에서 중복된 항목을 없애보자.
temp_list = [1,2,3,4,5,6,7,8,9,10,2,4,7,9,1]
temp_set = set(temp_list)
print(temp_set)

# 4. 다음 리스트에서 중복된 항목을 없앤 후 오름차순으로 정렬해보라.
temp_list = [9,5,3,7,2,1,2,3,9,6,7,6,7,4,1]
temp_set = set(temp_list)
temp_list = list(temp_set)
temp_list.sort()
print(temp_list)

# 5. 파이썬을 이용하여 다음 세 문자열에서 공통된 문자만 출력해보자.(set 타입의 교집합을 이용하도록 한다.)
a = set('python is simple')
b = set('apple is delicious')
c = set('programming')
print(a & b & c)