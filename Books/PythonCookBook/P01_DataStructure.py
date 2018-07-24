# Chapter 1. 자료구조와 알고리즘
#  1.1 시퀀스를 개별 변수로 나누기
#  ▣ 문제 : N개의 요소를 가진 튜플이나 시퀀스가 있다. 이를 변수 N개로 나누어야 한다.
#  ▣ 해결 : 모든 시퀀스는 간단한 할당문을 사용해서 개별 변수로 나눌 수 있다.
data = ['ACME', 50, 91.1, (2012, 12, 21)]
name, shares, price, date = data
print(name, shares, price, date)
name, shares, price, (year, mon, day) = data
print(name, shares, price, year, mon, day)

s = 'Hello'
a, b, c, d, e = s
print(a, b, c, d, e)

#  ▣ 토론 : 언패킹은 사실 튜플이나 리스트 뿐만 아니라 순환 가능한 모든 객체에 적용할 수 있다.
#   - 언패킹 시 특정 값 무시(_)
data = ['ACME', 50, 91.1, (2012, 12, 21)]
_, shares, price, _ = data
print(shares, price)


#  1.2 임의 순환체의 요소 나누기
#  ▣ 문제 : 순환체를 언패킹하려는데 요소가 N개 이상 포함되어 "값이 너무 많습니다"라는 예외가 발생한다.
#  ▣ 해결 : 이 문제 해결을 위해 "별 표현식"을 사용한다.
def drop_first_last(grades):
    first, *middle, last = grades
    return avg(middle)

def avg(data):
    return sum(data)/len(data)

print(drop_first_last([1,2,3,4,5,6,7,8,9,10]))

record = ('Dave', 'dave@example.com', '773-555-1212', '847-555-1212')
name, email, *phone_numbers = record
print(name, email, phone_numbers)

*trailing_qtrs, current_qtr = [10, 8, 7, 1, 9, 5, 10, 3]
trailing_avg = sum(trailing_qtrs) / len(trailing_qtrs)

def avg_comparison(trailing_avg, current_qtr):
    return trailing_avg - current_qtr

print(avg_comparison(trailing_avg, current_qtr))

#  ▣ 토론 : 때때로 순환체에 들어 있는 패턴이나 구조를 가지고 있는데, 이럴 때도 별표 구문을 사용하면 개발자의 수고를 많이 덜어준다.
records = [('foo', 1, 2), ('bar', 'hello'), ('foo', 3, 4)]

def do_foo(x, y):
    print('foo', x, y)

def do_bar(s):
    print('bar', s)

for tag, *args in records:
    if tag == 'foo':
        do_foo(*args)
    elif tag == 'bar':
        do_bar(*args)

line = 'nobody:*:-2:-2:Unprivileged User:/var/empty:/usr/bin/false'
uname, *fields, homedir, sh = line.split(':')
print(uname, fields, homedir, sh)

record = ('ACME', 50, 123.45, (12, 18, 2012))
name, *_, (*_, year) = record
print(name, year)

items = [1, 10, 7, 4, 5, 9]
head, *tail = items
print(head, tail)

def sum(items):
    head, *tail = items
    return (head + sum(tail)) if tail else head  # list 개수가 0 개면 false 를 리턴

print(sum(items))


#  1.3 마지막 N개 아이템 유지
#  ▣ 문제 : 순환이나 프로세싱 중 마지막으로 발견한 N개의 아이템을 유지하고 싶다.
#  ▣ 해결 : collections.deque 를 사용한다.
from collections import deque

def search(lines, pattern, history=5):
    previous_lines = deque(maxlen=history)
    for line in lines:
        if pattern in line:
            yield line, previous_lines
        previous_lines.append(line)

# 파일 사용 예
if __name__ == '__main__':
    with open('files\\somefile.txt') as f:
        for line, prevlines in search(f, 'python', 5):
            for pline in prevlines:
                print(pline, end='')
            print(line, end='')
            print('-'*20)

#  ▣ 토론 : 아이템을 찾는 코드를 작성할 때, 주로 yield 를 포함한 제너레이터 함수를 만들곤 한다.
#           이렇게 하면 검색 과정과 결과를 사용하는 코드를 분리할 수 있다.
q = deque(maxlen=3)
q.append(1)
q.append(2)
q.append(3)
print(q)
q.append(4)
print(q)

q.append(5)
print(q)

q = deque()
q.append(1)
q.append(2)
q.append(3)
print(q)
q.appendleft(4)
print(q)
q.pop()
print(q)
q.popleft()
print(q)


#  1.4 N 아이템의 최대 혹은 최소값 찾기
#  ▣ 문제 : 컬렉션 내부에서 가장 크거나 작은 N개의 아이템을 찾아야 한다.
#  ▣ 해결 : heapq 모듈에는 이 용도에 적합한 nlargest() 와 nsmallest() 두 함수가 있다.
import heapq
nums = [1, 8, 2, 23, 7, -4, 18, 23, 42, 37, 2]
print(heapq.nlargest(3, nums))
print(heapq.nsmallest(3, nums))

portfolio = [{'name': 'IBM', 'shares': 100, 'price': 91.1},
             {'name': 'AAPL', 'shares': 50, 'price': 543.22},
             {'name': 'FB', 'shares': 200, 'price': 21.09},
             {'name': 'HPQ', 'shares': 35, 'price': 31.75},
             {'name': 'YHOO', 'shares': 45, 'price': 16.35},
             {'name': 'ACME', 'shares': 75, 'price': 115.65}]
cheap = heapq.nsmallest(3, portfolio, key=lambda x: x['price'])
expensive = heapq.nlargest(3, portfolio, key=lambda x: x['price'])
print(cheap, expensive)

#  ▣ 토론 : 가장 작거나 큰 N개의 아이템을 찾고 있고 N이 컬렉션 전체 크기보다 작다면 앞에 나온 함수가 더 나은 성능을 제공한다.
#   ※ 힙의 가장 중요한 기능은 바로 heap[0]이 가장 작은 아이템이 된다는 것.
nums = [1, 8, 2, 23, 7, -4, 18, 23, 42, 37, 2]
import heapq
heap = list(nums)
heapq.heapify(heap)
print(heap)
print(heapq.heappop(heap), heap)
print(heapq.heappop(heap), heap)
print(heapq.heappop(heap), heap)


#  1.5 우선 순위 큐 구현
#  ▣ 문제 : 주어진 우선 순위에 따라 아이템을 정렬하는 큐를 구현하고 항상 우선 순위가 가장 높은 아이템을 먼저 팝하도록 만들어야 한다.
#  ▣ 해결 : 다음에 나온 코드에서 heapq 모듈을 사용해 간단한 우선 순위 큐를 구현한다.
import heapq

class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        heapq.heappush(self._queue, (-priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]

class Item:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return 'Item({!r})'.format(self.name)

q = PriorityQueue()
q.push(Item('foo'), 1)
q.push(Item('bar'), 5)
q.push(Item('spam'), 4)
q.push(Item('grok'), 1)
print(q.pop())

#  ▣ 토론 : heapq.heappush() 와 heapq.heappop() 은 list_queue 의 첫 번째 아이템이 가장 작은 우선 순위를 가진 것처럼 아이템을
#            삽입하거나 제거한다.
a = Item('foo')
b = Item('bar')
print(a < b)

a = (1, Item('foo'))
b = (5, Item('bar'))
print(a < b)

c = (1, Item('grok'))
print(a < c)


#  1.6 딕셔너리의 키를 여러 값에 매핑하기
#  ▣ 문제 : 딕셔너리의 키를 하나 이상의 값에 매핑하고 싶다.
#  ▣ 해결 : 키에 여러 값을 매핑하려면, 그 여러 값을 리스트나 세트와 같은 컨테이너에 따로 저장해 두어야 한다.
d = {'a': [1, 2, 3], 'b': [4, 5]}
e = {'a': {1, 2, 3}, 'b': {4, 5}}

from collections import defaultdict

d = defaultdict(list)
d['a'].append(1)
d['a'].append(2)
d['b'].append(4)
print(d)

d = defaultdict(set)
d['a'].add(1)
d['a'].add(2)
d['b'].add(4)
print(d['c'])  # 딕셔너리에 존재하지 않는 값이라도 한 번이라도 접근했던 키의 엔트리를 자동으로 생성한다.

#  ▣ 토론 : 여러 값을 가지는 딕셔너리를 만드는 것은 복잡하지 않지만, 첫 번째 값에 대한 초기화를 스스로 하려면 복잡한 과정을 거쳐야 한다.
pairs = {'a': 1, 'b': 2}
d = {}
for key, value in pairs:
    if key not in d:
        d[key] = []
    d[key].append(value)

d = defaultdict(list)
for key, value in pairs:
    d[key].append(value)


#  1.7 딕셔너리 순서 유지
#  ▣ 문제 : 딕셔너리를 만들고, 순환이나 직렬화할 때 순서를 조절하고 싶다.
#  ▣ 해결 : 딕셔너리 내부 아이템의 순서를 조절하려면 collections 모듈의 OrderedDict 를 사용한다.
from collections import OrderedDict

d = OrderedDict()
d['foo'] = 1
d['bar'] = 2
d['spam'] = 3
d['grok'] = 4

for key in d:
    print(key, d[key])

import json
print(json.dumps(d))

#  ▣ 토론 : OrderedDict는 내부적으로 더블 링크드 리스트로 삽입 순서와 관련 있는 키를 기억한다.
#            더블 링크드 리스트를 사용하기 때문에 OrderedDict의 크기는 일반적인 딕셔너리에 비해서 두 배로 크다.


#  1.8 딕셔너리 계산
#  ▣ 문제 : 딕셔너리 데이터에 여러 계산을 수행하고 싶다.(최소값, 최대값, 정렬 등)
#  ▣ 해결 : zip() 과 sorted() 를 함께 사용한다.
prices = {'ACME': 45.23,
          'AAPL': 612.78,
          'IBM': 205.55,
          'HPQ': 37.20,
          'FB': 10.75}

min_price = min(zip(prices.values(), prices.keys()))  # zip() : 한번 사용가능한 이터레이터 생성한다.
print(min_price)
max_price = max(zip(prices.values(), prices.keys()))
print(max_price)

prices_and_names = zip(prices.values(), prices.keys())
print(max(prices_and_names))
print(min(prices_and_names))

#  ▣ 토론 : 딕셔너리에서 일반적인 데이터 축소를 시도하면, 오직 키에 대해서만 작업이 이루어진다.
print(min(prices))
print(max(prices))

print(min(prices.values()))
print(max(prices.values()))

print(min(prices, key=lambda v: prices[v]), prices[min(prices, key=lambda v: prices[v])])
print(max(prices, key=lambda v: prices[v]))

prices = {'AAA': 45.23, 'ZZZ': 45.23}
print(min(zip(prices.values(), prices.keys())))
print(max(zip(prices.values(), prices.keys())))


#  1.9 두 딕셔너리의 유사점 찾기
#  ▣ 문제 : 두 딕셔너리가 있고 여기서 유사점을 찾고 싶다.(동일한 키, 동일한 값 등)
#  ▣ 해결 : keys() 와 items() 메소드에 집합 연산을 수행 한다.
a = {'x': 1, 'y': 2, 'z': 3}
b = {'w': 10, 'x': 11, 'y': 2}

#   - 동일한 키 찾기
print(a.keys() & b.keys())

#   - a 에만 있고 b 에는 없는 키 찾기
print(a.keys() - b.keys())

#   - 동일한 키, 값 찾기
print(a.items() & b.items())

#   - 특정 키를 제거한 새로운 딕셔너리 만들기
c = {key: a[key] for key in a.keys() - {'z', 'w'}}
print(c, type(c))

#  ▣ 토론 : keys(), items() 메소드는 각각 키-뷰 객체와 아이템-뷰 객체를 리턴하는데 해당 객체는 집합 연산이 가능하다.
#            values() 메소드는 값-뷰를 리턴하지만 값-뷰는 유일하다는 보장이 없기 때문에 집합 연산이 불가능하다.
#            따라서, values() 메소드를 통해 나온 값-뷰는 집합으로 변환한 후에야 집합 연산이 가능하다.


#  1.10 순서를 깨지 않고 시퀀스의 중복 없애기
#  ▣ 문제 : 시퀀스에서 중복된 값을 없애고 싶지만, 아이템의 순서는 유지하고 싶다.
#  ▣ 해결 : 시퀀스의 값이 해시 가능하다면 이 문제는 set 과 제너레이터를 사용해서 쉽게 해결할 수 있다.
#   - 해시 가능한 타입
def dedupe(items):
    seen = set()
    for item in items:
        if item not in seen:
            yield item
            seen.add(item)

a = [1, 5, 2, 1, 9, 1, 5, 10]
print(list(dedupe(a)))

#   - 해시 불가능한 타입
def dedupe(items, key=None):
    seen = set()
    for item in items:
        val = item if key is None else key(item)
        if val not in seen:
            yield val
            seen.add(val)

a = [{'x': 1, 'y': 2}, {'x': 1, 'y': 3}, {'x': 1, 'y': 2}, {'x': 2, 'y': 4}]
print(list(dedupe(a, lambda d: (d['x'], d['y']))))

#  ▣ 토론 : 중복을 만드려면 대개 set 을 만드는 것이 가장 쉽지만, 해당 방법은 순서가 훼손된다.
#            따라서, 제너레이터 함수를 사용하는 것이 순서도 유지되고 좋다.


#  1.11 슬라이스 이름 붙이기
#  ▣ 문제 : 프로그램 코드에 슬라이스를 지시하는 하드코딩이 너무 많아 이해하기 어려운 상황이다. 이를 정리해야 한다.
#  ▣ 해결 : slice() 를 이용한다.
record = '....................100          .......513.25  ..........'
SHARES = slice(20, 32)
PRICE = slice(40, 48)
print(record[SHARES], record[PRICE])

#  ▣ 토론 : 일반적으로 프로그램을 작성할 때 하드코딩이 늘어날수록 이해하기 어렵고 지저분해진다.
#            따라서, slice() 를 사용해 조각을 미리 생성해 놓고 재사용하면 편리하다.
items = [0, 1, 2, 3, 4, 5, 6]
a = slice(2, 4)
print(items[2:4], items[a])

items[a] = [10, 11]
print(items)

del items[a]
print(items)

#   - start, stop, step 속성
a = slice(5, 50, 2)
print(a.start, a.stop, a.step)

#   - indices(size) 메소드 사용
s = 'HelloWorld'
print(a.indices(len(s)))

for i in range(*a.indices(len(s))):
    print(s[i])


#  1.12 시퀀스에 가장 많은 아이템 찾기
#  ▣ 문제 : 시퀀스에 가장 많이 나타난 아이템을 찾고 싶다.
#  ▣ 해결 : collections.Counter 클래스의 most_common() 메소드를 사용한다.
words = ['look', 'into', 'my', 'eyes', 'look', 'into', 'my', 'eyes', 'the', 'eyes', 'the', 'eyes', 'the', 'eyes', 'not', 'around',
         'the', 'eyes', "dont't", 'look', 'around', 'the', 'eyes', 'look', 'into', 'my', 'eyes', "you're", 'under']
from collections import Counter
word_counts = Counter(words)
top_three = word_counts.most_common(3)
print(top_three)

#  ▣ 토론 : Counter 객체에는 해시 가능한 모든 아이템을 입력할 수 있다. 내부적으로 Counter 는 아이템이 나타난 횟수를 가리키는 딕셔너리이다.
print(word_counts['not'], word_counts['eyes'])

#   - 카운트를 수동으로 증가시키고 싶다면 단순하게 더하기 또는 update() 메소드를 사용한다.
morewords = ['why', 'are', 'you', 'not', 'looking', 'in', 'my', 'eyes']
for word in morewords:
    word_counts[word] += 1
print(word_counts['eyes'])

word_counts.update(morewords)
print(word_counts)

#   - Counter 인스턴스에 잘 알려지지 않은 기능으로 여러 가지 수식을 사용할 수 있다는 점이 있다.
a = Counter(words)
b = Counter(morewords)
c = a + b
print(c)
d = a - b
print(d)


#  1.13 일반 키로 딕셔너리 리스트 정렬
#  ▣ 문제 : 딕셔너리 리스트가 있고, 하나 혹은 그 이상의 딕셔너리 키로 이를 정렬하고 싶다.
#  ▣ 해결 : operator 모듈의 itemgetter 함수를 사용하면 쉽게 정렬할 수 있다.
rows = [{'fname': 'Brian', 'lname': 'Jones', 'uid': 1003},
        {'fname': 'David', 'lname': 'Beazley', 'uid': 1002},
        {'fname': 'John', 'lname': 'Cleese', 'uid': 1001},
        {'fname': 'Big', 'lname': 'Jones', 'uid': 1004}]
from operator import itemgetter
rows_by_fname = sorted(rows, key=itemgetter('fname'))
rows_by_uid = sorted(rows, key=itemgetter('uid'))
print(rows_by_fname)
print(rows_by_uid)

rows_by_lfname = sorted(rows, key=itemgetter('lname', 'fname'))
print(rows_by_lfname)

#  ▣ 토론 : 이번 예제에서 키워드 인자 key 를 받는 내장 함수 sorted() 에 rows 를 전달했다.
#            이 인자는 rows 로부터 단일 아이템을 받는 호출 가능 객체를 입력으로 받고 정렬의 기본이 되는 값(튜플)을 반환한다.

#   - itemgetter() -> lambda 표현식으로 대체 가능
rows_by_fname = sorted(rows, key=lambda r: r['fname'])
rows_by_lfname = sorted(rows, key=lambda r: (r['lname'], r['fname']))
print(min(rows, key=lambda r: r['uid']), min(rows, key=itemgetter('uid')))
#  ※ lambda 표현식보다 itemgetter() 메소드가 실행 속도가 조금 더 빠르므로 성능상 itemgetter() 를 사용하는 것이 좋다.


#  1.14 기본 비교 기능 없이 객체 정렬
#  ▣ 문제 : 동일한 클래스 객체를 정렬해야 하는데, 이 클래스는 기본적인 비교 연산을 제공하지 않는다.
#  ▣ 해결 : 내장 함수 sorted()는 key 인자에 호출 가능 객체를 받아 sorted 가 객체 비교에 사용할 수 있는 값을 반환한다.
#            특정 클래스의 인스턴스를 입력으로 받고 특정 요소를 반환하는 코드를 작성한다.
class User:
    def __init__(self, user_id):
        self.user_id = user_id

    def __repr__(self):
        return 'User({})'.format(self.user_id)

users = [User(23), User(3), User(99)]
print(users)
print(sorted(users, key=lambda v: v.user_id))
#   - lambda 대신에 operator.attrgetter() 를 사용해도 된다.
from operator import attrgetter
print(sorted(users, key=attrgetter('user_id')))

#  ▣ 토론 : lambda 나 attrgetter() 를 사용할지 여부는 개인의 선호도에 따라 갈리지만, attrgetter() 의 속도가 빠른 경우가
#            종종 있고 동시에 여러 필드를 추출하는 기능이 추가되어 있다.
#            min(), max() 함수에도 사용할 수 있다.
by_name = sorted(users, key=attrgetter('last_name', 'first_name'))
print(min(users, key=attrgetter('user_id')), max(users, key=attrgetter('user_id')))


#  1.15 필드에 따라 레코드 묶기
#  ▣ 문제 : 일련의 딕셔너리나 인스턴스가 있고 특정 필드 값에 기반한 그룹의 데이터를 순환하고 싶다.
#  ▣ 해결 : itertools.groupby() 함수로 데이터를 묶는다.
rows = [{'address': '5412 N CLARK', 'date': '07/01/2012'},
        {'address': '5148 N CLARK', 'date': '07/04/2012'},
        {'address': '5800 E 58TH', 'date': '07/02/2012'},
        {'address': '2122 N CLARK', 'date': '07/03/2012'},
        {'address': '5645 N RAVENSWOOD', 'date': '07/02/2012'},
        {'address': '1060 W ADDISON', 'date': '07/02/2012'},
        {'address': '4801 N BROADWAY', 'date': '07/01/2012'},
        {'address': '1039 W GRANVILLE', 'date': '07/04/2012'}]
from operator import itemgetter
from itertools import groupby

#   - 정렬 후 group by 를 수행해야 정상적인 결과가 나온다.
rows.sort(key=itemgetter('date'))

for date, items in groupby(rows, key=itemgetter('date')):
    print(date)
    for i in items:
        print('    ', i)

#  ▣ 토론 : groupby() 함수는 시퀀스를 검색하고 동일한 값에 대한 일련의 '실행'을 찾는다. 개별 순환에 대해서 값, 그리고
#            같은 값을 가진 그룹의 모든 아이템을 만드는 이터레이터를 함께 반환한다.
#   - defaultdict 를 사용해서 multidict 를 구성하는 방법
from collections import defaultdict
rows_by_date = defaultdict(list)
for row in rows:
    rows_by_date[row['date']].append(row)

for r in rows_by_date['07/01/2012']:
    print(r)


#  1.16 시퀀스 필터링
#  ▣ 문제 : 시퀀스 내부에 데이터가 있고 특정 조건에 따라 값을 추출하거나 줄이고 싶다.
#  ▣ 해결 : 가장 간단한 해결책은 list comprehension 이다.
mylist = [1, 4, -5, 10, -7, 2, 3, -1]
print([n for n in mylist if n > 0])
print([n for n in mylist if n < 0])

#   - 위의 경우 입력된 내용이 크면 매우 큰 결과가 생성될 수도 있으므로, 생성자 표현식을 사용해서 값을 걸러낸다.
pos = (n for n in mylist if n > 0)
for x in pos:
    print(x)

#   - 필터링 도중에 예외 처리를 해야하는 경우 filter() 를 사용한다.
values = ['1', '2', '-3', '-', '4', 'N/A', '5']

def is_int(val):
    try:
        x = int(val)
        return True
    except ValueError:
        return False

ivals = list(filter(is_int, values))
print(ivals)

#  ▣ 토론 : list comprehension 과 생성자 표현식은 간단한 데이터를 걸러 내기 위한 가장 쉽고 직관적인 방법이다.
#            또한 동시에 데이터 변형 기능도 가지고 있다.
mylist = [1, 4, -5, 10, -7, 2, 3, -1]
import math
print([math.sqrt(n) for n in mylist if n > 0])

#   - list comprehension 을 통한 새로운 값으로 치환
clip_neg = [n if n > 0 else 0 for n in mylist]
print(clip_neg)

#   - itertools.compress() 를 통한 데이터 필터링.
addresses = ['5412 N CLARK',
             '5148 N CLARK',
             '5800 E 58TH',
             '2122 N CLARK',
             '5645 N RAVENSWOOD',
             '1060 W ADDISON',
             '4801 N BROADWAY',
             '1039 W GRANVILLE']
counts = [0, 3, 10, 4, 1, 7, 6, 1]

from itertools import compress
more5 = [n > 5 for n in counts]
print(list(compress(addresses, more5)))


#  1.17 딕셔너리의 부분 추출
#  ▣ 문제 : 딕셔너리의 특정 부분으로부터 다른 딕셔너리를 만들고 싶다.
#  ▣ 해결 : dictionary comprehension 을 사용하면 간단하게 해결된다.
prices = {'ACME': 45.23, 'AAPL': 612.78, 'IBM': 205.55, 'HPQ': 37.20, 'FB': 10.75}

#   - 가격이 200 이상인 것에 대한 딕셔너리
p1 = {k: v for k, v in prices.items() if v >= 200}

#   - 기술 관련 주식으로 딕셔너리 구성
tech_names = {'AAPL', 'IBM', 'HPQ', 'MSFT'}
p2 = {k: v for k, v in prices.items() if k in tech_names}
print(p1, p2)

#  ▣ 토론 : dictionary comprehension 으로 할 수 있는 대부분의 일은 튜플 시퀀스를 만들고 dict() 함수에 전달하는 것으로도 할 수 있다.
p1 = dict((key, value) for key, value in prices.items() if value > 200)
#   ※ 위와 같이 해도 가능하지만 실행 속도 측면에서 dictionary comprehension 을 사용한 경우보다 더 느리다.
p2 = {key: prices[key] for key in prices.keys() & tech_names}
#   ※ 위 방식도 처음 방식보다 1.6배 정도 느리다.


#  1.18 시퀀스 요소에 이름 매핑
#  ▣ 문제 : 리스트나 튜플의 요소에 이름으로 접근 가능하도록 수정하고 싶다.
#  ▣ 해결 : collections.namedtuple() 을 사용하면 일반적인 튜플 객체를 사용하는 것에 비해 그리 크지 않은 오버헤드로 이 기능을 구현한다.
#            collections.namedtuple() 은 실제로 표준 파이썬 tuple 타입의 서브클래스를 반환하는 팩토리 메소드이다.
from collections import namedtuple
Subscriber = namedtuple('Subscriber', ['addr', 'joined'])
sub = Subscriber('jonesy@example.com', '2012-10-19')
print(sub)
print(sub.addr, sub.joined)

#   - namedtuple 의 인스턴스는 일반적인 클래스 인스턴스와 비슷해 보이지만 튜플과 교환이 가능하고, 인덱싱이나 언패킹과 같은 튜플의
#     일반적인 기능을 모두 지원한다.
print(len(sub))
addr, joined = sub  # 언패킹
print(addr, joined)

#   - 일반적인 튜플을 사용하는 코드
def compute_cost(records):
    total = 0.0
    for rec in records:
        total += rec[1] * rec[2]
    return total

#   - namedtuple 을 사용한 코드
from collections import namedtuple
Stock = namedtuple('Stock', ['name', 'shares', 'price'])
def compute_cost(records):
    total = 0.0
    for rec in records:
        s = Stock(*rec)
        total += s.shares * s.price
    return total

#  ▣ 토론 : namedtuple 은 저장 공간을 더 필요로 하는 딕셔너리 대신 사용할 수 있다.
#            딕셔너리를 포함한 방대한 자료 구조를 구상하고 있다면 namedtuple 을 사용하는 것이 더 효율적이다.
#            하지만 딕셔너리와는 다르게 네임드 튜플은 수정할 수 없다.
s = Stock('ACME', 100, 123.45)
print(s)
s.shares = 75

#   - 속성을 수정해야 한다면 namedtuple 인스턴스의 _replace() 메소드를 사용해야 한다.
s = s._replace(shares=75)
print(s)

#   - _replace() 메소드를 사용해서 옵션이나 빈 필드를 가진 네임드 튜플을 간단히 만들 수 있다.
from collections import namedtuple
Stock = namedtuple('Stock', ['name', 'shares', 'price', 'date', 'time'])

stock_prototype = Stock('', 0, 0.0, None, None)  # prototype instance 생성

def dict_to_stock(s):  # dictionary 를 Stock 으로 변환하는 함수
    return stock_prototype._replace(**s)

a = {'name': 'ACME', 'shares': 100, 'date': '2012-02-22'}
print(dict_to_stock(a))


#  1.19 데이터를 변환하면서 줄이기
#  ▣ 문제 : 감소 함수(sum, min, max)를 실행해야 하는데, 먼저 데이터를 변환하거나 필터링해야 한다.
#  ▣ 해결 : 생성자 표현식을 사용해서 처리한다.
nums = [1, 2, 3, 4, 5]
s = sum(x * x for x in nums)
print(s)

#   - 디렉터리에 또 다른 .py 파일이 있는지 살펴본다.
import os
files = os.listdir('./files')
if any(name.endswith('.py') for name in files):
    print('There be python!')
else:
    print('Sorry, no python.')

#   - 튜플을 CSV 로 출력한다.
s = ('ACME', 50, 123.45)
print(','.join(str(x) for x in s))

#   - 자료 구조의 필드를 줄인다.
portfolio = [{'name': 'GOOG', 'shares': 50},
             {'name': 'YHOO', 'shares': 75},
             {'name': 'AOL', 'shares': 20},
             {'name': 'SCOX', 'shares': 65}]
min_shares = min(s['shares'] for s in portfolio)

#  ▣ 토론 : 위의 코드는 함수에 인자로 전달된 생성자 표현식의 문법적인 측면을 보여준다.
s = sum((x * x for x in nums))
s = sum(x * x for x in nums)
#   ※ 위의 두 식은 같다.

#   - min, max 같은 함수는 key 라는 여러 상황에 유용한 인자를 받기 때문에 생성자 방식을 사용해야 하는 이유를 더 만들어 준다.
min_shares = min(s['shares'] for s in portfolio)
min_shares = min(portfolio, key=lambda v: v['shares'])
from operator import itemgetter
min_shares = min(portfolio, key=itemgetter('shares'))
print(min_shares)


#  1.20 여러 매핑을 단일 매핑으로 합치기
#  ▣ 문제 : 딕셔너리나 매핑이 여러 개 있고, 자료 검색이나 데이터 확인을 위해서 하나의 매핑으로 합치고 싶다.
#  ▣ 해결 : collections 모듈의 ChainMap 클래스를 사용하면 된다.
a = {'x': 1, 'z': 3}
b = {'y': 2, 'z': 4}
from collections import ChainMap
c = ChainMap(a, b)
print(c)
print(c['x'])
print(c['y'])
print(c['z'])

#  ▣ 토론 : ChainMap 은 매핑을 여러 개 받아서 하나처럼 보이게 한다.
#            하지만 그렇게 보이는 것일뿐 하나로 합치는 것은 아니다. 단지 매핑에 대한 리스트를 유지하면서 리스트를 스캔하도록
#            일반적인 딕셔너리 동작을 재정의한다.
print(len(c))
print(list(c.keys()), list(c.values()))

#   - 매핑의 값을 변경하는 동작은 언제나 리스트의 첫 번째 매핑에 영향을 준다
c['z'] = 10
c['w'] = 40
del c['z']
print(a)
del c['y']  # 두 번째 매핑에 있는 값이므로 변경이 안된다.

#   - ChainMap 은 프로그래밍 언어의 변수와 같이 범위가 있는 값(전역변수, 지역변수)에 사용하면 유용하다.
values = ChainMap()
values['x'] = 1
values = values.new_child()  # 새로운 매핑 추가
values['x'] = 2
values = values.new_child()
values['x'] = 3
print(values)
print(values['x'])
values = values.parents  # 마지막 매핑 삭제
print(values['x'])
values = values.parents
print(values['x'])
print(values)

#   - ChainMap 의 대안으로 update() 를 사용해 딕셔너리를 하나로 합칠 수도 있다.
a = {'x': 1, 'z': 3}
b = {'y': 2, 'z': 4}
merged = dict(b)
merged.update(a)
print(merged['x'], merged['y'], merged['z'])
a['x'] = 13
print(merged['x'])

#   - ChainMap 은 원본 딕셔너리를 참조하기 때문에 이와 같은 문제가 발생하지 않는다.
a = {'x': 1, 'z': 3}
b = {'y': 2, 'z': 4}
merged = ChainMap(a, b)
print(merged['x'])
a['x'] = 42
print(merged['x'])