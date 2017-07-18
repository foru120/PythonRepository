# Chapter 4. 이터레이터와 제너레이터
#  4.1 수동으로 이터레이터 소비
#  ▣ 문제 : 순환 가능한 아이템에 접근할 때 for 순환문을 사용하고 싶지 않다.
#  ▣ 해결 : 수동으로 이터레이터를 소비하려면 next() 함수를 사용하고 StopIteration 예외를 처리하기 위한 코드를 직접 작성한다.
with open('files/somefile.txt', 'r') as f:
    try:
        while True:
            line = next(f)
            print(line, end='')
    except StopIteration:
        pass

with open('files/somefile.txt', 'r') as f:
    while True:
        line = next(f, None)
        if line is None:
            break
        print(line, end='')

#  ▣ 토론 : 대개의 경우 순환에 for 문을 사용하지만 보다 더 정교한 조절이 필요한 때도 있다.
#           기저에서 어떤 동작이 일어나는지 정확히 알아둘 필요가 있다.
items = [1, 2, 3]
it = iter(items)
print(next(it))
print(next(it))
print(next(it))
print(next(it))


#  4.2 델리게이팅 순환
#  ▣ 문제 : 리스트, 튜플 등 순환 가능한 객체를 담은 사용자 정의 컨테이너를 만들었다.
#           이 컨테이너에 사용 가능한 이터레이터를 만들고 싶다.
#  ▣ 해결 : 일반적으로 컨테이너 순환에 사용할 __iter__() 메소드만 정의해 주면 된다.
class Node:
    def __init__(self, value):
        self._value = value
        self._children = []

    def __repr__(self):
        return 'Node({!r})'.format(self._value)

    def add_child(self, node):
        self._children.append(node)

    def __iter__(self):
        return iter(self._children)

if __name__ == '__main__':
    root = Node(0)
    child1 = Node(1)
    child2 = Node(2)
    root.add_child(child1)
    root.add_child(child2)
    for ch in root:
        print(ch)

#  ▣ 토론 : 파이썬의 이터레이터 프로토콜은 __iter__() 가 실제 순환을 수행하기 위한 __next__() 메소드를 구현하는 특별 이터레이터
#           객체를 반환하기를 요구한다.


#  4.3 제너레이터로 새로운 순환 패턴 생성
#  ▣ 문제 : 내장 함수(range(), reversed())와는 다른 동작을 하는 순환 패턴을 만들고 싶다.
#  ▣ 해결 : 새로운 순환 패턴을 만들고 싶다면, 제너레이터 함수를 사용해서 정의해야 한다.
def frange(start, stop, increment):
    x = start
    while x < stop:
        yield x
        x += increment

for n in frange(0, 4, 0.5):
    print(n)

for n in frange(0, 1, 0.125):
    print(n)

#  ▣ 토론 : 내부의 yield 문의 존재로 인해 함수가 제너레이터가 되었다.
#           일반 함수와는 다르게 제너레이터는 순환에 응답하기 위해 실행된다.
def countdown(n):
    print('Starting to count from', n)
    while n > 0:
        yield n
        n -= 1
    print('Done!')

c = countdown(3)
print(c)

print(next(c))
print(next(c))
print(next(c))
print(next(c))


#  4.4 이터레이터 프로토콜 구현
#  ▣ 문제 : 순환을 지원하는 객체를 만드는데, 이터레이터 프로토콜을 구현하는 쉬운 방법이 필요하다.
#  ▣ 해결 : 객체에 대한 순환을 가장 쉽게 구현하는 방법은 제너레이터 함수를 사용하는 것이다.
#           노드를 깊이-우선 패턴으로 순환하는 이터레이터를 구현하고 싶다면 다음 코드를 참고한다.
class Node:
    def __init__(self, value):
        self._value = value
        self._children = []

    def __repr__(self):
        return 'Node({!r})'.format(self._value)

    def add_child(self, node):
        self._children.append(node)

    def __iter__(self):
        return iter(self._children)

    def depth_first(self):
        yield self
        for c in self:
            yield from c.depth_first()

if __name__ == '__main__':
    root = Node(0)
    child1 = Node(1)
    child2 = Node(2)
    root.add_child(child1)
    root.add_child(child2)
    child1.add_child(Node(3))
    child1.add_child(Node(4))
    child2.add_child(Node(5))

    for ch in root.depth_first():
        print(ch)


#  4.5 역방향 순환
#  ▣ 문제 : 시퀀스 아이템을 역방향으로 순환하고 싶다.
#  ▣ 해결 : 내장 함수 reversed() 를 사용한다.
a = [1, 2, 3, 4]
for x in reversed(a):
    print(x)

#   - 역방향 순환은 객체가 __reversed__() 특별 메소드를 구현하고 있거나 크기를 알 수 있는 경우에만 가능하다.
#     두 조건 중에서 아무것도 만족하지 못하면 객체를 먼저 리스트로 변환해야 한다.
f = open('files/somefile.txt')
for line in reversed(list(f)):
    print(line, end='')

#   - 순환 가능 객체를 리스트로 변환할 때 많은 메모리가 필요하다.


#  4.6 추가 상태를 가진 제너레이터 함수 정의
#  ▣ 문제 : 제너레이터 함수를 정의하고 싶지만, 사용자에게 노출할 추가적인 상태를 넣고 싶다.
#  ▣ 해결 : 사용자에게 추가 상태를 노출하는 제너레이터를 원할 때, __iter__() 메소드에 제너레이터 함수 코드를 넣어서 쉽게 클래스로
#           구현할 수 있다.
from collections import deque

class linehistory:
    def __init__(self, lines, histlen=3):
        self.lines = lines
        self.history = deque(maxlen=histlen)

    def __iter__(self):
        for lineno, line in enumerate(self.lines, 1):  # enumerate(self.lines, 1) 함수의 출력값 : (줄 번호, 줄 내용), 1번부터 번호 출력
            self.history.append((lineno, line))
            yield line

    def clear(self):
        self.history.clear()

with open('files/somefile.txt') as f:
    lines = linehistory(f)
    for line in lines:
        if 'python' in line:
            for lineno, hline in lines.history:
                print('{}:{}'.format(lineno, hline), end='')

#  ▣ 토론 : 제너레이터를 사용하면 모든 작업을 함수만으로 하려는 유혹에 빠지기 쉽다.
#           만약 제너레이터 함수가 프로그램의 다른 부분과 일반적이지 않게 상호작용해야 할 경우 코드가 꽤 복잡해질 수 있다.
#           이럴 때는 앞에서 본 대로 클래스 정의만을 사용한다.
f = open('files/somefile.txt')
lines = linehistory(f)
next(lines)  # __iter__ 메서드가 iter 객체를 리턴하지 않아 next 메소드가 호출되지 않는다.

it = iter(lines)
next(it)
next(it)


#  4.7 이터레이터의 일부 얻기
#  ▣ 문제 : 이터레이터가 만드는 데이터의 일부를 얻고 싶지만, 일반적인 자르기 연산자가 동작하지 않는다.
#  ▣ 해결 : 이터레이터와 제너레이터의 일부를 얻는 데는 itertools.islice() 함수가 가장 이상적이다.
def count(n):
    while True:
        yield n
        n += 1

c = count(0)
print(c[10:20])  # 제너레이터는 슬라이스 연산자가 동작하지 않는다.

import itertools
for x in itertools.islice(c, 10 ,20):
    print(x)

#  ▣ 토론 : 이터레이터와 제너레이터는 일반적으로 일부를 잘라낼 수 없다. 왜냐하면 데이터의 길이를 알 수 없기 때문이다.
#            islice() 의 실행 결과는 원하는 아이템의 조각을 생성하는 이터레이터지만, 이는 시작 인덱스까지 모든 아이템을 소비하고
#            버리는 식으로 수행한다. 그리고 그 뒤의 아이템은 마지막 인덱스를 만날 때까지 islice 객체가 생성한다.


#  4.8 순환 객체 첫 번째 부분 건너뛰기
#  ▣ 문제 : 순환 객체의 아이템을 순환하려고 하는데, 처음 몇가지 아이템에는 관심이 없어 건너뛰고 싶다.
#  ▣ 해결 : itertools 모듈이 이 용도로 사용할 수 있는 몇 가지 함수가 있다. 첫 번째는 itertools.dropwhile() 함수이다.
#            이 함수를 사용하려면, 함수와 순환 객체를 넣으면 된다. 반환된 이터레이터는 넘겨준 함수가 True 를 반환하는 동안은
#            시퀀스의 첫 번째 아이템을 무시한다.
with open('files/somefile.txt') as f:
    for line in f:
        print(line, end='')

from itertools import dropwhile
with open('files/somefile.txt') as f:
    for line in dropwhile(lambda line: line.startswith('#'), f):
        print(line, end='')

from itertools import islice
items = ['a', 'b', 'c', 1, 4, 10, 15]
for x in islice(items, 3, None):
    print(x)

#  ▣ 토론 : dropwhile() 과 islice() 함수는 다음과 같이 복잡한 코드를 작성하지 않도록 도와준다.
with open('files/somefile.txt') as f:
    # 처음 주석을 건너뛴다.
    while True:
        line = next(f, '')
        if not line.startswith('#'):
            break

    # 남아 있는 라인을 처리한다.
    while line:
        # 의미 있는 라인으로 치환한다.
        print(line, end='')
        line = next(f, None)

#   - 파일 전체에 걸쳐 주석으로 시작하는 모든 라인을 필터링
with open('files/somefile.txt') as f:
    lines = (line for line in f if not line.startswith('#'))
    for line in lines:
        print(line, end='')


#  4.9 가능한 모든 순열과 조합 순환
#  ▣ 문제 : 아이템 컬렉션에 대해 가능한 모든 순열과 조합을 순환하고 싶다.
#  ▣ 해결 : itertools 모듈은 이와 관련 있는 세 함수를 제공한다.

#   - itertools.permutations() : 아이템 컬렉션을 받아 가능한 모든 순열을 튜플 시퀀스로 생성
items = ['a', 'b', 'c']
from itertools import permutations
for p in permutations(items):
    print(p)

for p in permutations(items, 2):  # 특정 길이의 순열을 원하는 경우
    print(p)

#   - itertools.combinations() : 입력 받은 아이템의 가능한 조합을 생성
#     조합의 경우 실제 요소의 순서는 고려하지 않는다
from itertools import combinations
for c in combinations(items, 3):
    print(c)

for c in combinations(items, 2):
    print(c)

#   - itertools.combinations_with_replacement() : 같은 아이템을 두 번 이상 선택할 수 있게 한다.
from itertools import combinations_with_replacement
for c in combinations_with_replacement(items, 3):
    print(c)

#  ▣ 토론 : 이번 레시피에서 itertools 모듈의 편리한 도구 중 몇 가지만을 다루었다.
#            사실 순열이나 조합을 순환하는 코드를 직접 작성할 수도 있겠지만, 그렇게 하려면 꽤 많은 고민을 해야 한다.
#            순환과 관련해서 복잡한 문제에 직면한다면 우선 itertools 부터 살펴보는 것이 좋다.


#  4.10 인덱스-값 페어 시퀀스 순환
#  ▣ 문제 : 시퀀스를 순환하려고 한다. 이때 어떤 요소를 처리하고 있는지 번호를 알고 싶다.
#  ▣ 해결 : 내장 함수 enumerate() 를 사용하면 간단히 해결할 수 있다.
my_list = ['a', 'b', 'c']
for idx, val in enumerate(my_list):
    print(idx, val)

#   - 출력 시 번호를 1번부터 시작
my_list = ['a', 'b', 'c']
for idx, val in enumerate(my_list, 1):
    print(idx, val)

def parse_data(filename):
    with open('PythonCookBook/files/'+filename, 'rt') as f:
        for lineno, line in enumerate(f, 1):
            fields = line.split()
            try:
                count = int(fields[0])
            except ValueError as e:
                print('Line {}: Parse error: {}'.format(lineno, e))

parse_data('somefile.txt')

#   - enumerate() 는 특정 값의 출현을 위한 오프셋 추적에 활용하기 좋다.
from collections import defaultdict
word_summary = defaultdict(list)

with open('PythonCookBook/files/somefile.txt', 'r') as f:
    lines = f.readlines()

for idx, line in enumerate(lines):
    # 현재 라인에 단어 리스트를 생성
    words = [w.strip().lower() for w in line.split()]
    for word in words:
        word_summary[word].append(idx)

print(word_summary)

#  ▣ 토론 : 카운터 변수를 스스로 다루는 것에 비해 enumerate() 를 사용하는 것이 훨씬 보기 좋다.
lineno = 1
for line in f:
    lineno += 1

for lineno, line in enumerate(f):
    print(lineno)

#   - 한 번 더 풀어 줘야 하는 튜플의 시퀀스에 enumerate() 를 사용하는 경우
data = [(1, 2), (3, 4), (5, 6), (7, 8)]

for n, (x, y) in enumerate(data):  # 올바른 방법
    print(n, (x, y))

for n, x, y in enumerate(data):  # 에러!
    print(n, (x, y))


#  4.11 여러 시퀀스 동시에 순환
#  ▣ 문제 : 여러 시퀀스에 들어 있는 아이템을 동시에 순환하고 싶다.
#  ▣ 해결 : 여러 시퀀스를 동시에 순환하려면 zip() 함수를 사용한다.
xpts = [1, 5, 4, 2, 10, 7]
ypts = [101, 78, 37, 15, 62, 99]
for x, y in zip(xpts, ypts):
    print(x, y)

#   - 순환은 한쪽 시퀀스의 모든 입력이 소비되었을 때 정지한다. 따라서 순환의 길이는 입력된 시퀀스 중 짧은 것과 같다.
a = [1, 2, 3]
b = ['w', 'x', 'y', 'z']
for i in zip(a, b):
    print(i)

#   - 긴 시퀀스를 기준으로 순환을 수행하려면 itertools.zip_longest() 를 사용한다.
from itertools import zip_longest
for i in zip_longest(a, b):
    print(i)

for i in zip_longest(a, b, fillvalue=0):
    print(i)

#  ▣ 토론 : zip() 은 데이터를 묶어야 할 때 주로 사용한다.
headers = ['name', 'shares', 'price']
values = ['ACME', 100, 490.1]
s = dict(zip(headers, values))  # zip() 을 사용해서 두 값을 딕셔너리로 생성
for name, val in zip(headers, values):
    print(name, '=', val)

#   - zip() 에 시퀀스를 두 개 이상 입력할 수 있다.
a = [1, 2, 3]
b = [10, 11, 12]
c = ['x', 'y', 'z']
for i in zip(a, b, c):
    print(i)

#   - zip() 이 결과적으로 이터레이터를 생성한다는 점을 기억하자. 묶은 값이 저장된 리스트가 필요하다면 list() 함수를 사용한다.
zip(a, b)
list(zip(a, b))


#  4.12 서로 다른 컨테이너 아이템 순환
#  ▣ 문제 : 여러 객체에 동일한 작업을 수행해야 하지만, 객체가 서로 다른 컨테이너에 들어 있다.
#  ▣ 해결 : itertools.chain() 메소드로 이 문제를 간단히 해결할 수 있다. 타입이 달라도 가능하다.
from itertools import chain
a = [1, 2, 3, 4]
b = ['x', 'y', 'z']
for x in chain(a, b):
    print(x)

#   - chain() 은 일반적으로 모든 아이템에 동일한 작업을 수행하고 싶지만 이 아이템이 서로 다른 세트에 포함되어 있을 때 사용한다.
#     반복문을 두 번 사용하는 것보다 훨씬 보기 좋다.
active_items = set(list([1,2,3,4]))
inactive_items = [1,2,3,4]

for item in chain(active_items, inactive_items):
    print(item)

#  ▣ 토론 : itertools.chain() 은 하나 혹은 그 이상의 순환 객체를 인자로 받는다.
#            그리고 입력 받은 순환 객체 속 아이템을 차례대로 순환하는 이터레이터를 생성한다.
#            큰 차이는 아니지만, 우선적으로 시퀀스를 하나로 합친 다음 순환하는 것보다 chain() 을 사용하는 것이 더 효율적이다.
for x in a + b:  # 비효율적(a 와 b 가 동일한 타입이어야 한다.)
    pass

for x in chain(a, b):  # 더 나은 방식
    pass


#  4.13 데이터 처리 파이프라인 생성
#  ▣ 문제 : 데이터 처리를 데이터 처리 파이프라인과 같은 방식으로 순차적으로 처리하고 싶다.
#            예를 들어, 처리해야 할 방대한 데이터가 있지만 메모리에 한꺼번에 들어가지 않는 경우에 적용할 수 있다.
#  ▣ 해결 : 제너레이터 함수를 사용하는 것이 처리 파이프라인을 구현하기에 좋다.
import os
import fnmatch
import gzip
import bz2
import re

def gen_find(filepat, top):
    '''
        디렉터리 트리에서 와일드 카드 패턴에 매칭하는 모든 파일 이름을 찾는다.
    '''
    for path, dirlist, filelist in os.walk(top):  # walk() : 파일 시스템 경로, 디렉토리 리스트, 파일 리스트를 트리 탐색으로 가져오는 함수
        for name in fnmatch.filter(filelist, filepat):  # filter() : 패턴에 해당하는 파일을 걸러내는 함수
            yield os.path.join(path, name)  # join() : 여러개의 경로를 합치는 함수

def gen_opener(filenames):
    '''
        파일 이름 시퀀스를 하나씩 열어 파일 객체를 생성한다.
        다음 순환으로 넘어가는 순간 파일을 닫는다.
    '''
    for filename in filenames:
        if filename.endswith('.gz'):
            f = gzip.open(filename, 'rt')
        elif filename.endswith('.bz2'):
            f = bz2.open(filename, 'rt')
        else:
            f = open(filename, 'rt')
        yield f
        f.close()

def gen_concatenate(iterators):
    '''
        이터레이터 시퀀스를 합쳐 하나의 시퀀스로 만든다.
    '''
    for it in iterators:
        yield from it

def gen_grep(pattern, lines):
    '''
        라인 시퀀스에서 정규식 패턴을 살펴본다.
    '''
    pat = re.compile(pattern)
    for line in lines:
        if pat.search(line):
            yield line

lognames = gen_find('*.txt', 'PythonCookBook/files/')
files = gen_opener(lognames)
lines = gen_concatenate(files)
pylines = gen_grep('(?i)Python', lines)

for line in pylines:
    print(line)

#  - 전송한 바이트 수를 찾고 그 총합을 구함
bytecolumn = (line.rsplit(None, 1)[1] for line in pylines)  # rsplit 오른쪽부터 처리
print([x for x in bytecolumn])
bytes = (int(x) for x in bytecolumn if x != '-')
print('Total', sum(bytes))

#  ▣ 토론 : 파이프라인으로 데이터를 처리하는 방식은 파싱, 실시간 데이터 읽기, 주기적 폴링 등 다른 문제에도 사용할 수 있다.
#            코드를 이해할 때, yield 문이 데이터 생성자처럼 동작하고 for 문은 데이터 소비자처럼 동작한다는 점이 중요하다.
#            제너레이터가 쌓이면, 각 yield 가 순환을 하며 데이터의 아이템 하나를 파이프라인의 다음 단계로 넘긴다.
#            마지막 예제에서 sum() 함수가 실질적으로 프로그램을 운용하며 제너레이터 파이프라인에서 한 번에 하나씩 아이템을 꺼낸다.


#  4.14 중첩 시퀀스 풀기
#  ▣ 문제 : 중첩된 시퀀스를 합쳐 하나의 리스트로 만들고 싶다.
#  ▣ 해결 : 이 문제는 yield from 문이 있는 재귀 제너레이터를 만들어 쉽게 해결할 수 있다.
from collections import Iterable

def flatten(items, ignore_types=(str, bytes)):
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, ignore_types):
            yield from flatten(x)  # flatten 호출시 수행된 yield 값들을 모두 가져옴
        else:
            yield x

items = [1, 2, [3, 4, [5, 6], 7], 8]

for x in flatten(items):
    print(x)

#   - 앞의 코드에서 isinstance(x, Iterable) 은 아이템이 순환 가능한 것인지 확인한다.
#     순환이 가능하다면 yield from 으로 모든 값을 하나의 서브루틴으로 분출한다.
#     결과적으로 중첩되지 않은 시퀀스 하나가 만들어진다.

items = ['Dave', 'Paula', ['Thomas', 'Lewis']]
for x in flatten(items):
    print(x)

#   - 추가적으로 전달 가능한 인자 ignore_types 와 not isinstance(x, ignore_types) 로 문자열과 바이트가 순환 가능한 것으로
#     해석되지 않도록 했다.
#     이렇게 해야만 리스트에 담겨있는 문자열을 전달했을 때 문자를 하나하나 펼치지 않고 문자열 단위로 전개한다.

#  ▣ 토론 : 서브루틴으로써 다른 제너레이터를 호출할 때 yield from 을 사용하면 편리하다.
#            이 구문을 사용하지 않으면 추가적인 for 문이 있는 코드를 작성해야 한다.
def flatten(items, ignore_types=(str, bytes)):
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, ignore_types):
            for i in flatten(x):
                yield i
        else:
            yield x


#  4.15 정렬된 여러 시퀀스를 병합 후 순환
#  ▣ 문제 : 정렬된 시퀀스가 여럿 있고, 이들을 하나로 합친 후 정렬된 시퀀스를 순환하고 싶다.
#  ▣ 해결 : 간단하다. heapq.merge() 함수를 사용하면 된다.
import heapq
a = [1, 4, 7, 10]
b = [2, 5, 6, 11]
for c in heapq.merge(a, b):
    print(c)

#  ▣ 토론 : heapq.merge 는 아이템에 순환적으로 접근하며 제공한 시퀀스를 한꺼번에 읽지 않는다.
#           따라서 아주 긴 시퀀스도 별다른 무리 없이 사용할 수 있다.
import heapq

with open('sorted_file_1', 'rt') as file1, open('sorted_file_2', 'rt') as file2, open('merged_file', 'wt') as outf:
    for line in heapq.merge(file1, file2):
        outf.write(line)


#  4.16 무한 while 순환문을 이터레이터로 치환
#  ▣ 문제 : 함수나 일반적이지 않은 조건 테스트로 인해 무한 while 순환문으로 데이터에 접근하는 코드를 만들었다.
#  ▣ 해결 : 입출력과 관련 있는 프로그램에 일반적으로 다음과 같은 코드를 사용한다.
# CHUNKSIZE = 8192
#
# def reader(s):
#     while True:
#         data = s.recv(CHUNKSIZE)
#         if data == b'':
#             break
#         process_data(data)
#
# def reader(s):
#     for chunk in iter(lambda: s.recv(CHUNKSIZE), b''):
#         process_data(data)
import sys
f = open('PythonCookBook/files/somefile.txt')
for chunk in iter(lambda: f.read(10), ''):  # '' 를 만날때까지 파일 read 를 수행.
    n = sys.stdout.write(chunk)

#  ▣ 토론 : 내장 함수 iter() 의 기능은 거의 알려져 있지 않다.
#           이 함수에는 선택적으로 인자 없는 호출 가능 객체와 종료 값을 입력으로 받는다.
#           이렇게 사용하면 주어진 종료 값을 반환하기 전까지 무한히 반복해서 호출 가능 객체를 호출한다.
