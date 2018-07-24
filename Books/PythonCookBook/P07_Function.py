# Chapter 7. 함수
#  7.1 매개변수 개수에 구애 받지 않는 함수 작성
#  ▣ 문제 : 입력 매개변수 개수에 제한이 없는 함수를 작성하고 싶다.
#  ▣ 해결 : 위치 매개변수 개수에 제한이 없는 함수를 작성하려면 * 인자를 사용한다.
def avg(first, *rest):
    return (first + sum(rest)) / (1 + len(rest))

print(avg(1, 2), avg(1, 2, 3, 4))

#   - 키워드 매개변수 수에 제한이 없는 함수를 작성하려면 ** 로 시작하는 인자를 사용한다.
import html

def make_element(name, value, **attrs):
    keyvals = [' %s="%s"' % item for item in attrs.items()]
    attr_str = ''.join(keyvals)
    element = '<{name}{attrs}>{value}</{name}>'.format(name=name, attrs=attr_str, value=html.escape(value))
    return element

print(make_element('item', 'Albatross', size='large', quantity=6))
print(make_element('p', '<spam>'))

#   - 위치 매개변수와 키워드 매개변수를 동시에 받는 함수를 작성하려면, * 와 ** 를 함께 사용하면 된다.
def anyargs(*args, **kwargs):
    print(args)  # 튜플 args
    print(kwargs)  # 딕셔너리 kwargs

#  ▣ 토론 : * 는 함수 정의의 마지막 위치 매개변수 자리에만 올 수 있다.
#           ** 는 마지막 매개변수 자리에만 올 수 있다. 그리고 * 뒤에도 매개변수가 또 나올 수 있다는 것이 함수 정의의 미묘한 점이다.
def a(x, *args, y):
    pass

def b(x, *args, y, **kwargs):
    pass


#  7.2 키워드 매개변수만 받는 함수 작성
#  ▣ 문제 : 키워드로 지정한 특정 매개변수만 받는 함수가 필요하다.
#  ▣ 해결 : 이 기능은 키워드 매개변수를 * 뒤에 넣거나 이름 없이 * 만 사용하면 간단히 구현할 수 있다.
def recv(maxsize, *, block):
    print('Receives a message')
    pass

recv(1024, True)
recv(1024, block=True)

#   - 숫자가 다른 위치 매개변수를 받는 함수에 키워드 매개변수를 명시하는 경우.
def mininum(*values, clip=None):
    m = min(values)
    if clip is not None:
        m = clip if clip > m else m
    return m

print(mininum(1, 5, 2, -5, 10))
print(mininum(1, 5, 2, -5, 10, clip=0))

#  ▣ 토론 : 키워드로만 넣을 수 있는(keyword-only) 인자는 추가적 함수 인자를 명시할 때 코드의 가독성을 높이는 좋은 수단이 될 수 있다.
msg = recv(1024, False)
#  ※ 위의 경우처럼 recv() 가 어떻게 동작하는지 잘 모르는 사람이 있다면 False 인자가 무엇을 의미하는지도 모를 것이다.
#     따라서 호출하는 측에서 다음과 같은 식으로 표시해 준다면 이해하기 훨씬 쉽다.
msg = recv(1024, block=False)

#   - 키워드로만 넣을 수 있는 인자는 **kwargs 와 관련된 것에 사용자가 도움을 요청하면 도움말 화면에 나타난다.
print(help(recv))


#  7.3 함수 인자에 메타데이터 넣기
#  ▣ 문제 : 함수를 작성했다. 이때 인자에 정보를 추가해서 다른 사람이 함수를 어떻게 사용해야 하는지 알 수 있도록 하고 싶다.
#  ▣ 해결 : 함수 인자 주석으로 프로그래머에게 이 함수를 어떻게 사용해야 할지 정보를 줄 수 있다.
def add(x: int, y: int) -> int:
    return x + y

#   - 파이썬 인터프리터는 주석에 어떠한 의미도 부여하지 않는다.
#     타입을 확인하지도 않고, 파이썬의 실행 방식이 달라지지도 않는다.
#     단지 소스 코드를 읽는 사람이 이해하기 쉽도록 설명을 할 뿐이다.
help(add)
print(add(5, 4))

#   - 어떠한 객체도 함수에 주석으로 붙일 수 있지만, 대개 클래스나 문자열이 타당하다.

#  ▣ 토론 : 함수 주석은 함수의 __annotations__ 속성에 저장된다.
print(add.__annotations__)


#  7.4 함수에서 여러 값을 반환
#  ▣ 문제 : 함수에서 값을 여러 개 반환하고 싶다.
#  ▣ 해결 : 함수에서 값을 여러 개 반환하고 싶다면 간단히 튜플을 사용하면 된다.
def myfun():
    return 1, 2, 3

a, b, c = myfun()
print(a, b, c)

#  ▣ 토론 : myfun() 이 값을 여러 개 반환하는 것처럼 보이지만, 사실은 튜플 하나를 반환한 것이다.
#            조금 이상해 보이지만, 실제로 튜플을 생성하는 것은 쉼표지 괄호가 아니다.
a = (1, 2)
print(a)
b = 1, 2
print(b)

#   - 튜플 언패킹시 반환 값을 변수 하나에 할당하는 경우
x = myfun()
print(x)


#  7.5 기본 인자를 사용하는 함수 정의
#  ▣ 문제 : 함수나 메소드를 정의할 때 하나 혹은 그 이상 인자에 기본 값을 넣어 선택적으로 사용할 수 있도록 하고 싶다.
#  ▣ 해결 : 표면적으로 선택적 인자를 사용하는 함수를 정의하기는 쉽다. 함수 정의부에 값을 할당하고 가장 뒤에 이름 위치시키기만 하면 된다.
def spam(a, b=42):
    print(a, b)

spam(1)
spam(1, 2)

#   - 기본 값이 리스트, 세트, 딕셔너리 등 수정 가능한 컨테이너인 경우 None 을 사용해 코드 작성
def spam(a, b=None):
    if b is None:
        b = []

#   - 함수가 받은 값이 특정 값인지 아닌지 확인하는 코드
_no_value = object()

def spam(a, b=_no_value):
    if b is _no_value:
        print('No b value supplied')

spam(1)
spam(1, 2)
spam(1, None)

#  ▣ 토론 : 신경 써야 할 부분.
#   1. 할당하는 기본 값은 함수를 정의할 때 한 번만 정해지고 그 이후에는 변하지 않는다.
x = 42
def spam(a, b=x):
    print(a, b)

spam(1)
x = 23  # x 값이 바뀌어도 x = 42 일때 함수가 정의되었으므로 b = 42 이다.
spam(1)

#   2. 기본 값으로 사용하는 값은 None, True, False, 숫자, 문자열 같이 항상 변하지 않는 객체를 사용해야 한다.
def spam(a, b=[]):
    print(b)
    return b

x = spam(1)
print(x)

x.append(99)
x.append('Yow!')
print(x)
spam(1)
#   ※ 이런 부작용을 피하려면 앞의 예제에 나왔듯이 기본 값으로 None 을 할당하고 함수 내부에서 이를 확인하는 것이 좋다.

def spam(a, b=None):
    if not b:
        b = []

spam(1)      # 올바름
x = []
spam(1, x)   # 에러. x 값이 기본으로 덮어쓰여진다.
spam(1, 0)   # 에러. 0이 무시된다.
spam(1, '')  # 에러. ''이 무시된다.


#  7.6 이름 없는 함수와 인라인 함수 정의
#  ▣ 문제 : sort() 등에 사용할 짧은 콜백 함수를 만들어야 하는데, 한 줄짜리 함수를 만들면서 def 구문까지 사용하고 싶지는 않다.
#            그 대신 "인라인(in line)"이라 불리는 짧은 함수를 만들고 싶다.
#  ▣ 해결 : 표현식 계산 외에 아무 일도 하지 않는 간단한 함수는 lambda 로 치환할 수 있다.
add = lambda x, y: x + y
print(add(2, 3))
print(add('hello', 'world'))

#   - 앞에 나온 lambda 는 다음의 예제 코드와 완전히 동일하다.
def add(x, y):
    return x + y
print(add(2, 3))

#   - 일반적으로 lambda 는 정렬이나 데이터 줄이기 등 다른 작업에 사용할 때 많이 쓴다.
names = ['David Beazley', 'Brian Jones', 'Raymond Hettinger', 'Ned Batchelder']
print(sorted(names, key=lambda name: name.split()[-1].lower()))  # 뒤에 글자로 정렬하는 경우

#  ▣ 토론 : lambda 를 사용해서 간단한 함수를 정의할 수 있지만, 제약이 아주 많다.
#            우선 표현식을 하나만 사용해야 하고 그 결과가 반환 값이 된다.
#            따라서 명령문을 여러 개 쓴다거나 조건문, 순환문, 에러 처리 등을 넣을 수 없다.


#  7.7 이름 없는 함수에서 변수 고정
#  ▣ 문제 : lambda 를 사용해서 이름 없는 함수를 정의했는데, 정의할 때 특정 변수의 값을 고정하고 싶다.
#  ▣ 해결 : 다음 코드의 동작성을 고려해 보자.
x = 10
a = lambda y: x + y
x = 20
b = lambda y: x + y

print(a(10), b(10))  # lambda 에서 사용하는 값은 실행 시간에 따라 달라지는 변수이므로 값은 같다.

#   - 이름 없는 함수를 정의할 때 특정 값을 고정하는 경우
x = 10
a = lambda y, x=x: x + y
x = 20
b = lambda y, x=x: x + y
print(a(10), b(10))

#  ▣ 토론 : 리스트 컴프리핸션이나 반복문에서 람다 표현식을 생성하고 람다 함수가 순환 변수를 기억하려고 할때 문제가 발생한다.
funcs = [lambda x: x+n for n in range(5)]
for f in funcs:
    print(f(0))

#   - 다음 코드와 비교해 보자
funcs = [lambda x, n=n: x+n for n in range(5)]
for f in funcs:
    print(f(0))
#   ※ 이제 n 값을 함수를 정의하는 시점의 값으로 고정해 놓고 사용한다.


#  7.8 인자를 N개 받는 함수를 더 적은 인자로 사용
#  ▣ 문제 : 파이썬 코드에 콜백 함수나 핸들러로 사용할 호출체가 있다. 하지만 함수의 인자가 너무 많고 호출했을 때 예외가 발생한다.
#  ▣ 해결 : 함수의 인자 개수를 줄이려면 functools.partial() 을 사용해야 한다.
#            partial() 함수를 사용하면 함수의 인자에 고정 값을 할당할 수 있고, 따라서 호출할 때 넣어야 하는 인자 수를 줄일 수 있다.
def spam(a, b, c, d):
    print(a, b, c, d)

from functools import partial
s1 = partial(spam, 1)
s1(2, 3, 4)
s1(4, 5, 6)
s2 = partial(spam, d=42)
s2(1, 2, 3)
s2(4, 5, 5)
s3 = partial(spam, 1, 2, d=42)
s3(3)
s3(4)
s3(5)
#  ※ partial() 이 특정 인자의 값을 고정하고 새로운 호출체를 반환한다.
#     이 새로운 호출체는 할당하지 않은 인자를 받고, partial() 에 주어진 인자와 합친 후 원본 함수에 전달한다.

#  ▣ 토론 : 이번 레시피는 겉으로 보기에 호환되지 않을 것 같은 코드를 함께 동작하도록 하는 문제와 관련 있다.
#   - (x, y) 튜플로 나타낸 좌표 리스트가 있다. 다음 함수를 사용해서 두 점 사이의 거리를 구할 수 있다.
points = [(1, 2), (3, 4), (5, 6), (7, 8)]

import math
def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.hypot(x2 - x1, y2 - y1)  # math.hypot(x, y) : 평면에서 두 점 사이의 직선 거리를 구하는 함수

pt = (4, 3)
points.sort(key=partial(distance, pt))
print(points)

#   - 이 발상을 확장해서, 다른 라이브러리에서 사용하는 콜백 함수의 매개변수 설명을 변경하는 데 partial() 을 사용할 수도 있다.
#     예를 들어 multiprocessing 을 사용해서 비동기식으로 결과를 계산하고, 결과 값과 로깅 인자를 받는 콜백 함수에 그 결과를 전달하는 코드가 있다.
def output_result(result, log=None):
    if log is not None:
        log.debug('Got : %r', result)

def add(x, y):
    return x + y

if __name__ == '__main__':
    import logging
    from multiprocessing import Pool
    from functools import partial

    logging.basicConfig(level=logging.DEBUG)
    log = logging.getLogger('test')

    p = Pool()
    p.apply_async(add, (3, 4), callback=partial(output_result, log=log))

    p.close()
    p.join()
#  ※ apply_async() 로 콜백 함수를 지원할 때, partial() 을 사용해서 추가적인 로깅 인자를 넣었다.
#     multiprocessing 은 하나의 값으로 콜백 함수를 호출하게 된다.

#   - 유사한 예제로 네트워크 서버를 작성한다고 생각해 보자.
#     socketserver 모듈을 사용하면 상대적으로 편하게 작업할 수 있다.
from socketserver import StreamRequestHandler, TCPServer

class EchoHandler(StreamRequestHandler):
    # ack 는 키워드로만 넣을 수 있는 인자이다.
    # *args, **kwargs 는 그 외 일반적인 파라미터이다.
    def __init__(self, *args, ack, **kwargs):
        self.ack = ack
        super().__init__(*args, **kwargs)

    def handle(self):
        for line in self.rfile:
            self.wfile.write(self.ack + line)

serv = TCPServer(('', 15000), partial(EchoHandler, ack=b'RECEIVED:'))
serv.serve_forever()

#   - partial() 의 기능을 lambda 표현식으로 대신하기도 한다.
points.sort(key=lambda p: distance(pt, p))
p.apply_async(add, (3, 4), callback=lambda result: output_result(result, log=log))
serv = TCPServer(('', 15000), lambda *args, **kwargs: EchoHandler(*args, ack=b'RECEIVED:', **kwargs))
#   ※ 이 코드도 동작하기는 하지만, 가독성이 떨어지고 나중에 소스 코드를 읽는 사람이 헷갈릴 확률이 더 크다.
#      partial() 을 사용하면 조금 더 작성자의 의도를 파악하기 쉽다.


#  7.9 메소드가 하나인 클래스를 함수로 치환
#  ▣ 문제 : __init__() 외에 메소드가 하나뿐인 클래스가 있는데, 코드를 간결하게 만들기 위해 이를 하나의 함수로 바꾸고 싶다.
#  ▣ 해결 : 많은 경우 메소드가 하나뿐인 클래스는 클로저를 사용해서 함수로 바꿀 수 있다.
#            템플릿 스킴을 사용해서 URL 을 뽑아 내는, 다음 클래스를 예로 들어 보자.
from urllib.request import urlopen

class UrlTemplate:
    def __init__(self, template):
        self.template = template

    def open(self, **kwargs):
        return urlopen(self.template.format_map(kwargs))

yahoo = UrlTemplate('http://finance.yahoo.com/d/quotes.csv?s={names}&f={fields}')
for line in yahoo.open(names='IBM,AAPL,FB', fields='sl1c1v'):
    print(line.decode('utf-8'))

#   - 클로저를 사용해서 함수로 변경
def urltemplate(template):
    def opener(**kwargs):
        return urlopen(template.format_map(kwargs))
    return opener

yahoo = urltemplate('http://finance.yahoo.com/d/quotes.csv?s={names}&f={fields}')
for line in yahoo(names='IBM,AAPL,FB', fields='sl1c1v'):
    print(line.decode('utf-8'))

#  ▣ 토론 : 대개의 경우 메소드가 하나뿐인 클래스가 필요할 때는 추가적인 상태를 메소드에 저장할 때뿐이다.
#            예를 들어 UrlTemplate 클래스의 목적은 open() 메소드에서 사용하기 위해 template 값을 저장해 놓으려는 것뿐이다.
#            단순히 생각해서 클로저는 함수라고 말할 수 있지만 함수 내부에서 사용하는 변수의 추가적인 환경이 있다.
#            클로저의 주요 기능은 정의할 때의 환경을 기억한다는 것이다.
#            따라서 앞의 예제에서 opener() 함수가 template 인자의 값을 기억하고 추후 호출에 사용한다.


#  7.10 콜백 함수에 추가적 상태 넣기
#  ▣ 문제 : 콜백 함수를 사용하는 코드를 작성 중이다.(이벤트 핸들러, 완료 콜백 등)
#            이때 콜백 함수에 추가 상태를 넣고 내부 호출에 사용하고 싶다.
#  ▣ 해결 : 이 레시피는 많은 라이브러리와 프레임워크에서 찾을 수 있는 콜백 함수의 활용을 알아본다.
#            예제를 위해 콜백 함수를 호출하는 다음 함수를 정의한다.
def apply_async(func, args, *, callback):
    result = func(*args)
    callback(result)

def print_result(result):
    print('Got:', result)

def add(x, y):
    return x + y

apply_async(add, (2, 3), callback=print_result)
apply_async(add, ('hello', 'world'), callback=print_result)

#   - 콜백 함수에 추가 정보를 넣는 한 가지 방법은 하나의 함수 대신 바운드-메소드를 사용하는 것이다.
#  ★ 바운드-메소드 : 자동으로 개체 인스턴스가 첫 번째 인자로 전달되는 함수.
class ResultHandler:
    def __init__(self):
        self.sequence = 0

    def handler(self, result):
        self.sequence += 1
        print('[{}] Got: {}'.format(self.sequence, result))

r = ResultHandler()
apply_async(add, (2, 3), callback=r.handler)
apply_async(add, ('hello', 'world'), callback=r.handler)

#   - 클래스의 대안으로 클로저를 사용해서 상태를 저장해도 된다.
def make_handler():
    sequence = 0
    def handler(result):
        nonlocal sequence  # nonlocal : 특정 변수 이름에 할당할 때 스코프 탐색이 일어나야 함을 나타내는 키워드. (단, 모듈 수준 스코프까지 탐색할 수 없다.)
        sequence += 1
        print('[{}] Got: {}'.format(sequence, result))
    return handler

handler = make_handler()
apply_async(add, (2, 3), callback=handler)

#   - 코루틴(coroutine)을 사용할 수도 있다.
#  ★ coroutine : generator 와 반대되는 개념으로, generator 는 생산자이지만, coroutine 은 소비자 역할을 한다.
#                 따라서, yield 구문이 generator 와 반대로 입력으로 동작한다.
#                 coroutine 에서는 send('입력값') 메서드를 호출함으로써 입력을 수행한다.
def make_handler():
    sequence = 0
    while True:
        result = yield
        sequence += 1
        print('[{}] Got: {}'.format(sequence, result))

handler = make_handler()
next(handler)
apply_async(add, (2, 3), callback=handler.send)

#   - 마지막으로, 추가적인 인자와 partial function 어플리케이션으로 콜백에 상태를 넣을 수 있다.
class SequenceNo:
    def __init__(self):
        self.sequence = 0

def handler(result, seq):
    seq.sequence += 1
    print('[{}] Got: {}'.format(seq.sequence, result))
    
seq = SequenceNo()
from functools import partial
apply_async(add, (2, 3), callback=partial(handler, seq=seq))
apply_async(add, ('hello', 'world'), callback=partial(handler, seq=seq))

#  ▣ 토론 : 콜백 함수를 사용하는 프로그램은 엉망으로 망가질 위험 요소를 안고 있다.
#            한 가지 문제점은 콜백 실행으로 이끄는 초기 요청 코드와 콜백 함수가 끊어진다는 점이다.
#            결과적으로 요청을 한 곳과 처리하는 곳이 서로를 찾지 못하게 된다.
#            콜백 함수가 여러 단계에 걸쳐 실행을 계속하도록 만들기 위해서는 어떻게 관련 상태를 저장하고 불러올지 정해야 한다.
#            상태를 고정시키고 저장하는 방식에는 크게 두 가지가 있다.
#             1). 인스턴스에 상태를 저장.
#             2). 클로저에 저장.
#            두 가지 기술 중에서는 함수에 기반해서 만드는 클로저가 조금 더 가볍고 자연스럽다.
#            그리고 클로저는 자동으로 사용하는 변수를 고정시키기 때문에, 저장해야 하는 정확한 상태를 걱정하지 않아도 된다.
#            클로저를 사용하면 수정 가능한 변수를 조심해서 사용해야 한다.
#            앞에 나온 예제에서 nonlocal 선언은 sequence 변수가 콜백 내부에서 수정됨을 가리킨다.
#            이 선언이 없으면 에러가 발생한다.
#            코루틴을 사용하면 단순히 하나의 함수로 이루어져 있기 때문에 더 깔끔하고 nonlocal 선언 없이도 자유롭게 변수를 수정할 수 있다.
#            코루틴의 잠재적 단점은 파이썬의 기능으로 받아들여지지 않을 때가 있다는 것이다.
#            또한 코루틴을 사용하기 전에 next() 를 호출해야 한다는 점도 있다.

#   - partial() 을 대신해서 lambda 로 해결하는 방법
apply_async(add, (2, 3), callback=lambda result: handler(result, seq))


#  7.11 인라인 콜백 함수
#  ▣ 문제 : 콜백 함수를 사용하는 코드를 작성하는데, 크기가 작은 함수를 너무 많이 만들어 낼까 걱정이 된다.
#            코드가 좀 더 정상적인 절차적 단계를 거치도록 하고 싶다.
#  ▣ 해결 : 제너레이터와 코루틴을 사용하면 콜백 함수를 함수 내부에 넣을 수 있다.
def apply_async(func, args, *, callback):
    result = func(*args)
    callback(result)

from queue import Queue
from functools import wraps

class Async:
    def __init__(self, func, args):
        self.func = func
        self.args = args

def inlined_async(func):
    @wraps(func)  # 원래 함수의 속성들이 사라지는 것을 방지하기 위해 사용되는 데코레이터
    def wrapper(*args):
        f = func(*args)
        result_queue = Queue()
        result_queue.put(None)
        while True:
            result = result_queue.get()
            try:
                a = f.send(result)
                apply_async(a.func, a.args, callback=result_queue.put)
            except StopIteration:
                break
    return wrapper

def add(x, y):
    return x + y

@inlined_async
def test():
    r = yield Async(add, (2, 3))
    print(r)
    r = yield Async(add, ('hello', 'world'))
    print(r)
    for n in range(10):
        r = yield Async(add, (n, n))
        print(r)
    print('Goodbye')

test()

#  ▣ 토론 : 이번 레시피를 통해 콜백 함수, 제너레이터, 컨트롤 플로우를 얼마나 잘 이해하고 있는지 알 수 있다.
#   1. 콜백과 관련 있는 코드에서 현재 연산이 모두 연기되고 특정 포인트에서 재시작한다는 점이 포인트이다.
#      연산이 재시작하면 프로세싱을 위해 콜백이 실행된다.
#      apply_async() 함수는 콜백 실행의 필수 부위를 보여주는데, 사실 실제 프로그램에서는 스레드, 프로세스, 이벤트 핸들러 등이
#      연관되면서 훨씬 복잡할 것이다.
#      프로그램 실행이 연기되고 재시작하는 발상은 자연스럽게 제너레이터 함수의 실행 모델과 매핑된다.
#      특히 yield 연산은 제너레이터 함수가 값을 분출하고 연기하도록 한다.
#      뒤이어 __next__() 나 send() 메소드를 호출하면 다시 실행된다.
#   2. inline_async() 데코레이터는 yield 구문을 통해 제너레이터 함수를 하나씩 수행한다.
#      이를 위해 결과 큐를 만들고 최소로 None 을 넣은다음 순환문을 돌며 결과를 큐에서 꺼내 제너레이터로 보낸다.
#      여기서 다음 생성으로 넘어가고 Async 인스턴스를 받는다.
#      순환문은 함수와 인자를 보고 비동기 계산인 apply_async() 를 시작한다.
#      여기서 가장 주의해야 할 부분은 이 동작이 일반적인 콜백 함수를 사용해서 이루어진 것이 아니라, 콜백이 큐 put() 메소드에 설정되었다는 것이다.
#   3. 메인 루프는 즉시 최상단으로 돌아가고 큐의 get() 을 실행한다.
#      데이터가 있으면 put() 콜백이 넣은 결과일 것이다.
#      아무것도 없다면 연산이 멈추었고, 결과 값이 도착하길 기다리고 있는 것이다.
#      실행 결과는 apply_async() 함수의 정확한 구현에 따라 달라진다.

#   - 멀티프로세싱 라이브러리와 비동기 연산을 사용해 여러 프로세스에서 실행하는 방법
if __name__ == '__main__':
    import multiprocessing
    pool = multiprocessing.Pool()
    apply_async = pool.apply_async

    test()

#  ※ 복잡한 컨트롤 플로우를 제너레이터 함수에 숨기는 것은 표준 라이브러리와 서드파티 패키지에서 쉽게 찾을 수 있다.
#     예를 들어 contextlib 의 @contextmanager 데코레이터는 yield 구문을 통해 시작점과 종료점을 하나로 합치는 트릭을 수행한다.
#     유명한 패키지인 Twisted package 에도 유사한 콜백이 포함되어 있다.


#  7.12 클로저 내부에서 정의한 변수에 접근
#  ▣ 문제 : 클로저는 확장해서 내부 변수에 접근하고 수정하고 싶다.
#  ▣ 해결 : 일반적으로 클로저 내부 변수는 외부와 완전히 단절되어 있다.
#            하지만 접근 함수를 만들고 클로저에 함수 속성으로 붙이면 내부 변수에 접근할 수 있다.
def sample():
    n = 0
    # 클로저 함수
    def func():
        print('n=', n)

    # n에 대한 접근 메소드
    def get_n():
        return n

    def set_n(value):
        nonlocal n
        n = value

    # 함수 속성으로 클로저에 붙임
    func.get_n = get_n
    func.set_n = set_n
    return func

f = sample()
f()
f.set_n(10)
f()
print(f.get_n())

#  ▣ 토론 : 이번 레시피를 이루는 두 가지 주요 기능이 있다.
#            1. nonlocal 선언으로 내부 변수를 수정하는 함수를 작성하는 것.
#            2. 접근 메소드를 클로저 함수에 붙여 마치 인스턴스 메소드인 것처럼 동작하는 것.

#   - 클로저를 클래스의 인스턴스인 것처럼 동작하게 하는 방법.
import sys

class ClosureInstance:
    def __init__(self, locals=None):
        if locals is None:
            # sys._getframe(n) : n 단계 전의 프레임을 가져온다.
            # sys._getframe(n).f_code.co_name : n 단계 전의 프레임에서 함수 이름을 얻는다.
            # sys._getframe(n).f_locals : n 단계 전의 해당 프레임의 지역 변수를 얻는다.
            locals = sys._getframe(1).f_locals

        # 인스턴스 딕셔너리를 호출체로 갱신
        self.__dict__.update((key, value) for key, value in locals.items() if callable(value))

    def __len__(self):
        return self.__dict__['__len__']()

def Stack():
    items = []

    def push(item):
        items.append(item)

    def pop():
        return items.pop()

    def __len__():
        return len(items)

    return ClosureInstance()

s = Stack()
print(s)
s.push(10)
s.push(20)
s.push('Hello')
print(len(s))  # Stack 클로저 내의 __len__() 함수를 호출
print(s.pop())
print(s.pop())
print(s.pop())

#   - 위의 클로저를 이용한 코드가 일반 클래스 정의보다 실행 속도가 더 빠르다.
class Stack2:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def __len__(self):
        return len(self.items)

from timeit import timeit
s = Stack()
print(timeit('s.push(1);s.pop()', 'from __main__ import s'))
s = Stack2()
print(timeit('s.push(1);s.pop()', 'from __main__ import s'))