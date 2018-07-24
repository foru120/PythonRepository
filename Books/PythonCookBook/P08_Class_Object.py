# Chapter 8. 클래스와 객체
#  8.1 인스턴스의 문자열 표현식 변형
#  ▣ 문제 : 인스턴스를 출력하거나 볼 때 생성되는 결과물을 좀 더 보기 좋게 바꾸고 싶다.
#  ▣ 해결 : 인스턴스의 문자열 표현식을 바꾸려면 __str__() 와 __repr__() 메소드를 정의한다.
class Pair:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return 'Pair({0.x!r}, {0.y!r})'.format(self)  # 0.x : 인자 0의 x 속성, !r : __repr__()

    def __str__(self):
        return '({0.x!s}, {0.y!s})'.format(self)  # 0.x : 인자 0의 x 속성, !s : __str__()


# ※ __repr__() 메소드는 인스턴스의 코드 표현식을 반환하고, 일반적으로 인스턴스를 재생성할 때 입력하는 텍스트이다.
#     내장 repr() 함수는 인터프리터에서 값을 조사할 때와 마찬가지로 이 텍스트를 반환한다.
#     __str__() 메소드는 인스턴스를 문자열로 반환하고, str() 와 print() 함수가 출력하는 결과가 된다.
p = Pair(3, 4)
p
print(p)

print('p is {0!r}'.format(p))
print('p is {0}'.format(p))

#  ▣ 토론 : __repr__() 와 __str__() 를 정의하면 디버깅과 인스턴스 출력을 간소화한다.
#            __repr__() 는 eval(repr(x)) == x 와 같은 텍스트를 만드는 것이 표준이다.
#            이것을 원하지 않거나 불가능하다면 일반적으로 <와> 사이에 텍스트를 넣는다.
f = open('PythonCookBook/files/somefile.txt')
f  # <_io.TextIOWrapper name='PythonCookBook/files/somefile.txt' mode='r' encoding='cp949'>


#   - __str__() 를 정의하지 않으면 대신 __repr__() 의 결과물을 사용한다.
def __repr__(self):
    return 'Pair({0.x!r}, {0.y!r}'.format(self)  # 여기에서 0 은 인스턴스 self 를 의미


#   - % 연산자를 이용하는 방법
def __repr__(self):
    return 'Pair(%r, %r)' % (self.x, self.y)


#  8.2 문자열 서식화 조절
#  ▣ 문제 : format() 함수와 문자열 메소드로 사용자가 정의한 서식화를 지원하고 싶다.
#  ▣ 해결 : 문자열 서식화를 조절하고 싶으면 클래스에 __format__() 메소드를 정의한다.
_formats = {'ymd': '{d.year}-{d.month}-{d.day}',
            'mdy': '{d.month}/{d.day}/{d.year}',
            'dmy': '{d.day}/{d.month}/{d.year}'}


class Date:
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day

    def __format__(self, code):
        if code == '':
            code = 'ymd'
        fmt = _formats[code]
        return fmt.format(d=self)


d = Date(2012, 12, 21)
print(format(d))
print(format(d, 'mdy'))
print('The date is {:ymd}'.format(d))
print('The date is {:mdy}'.format(d))

#  ▣ 토론 : __format__() 메소드는 파이썬의 문자열 서식화 함수에 hook 를 제공한다.
#            서식화 코드의 해석은 모두 클래스 자체에 달려있다는 점이 중요하다.
#            따라서 코드에는 거의 모든 내용이 올 수 있다. 예를 들어 다음 datetime 모듈을 보자.
from datetime import date

d = date(2012, 12, 21)
print(format(d))
print(format(d, '%A, %B, %d, %Y'))
print('The end is {:%d %b %Y}. Goodbye'.format(d))

#  8.3 객체의 콘텍스트 관리 프로토콜 지원
#  ▣ 문제 : 객체가 콘텍스트 관리 프로토콜(with 구문)을 지원하게 만들고 싶다.
#  ▣ 해결 : 객체와 with 구문을 함께 사용할 수 있게 만들려면, __enter__() 와 __exit__() 메소드를 구현해야 한다.
from socket import socket, AF_INET, SOCK_STREAM
from functools import partial


class LazyConnection:
    def __init__(self, address, family=AF_INET, type=SOCK_STREAM):
        self.address = address
        self.family = family
        self.type = type
        self.sock = None

    def __enter__(self):
        if self.sock is not None:
            raise RuntimeError('Already connected')
        self.sock = socket(self.family, self.type)
        self.sock.connect(self.address)
        return self.sock

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sock.close()
        self.sock = None


conn = LazyConnection(('www.python.org', 80))

with conn as s:  # 컨텍스트 관리 프로토콜
    # conn.__enter__() 실행: 연결
    s.send(b'GET /index.html HTTP/1.0\r\n')
    s.send(b'Host: www.python.org\r\n')
    s.send(b'\r\n')
    resp = b''.join(iter(partial(s.recv, 8192), b''))
    # conn.__exit__() 실행: 연결 종료

#  ▣ 토론 : 컨텍스트 매니저를 작성할 때 중요한 원리는 with 구문을 사용하여 정의된 블럭을 감싸는 코드를 작성한다는 것이다.
#           처음으로 with 를 만나면 __enter__() 메소드가 호출된다.
#           __enter__() 의 반환 값은 as 로 나타낸 변수에 위치시킨다.
#           그 후에 with 의 내부 명령어를 실행하고 마지막으로 __exit__() 메소드로 소거 작업을 한다.
#           이 흐름은 with 문 내부에서 어떤 일이 발생하는지와 상관 없이 일어나고, 예외가 발생한다해도 변함이 없다.
#           사실 __exit__() 메소드의 세 가지 인자에 예외 타입, 값, 트레이스백(traceback)이 들어 있다.
#           __exit__() 메소드는 예외 정보를 고르거나 아무 일도 하지 않고 None 을 반환하며 무시하는 방식을 선택할 수 있다.
#           만약 __exit__() 가 True 를 반환하면 예외를 없애고 아무 일도 일어나지 않았던 것처럼 with 블록 다음의 프로그램을
#           계속해서 실행한다.

#   - 이번 레시피에서 with 문을 여러 번 써서 중첩된 연결을 LazyConnection 클래스가 허용하는지 여부가 미묘한 측면이다.
#     앞에 나온 대로 한 번에 하나의 소켓 연결만 허용되고 소켓을 사용 중에 중복된 with 문이 나타나면 예외가 발생한다.
#     구현법을 조금만 바꾸면 이 제한을 피해 갈 수 있다.
from socket import socket, AF_INET, SOCK_STREAM


class LazyConnection:
    # AF_INET :  Internet Protocol(IP) socket 을 요청.
    # SOCK_STREAM : TCP 소켓 생성.
    def __init__(self, address, family=AF_INET, type=SOCK_STREAM):
        self.address = address
        self.family = family
        self.type = type
        self.connections = []

    def __enter__(self):
        sock = socket(self.family, self.type)
        sock.connect(self.address)
        self.connections.append(sock)
        return sock

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connections.pop().close()


from functools import partial

conn = LazyConnection(('www.python.org', 80))
with conn as s1:
    with conn as s2:
        pass
    pass


# - 두 번째 버전에서 LazyConnection 클래스는 연결을 위한 팩토리 역할을 한다.
#     내부적으로 스택을 위해 리스트를 사용했다. __enter__() 가 실행될 때마다, 새로운 연결을 만들고 스택에 추가한다.
#     __exit__() 메소드는 단순히 스택에서 마지막 연결을 꺼내고 닫는다.
#     사소한 문제지만 이로 인해서 중첩 with 구문으로 연결을 여러 개 생성할 수 있다.


#  8.4 인스턴스를 많이 생성할 때 메모리 절약
#  ▣ 문제 : 프로그램에서 많은 인스턴스를 생성하고 메모리를 많이 소비한다.
#  ▣ 해결 : 간단한 자료 구조 역할을 하는 클래스의 경우 __slots__ 속성을 클래스 정의에 추가하면 메모리 사용을 상당히 많이 절약할 수 있다.
class Date:
    __slots__ = ['year', 'month', 'day']

    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day


# - __slots__ 을 정의하면 파이썬은 인스턴스에서 훨씬 더 압축된 내부 표현식을 사용한다.
#     인스턴스마다 딕셔너리를 구성하지 않고 튜플이나 리스트 같은 부피가 작은 고정 배열로 인스턴스가 만들어진다.
#     __slots__ 명시자에 리스팅된 속성 이름은 내부적으로 이 배열의 특정 인덱스에 매핑된다.
#     슬롯을 사용하는 데서 발생하는 부작용은 인스턴스에 새로운 속성을 추가할 수 없다는 점이다.
#     __slots__ 명시자에 나열한 속성만 사용할 수 있다는 제약이 생긴다.

#  ▣ 토론 : 슬롯을 사용해서 절약하는 메모리는 속성의 숫자와 타입에 따라 다르다.
#           하지만, 일반적으로 그 데이터를 딕셔너리에 저장할 때의 메모리 사용과 비교할 만하다.
#           Date 인스턴스 하나를 슬롯 없이 저장하면, 64비트 파이썬에서 428 바이트를 소비한다.
#           만약 슬롯을 정의하면 메모리 사용은 156 바이트로 떨어진다.
#           하지만, 슬롯을 정의한 클래스는 다중 상속과 같은 특정 기능을 지원하지 않는다.
#           따라서 프로그램에서 자주 사용하는 자료 구조에만 슬롯 사용을 고려하는 것이 좋다.


#  8.5 클래스 이름의 캡슐화
#  ▣ 문제 : 클래스 인스턴스의 private 데이터를 캡슐화하고 싶지만, 파이썬에는 접근 제어 기능이 부족하다.
#  ▣ 해결 : 파이썬 프로그래머들은 언어의 기능에 의존하기보다는 데이터나 메소드의 이름에 특정 규칙을 사용하여서 의도를 나타낸다.
#           첫 번째 규칙은 밑줄(_)로 시작하는 모든 이름은 내부 구현에서만 사용하도록 가정하는 것이다.
class A:
    def __init__(self):
        self._internal = 0  # 내부 속성
        self.public = 1  # 외부 속성

    def public_method(self):
        '''
        A public method
        '''

    def _internal_method(self):
        '''
        A internal method
        '''


# - 클래스 정의에 밑줄 두 개(__)로 시작하는 이름이 나오기도 한다.
class B:
    def __init__(self):
        self.__private = 0

    def __private_method(self):
        '''
        A private method
        '''

    def public_method(self):
        '''
        self.__private_method()
        '''


# - 이름 앞에 밑줄을 두 개 붙이면 이름이 다른 것으로 변한다. 더 구체적으로, 앞에 나온 클래스의 private 속성은 _B__private 과
#     _B__private_method 로 이름이 변한다.
#     이러한 이름의 변화는 속성은 속성을 통해 오버라이드 할 수 없다는 것이다.
class C:
    def __init__(self):
        super().__init__()
        self.__private = 1  # B.__private 를 오버라이드하지 않는다.

    # B.__private_method() 를 오버라이드하지 않는다.
    def __private_method(self):
        pass


# - 여기서 __private 과 __private_method 의 이름은 _C__private 과 _C__private_method 로 변하기 때문에 B 클래스의 이름과
#     겹치지 않는다.

#  ▣ 토론 : 프라이빗 속성에 대한 규칙이 두 가지 존재해서, 이 중 어떤 스타일을 써야할지 의문이 들 수 있다.
#           대개의 경우 공용이 아닌 이름은 밑줄을 하나만 붙여야 한다.
#           하지만 코드가 서브클래싱을 사용할 것이고 서브클래스에서 숨겨야 할 내부 속성이 있다면 밑줄을 두 개 붙인다.
#           그리고 예약해 둔 단어 이름과 충돌하는 변수를 정의하고 싶을 때가 있다.
#           이런 경우 이름 뒤에 밑줄을 하나 붙인다.
lambda_ = 2.0  # lambda 키워드와의 충돌을 피하기 위해 밑줄을 붙인다.


#   - 밑줄을 변수 이름 앞에 붙이지 않는 이유는 내부적으로 사용하는 의도와의 혼동을 피하기 위해서이다.
#     변수 이름 뒤에 밑줄을 하나 붙여서 이 문제를 해결한다.


#  8.6 관리 속성 만들기
#  ▣ 문제 : 인스턴스 속성을 얻거나 설정할 때 추가적인 처리(타입 체크, 검증 등)를 하고 싶다.
#  ▣ 해결 : 속성에 대한 접근을 조절하고 싶으면 "프로퍼티(property)"로 정의하면 된다.
class Person:
    def __init__(self, first_name):
        # self.first_name = first_name 은 초기화시에 setter 함수를 사용해서 _first_name 에 입력한다.
        # self._first_name = first_name 은 초기화시에 직접 _first_name 에 입력한다.
        self.first_name = first_name

    # getter 함수
    @property
    def first_name(self):
        return self._first_name

    # setter 함수
    @first_name.setter
    def first_name(self, value):
        if not isinstance(value, str):
            raise TypeError('Expected a string')
        self._first_name = value

    # deleter 함수
    @first_name.deleter
    def first_name(self):
        raise AttributeError("Can't delete attribute")


a = Person('Guido')
print(a.first_name)
a.first_name = 4
del a.first_name


#   - 이미 존재하는 get 과 set 메소드로 프로퍼티를 정의하는 방법.
class Person:
    def __init__(self, first_name):
        self.set_first_name(first_name)

    # getter
    def get_first_name(self):
        return self._first_name

    # setter
    def set_first_name(self, value):
        if not isinstance(value, str):
            raise TypeError('Expected a string')
        self._first_name = value

    # deleter
    def del_first_name(self):
        raise AttributeError("Can't delete attribute")

    # setter/getter 메소드로 프로퍼티 만들기
    name = property(get_first_name, set_first_name, del_first_name)


# ▣ 토론 : 사실 프로퍼티 속성은 함께 묶여 있는 메소드 컬렉션이다.
#           프로퍼티가 있는 클래스를 조사해 보면 프로퍼티 자체의 fget, fset, fdel 속성에서 로우 메소드를 찾을 수 있다.
person = Person('Guido')
print(Person.first_name.fget)
print(Person.first_name.fset)
print(Person.first_name.fdel)


#   - 프로퍼티는 속성에 추가적인 처리가 필요할 때만 사용해야 한다.
class Person:
    def __init__(self, first_name):
        self.first_name = first_name

    @property
    def first_name(self):
        return self._first_name

    @first_name.setter
    def first_name(self, value):
        self._first_name = value


# - 프로퍼티는 계산한 속성을 정의할 때 사용하기도 한다.
import math


class Circle:
    def __init__(self, radius):
        self.radius = radius

    @property
    def area(self):
        return math.pi * self.radius ** 2

    @property
    def perimeter(self):
        return 2 * math.pi * self.radius


c = Circle(4.0)
print(c.radius)
print(c.area)
print(c.perimeter)

#   - 프로퍼티로 우아한 프로그래밍 인터페이스를 얻을 수 있지만, 게터와 세터 함수를 직접 사용하는 것도 가능하다.
p = Person('Guido')
print(p.get_first_name())
p.set_first_name('Larry')


#   - 프로퍼티 정의를 반복적으로 사용하는 파이썬 코드를 작성하지 않도록 주의 하자.
class Person:
    def __init__(self, first_name, last_name):
        self.first_name = first_name
        self.last_name = last_name

    @property
    def first_name(self):
        return self._first_name

    @first_name.setter
    def first_name(self, value):
        if not isinstance(value, str):
            raise TypeError('Expected a string')
        self._first_name = value

    @property
    def last_name(self):
        return self._last_name

    @last_name.setter
    def last_name(self, value):
        if not isinstance(value, str):
            raise TypeError('Expected a string')
        self._last_name = value


# 8.7 부모 클래스의 메소드 호출
#  ▣ 문제 : 오버라이드된 서브클래스 메소드가 아닌 부모 클래스에 있는 메소드를 호출하고 싶다.
#  ▣ 해결 : 부모의 메소드를 호출하려면 super() 함수를 사용한다.
class A:
    def spam(self):
        print('A.spam')


class B(A):
    def spam(self):
        print('B.spam')
        super().spam()  # 부모의 spam() 호출


# ※ super() 는 일반적으로 __init__() 메소드에서 부모를 제대로 초기화하기 위해 사용한다.

#   - 파이썬의 특별 메소드를 오버라이드한 코드에서 super() 를 사용하기도 한다.
class Proxy:
    def __init__(self, obj):
        self._obj = obj

    # 내부 obj 를 위해 델리게이트(delegate) 속성 찾기
    def __getattr__(self, item):
        return getattr(self._obj, item)

    # 델리게이트(delegate) 속성 할당
    def __setattr__(self, key, value):
        if key.startswith('_'):
            super().__setattr__(key, value)
        else:
            setattr(self._obj, key, value)


# ▣ 토론 : super() 함수를 올바르게 사용하기는 무척 어렵다.
#           때때로 부모 클래스 메소드를 직접 호출하기 위해 다음과 같이 작성한 코드를 볼 수 있다.
class Base:
    def __init__(self):
        print('Base.__init__')


class A(Base):
    def __init__(self):
        Base.__init__(self)
        print('A.__init__')


class B(Base):
    def __init__(self):
        Base.__init__(self)
        print('B.__init__')


class C(A, B):
    def __init__(self):
        A.__init__(self)
        B.__init__(self)
        print('C.__init__')


c = C()


#   - 위의 코드는 Base 클래스를 두 번 호출한다. super() 를 사용하여 코드를 수정한다면 올바르게 동작한다.
class Base:
    def __init__(self):
        print('Base.__init__')


class A(Base):
    def __init__(self):
        super().__init__()
        print('A.__init__')


class B(Base):
    def __init__(self):
        super().__init__()
        print('B.__init__')


class C(A, B):
    def __init__(self):
        super().__init__()
        print('C.__init__')


c = C()

#   - 클래스를 정의할 때마다 파이썬은 메소드 처리 순서(method resolution order, MRO) 리스트를 계산한다.
#     MRO 리스트는 모든 베이스 클래스를 단순히 순차적으로 나열한 리스트이다.
print(C.__mro__)


#  ※ 부모 클래스의 MRO 를 다음 세 가지 제약 조건하에서 합병 정렬한다.
#   - 자식 클래스를 부모보다 먼저 확인한다.
#   - 부모 클래스가 둘 이상이면 리스팅한 순서대로 확인한다. C(A, B) : A -> B
#   - 유효한 후보가 두 가지 있으면, 첫 번째 부모 클래스부터 선택한다.

#   - super() 의 놀라운 측면 중 하나는 이 함수가 MRO 의 바로 위에 있는 부모에 직접 접근하지 않을 때도 있고, 직접적인 부모 클래스가
#     없을 때도 super() 를 사용할 수 있다는 점이다.
class A:
    def spam(self):
        print('A.spam')
        super().spam()


a = A()
a.spam()


#   - 위의 코드는 AttributeError: 'super' object has no attribute 'spam' 에러 메시지가 발생한다.
class B:
    def spam(self):
        print('B.spam')


class C(A, B):
    pass


c = C()
c.spam()
print(C.__mro__)


#  8.8 서브클래스에서 프로퍼티 확장
#  ▣ 문제 : 서브클래스에서, 부모 클래스에 정의한 프로퍼티의 기능을 확장하고 싶다.
#  ▣ 해결 : 프로퍼티를 정의하는 다음 코드를 보자.
class Person:
    def __init__(self, name):
        self.name = name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise TypeError('Expected a string')
        self._name = value

    @name.deleter
    def name(self):
        raise AttributeError("Can't delete attribute")


class SubPerson(Person):
    @property
    def name(self):
        print('Getting name')
        return super().name

    @name.setter
    def name(self, value):
        print('Setting name to', value)
        super(SubPerson, SubPerson).name.__set__(self, value)  # super(클래스, 서브클래스 or 인스턴스)

    @name.deleter
    def name(self):
        print('Deleting name')
        super(SubPerson, SubPerson).name.__delete__(self)


s = SubPerson('Guido')
s.name
s.name = 'Larry'
s.name = 42


#   - 프로퍼티의 메소드 하나를 확장하고 싶은 경우
class SubPerson(Person):
    @Person.name.getter
    def name(self):
        print('Getting name')
        return super().name


# - 세터 하나만 확장하는 경우
class SubPerson(Person):
    @Person.name.setter
    def name(self, value):
        print('Setting name to', value)
        super(SubPerson, SubPerson).name.__set__(self, value)


# ▣ 토론 : 서브클래스의 프로퍼티를 확장하면 프로퍼티가 하나의 메소드가 아닌 게터, 세터, 딜리터 메소드의 컬렉션으로 정의되었다는 사실로 인해
#           자잘한 문제가 발생한다.
#           따라서 프로퍼티를 확장할 때 모든 메소드를 다시 정의할지, 메소드 하나만 다시 정의할지 결정해야 한다.

#   - 메소드 중 하나만 재정의하려면 @property 자체만 사용하는 것으로는 충분하지 않다.
class SubPerson(Person):
    @property
    def name(self):  # 동작하지 않음
        print('Getting name')
        return super().name


s = SubPerson('Guido')  # AttributeError: can't set attribute


#   - 디스크립터를 확장하는데 사용되는 방법.

# 디스크립터
class String:
    def __init__(self, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__[self.name]

    def __set__(self, instance, value):
        if not isinstance(instance, str):
            raise TypeError('Expected a string')
        instance.__dict__[self.name] = value


class Person:
    name = String('name')

    def __init__(self, name):
        self.name = name


class SubPerson(Person):
    @property
    def name(self):
        print('Getting name')
        return super().name

    @name.setter
    def name(self, value):
        print('Setting name to', value)
        super(SubPerson, SubPerson).name.__set__(self, value)

    @name.deleter
    def name(self):
        print('Deleting name')
        super(SubPerson, SubPerson).name.__delete__(self)


s = SubPerson('Guido')


#  8.9 새로운 클래스나 인스턴스 속성 만들기
#  ▣ 문제 : 타입 확인 등과 같이 추가적 기능을 가진 새로운 종류의 인스턴스 속성을 만들고 싶다.
#  ▣ 해결 : 완전히 새로운 종류의 인스턴스 속성을 만들려면, 그 기능을 디스크립터 클래스 형태로 정의해야 한다.
class Integer:
    def __init__(self, name):
        self.name = name  # 실제 데이터를 인스턴스 딕셔너리에 저장할 때 사용하는 딕셔너리 키를 가지는 변수.

    def __get__(self, instance, owner):
        if instance is None:
            return self
        else:
            return instance.__dict__[self.name]

    def __set__(self, instance, value):
        if not isinstance(value, int):
            raise TypeError('Expected an int')
        instance.__dict__[self.name] = value

    def __delete__(self, instance):
        del instance.__dict__[self.name]


# - 디스크립터는 세 가지 중요한 속성 접근 명령을 특별 메소드 __get__(), __set__(), __delete__() 형식으로 구현한 클래스이다.
#     이 메소드는 인스턴스를 입력으로 받는다. 그리고 인스턴스의 기반 딕셔너리는 속성으로 만들어진다.
#     디스크립터를 사용하려면, 디스크립터의 인스턴스는 클래스 정의에 클래스 변수로 들어가야 한다.
class Point:
    x = Integer('x')
    y = Integer('y')

    def __init__(self, x, y):
        self.x = x
        self.y = y


p = Point(2, 3)
p.x
p.y = 5
p.x = 2.3  # TypeError: Expected an int


#   ※ 입력으로, 디스크립터의 모든 메소드는 가공 중인 인스턴스를 받는다.
#      요청 받은 작업을 수행하기 위해서, 인스턴스 딕셔너리 역시 적절히 처리된다.
#      디스크립터의 self.name 속성은 실제 데이터를 인스턴스 딕셔너리에 저장할 때 사용하는 딕셔너리 키를 가지고 있다.

#  ▣ 토론 : 디스크립터는 파이썬 클래스 기능에 __slots__, @classmethod, @staticmethod, @property 와 같이 멋진 도구를 제공한다.
#            디스크립터를 정의하면 get, set, delete 와 같이 중요한 인스턴스 연산을 아주 하위 레벨에서 얻고 어떻게 동작할지도 입맛대로 바꿀 수 있다.
#            따라서 고급 라이브러리와 프레임워크를 작성하는 프로그래머에게 매우 중요한 도구가 된다.
#            단, 인스턴스 기반이 아닌 클래스 레벨에서만 정의가 가능하다.
#            따라서 다음과 같은 코드는 동작하지 않는다.
class Point:
    def __init__(self, x, y):
        self.x = Integer('x')
        self.y = Integer('y')
        self.x = x
        self.y = y


# - 또한 __get__() 메소드를 구현하는 것도 보기보다 간단하지 않다.
class Integer:
    def __get__(self, instance, owner):
        if instance is None:
            return self
        else:
            return instance.__dict__[self.name]


# - __get__() 이 조금 복잡해 보이는 이유는 인스턴스 변수와 클래스 변수를 구분해야 하기 때문이다.
#      만약 디스크립터를 클래스 변수로 접근하면 instance 인자가 None 이 된다.
#      이 경우에는 단순히 디스크립터 자신을 반환하는 것이 일반적이다.
p = Point(2, 3)
p.x  # Point.x.__get__(p, Point) 호출
Point.x  # Point.x.__get__(None, Point) 호출


#   - 다음은 클래스 데코레이터를 사용하는 디스크립터 기반의 고급 코드이다.
class Typed:
    def __init__(self, name, expected_type):
        self.name = name
        self.expected_type = expected_type

    def __get__(self, instance, owner):
        if instance is None:
            return self
        else:
            return instance.__dict__[self.name]

    def __set__(self, instance, value):
        if not isinstance(value, self.expected_type):
            raise TypeError('Expected ' + str(self.expected_type))
        instance.__dict__[self.name] = value

    def __delete__(self, instance):
        del instance.__dict__[self.name]


def typeassert(**kwargs):  # 선택한 속성에 적용되는 클래스 데코레이터
    def decorate(cls):
        for name, expected_type in kwargs.items():
            setattr(cls, name, Typed(name, expected_type))
        return cls

    return decorate


@typeassert(name=str, shares=int, price=float)
class Stock:
    def __init__(self, name, shares, price):
        self.name = name
        self.shares = shares
        self.price = price


# ※ 특정 클래스의 한 속성에 대한 접근을 제어하기 위해서 디스크립터를 사용하지 말아야 한다.
#      이런 경우에는 프로퍼티를 사용해야한다. 디스크립터는 코드 재사용이 빈번히 발생하는 상황에 더 유용하다.


#  8.10 게으른 계산을 하는 프로퍼티 사용
#  ▣ 문제 : 읽기 전용 속성을 프로퍼티로 정의하고, 이 속성에 접근할 때만 계산하도록 하고 싶다.
#            하지만 한 번 접근하고 나면 이 값을 캐시에 놓고 다음 번에 접근할 때에는 다시 계산하지 않도록 하고 싶다.
#  ▣ 해결 : 게으른 속성을 효율적으로 정의하기 위해서는 다음과 같이 디스크립터 클래스를 사용한다.
class lazyproperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            setattr(instance, self.func.__name__, value)
            return value


import math


class Circle:
    def __init__(self, radius):
        self.radius = radius

    @lazyproperty
    def area(self):
        print('Computing area')
        return math.pi * self.radius ** 2

    @lazyproperty
    def perimeter(self):
        print('Computing perimeter')
        return 2 * math.pi * self.radius


c = Circle(4.0)
c.radius
c.area
c.area
c.perimeter
c.perimeter

#  ▣ 토론 : 대개의 경우 게으르게 계산한 속성은 성능 향상을 위해 사용한다.
#            예를 들어 실제로 특정 값을 사용하기 전까지 계산하지 않도록 하는 것이다.
c = Circle(4.0)
print(vars(c))
c.area
print(vars(c))
c.area
del c.area
print(vars(c))
c.area
print(vars(c))
#   - 이번 레시피에서 불리한 점 한 가지는 계산한 값을 생성한 후에 수정할 수 있다는 것이다.
c.area
c.area = 25
c.area


#   - 이 부분이 걱정이 된다면 조금 덜 효율적인 구현을 할 수 있다.
def lazyproperty(func):
    name = '_lazy_' + func.__name__

    @property
    def lazy(self):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            value = func(self)
            setattr(self, name, value)
            return value

    return lazy


c = Circle(4.0)
c.area
c.area
c.area = 25


#   ※ 위의 버전은 값을 얻기 위해서 항상 프로퍼티의 getter 함수를 사용해야 한다는 불편함이 생긴다.


#  8.11 자료 구조 초기화 단순화하기
#  ▣ 문제 : 자료 구조로 사용하는 클래스를 작성하고 있는데, 반복적으로 비슷한 __init__() 함수를 작성하기에 지쳐 간다.
#  ▣ 해결 : 자료 구조의 초기화는 베이스 클래스의 __init__() 함수를 정의하는 식으로 단순화할 수 있다.
class Structure:
    _fields = []

    def __init__(self, *args):
        if len(args) != len(self._fields):
            raise TypeError('Expected {} arguments'.format(len(self._fields)))

        for name, value in zip(self._fields, args):
            setattr(self, name, value)


if __name__ == '__main__':
    class Stock(Structure):
        _fields = ['name', 'shares', 'price']


    class Point(Structure):
        _fields = ['x', 'y']


    class Circle(Structure):
        _fields = ['radius']

        def area(self):
            return math.pi * self.radius ** 2

s = Stock('ACME', 50, 91.1)
p = Point(2, 3)
c = Circle(4.5)
s2 = Stock('ACME', 50)


#   - 키워드 매개변수를 지원하기로 결정했다면 사용할 수 있는 디자인 옵션이 몇 가지 있다.
#     그 중 한 가지는 키워드 매개변수를 매핑해서 _fields 에 명시된 속성 이름에만 일치하도록 만드는 것이다.
class Structure:
    _fields = []

    def __init__(self, *args, **kwargs):
        if len(args) > len(self._fields):
            raise TypeError('Expected {} arguments'.format(len(self._fields)))

        for name, value in zip(self._fields, args):
            setattr(self, name, value)

        for name in self._fields[len(args):]:
            setattr(self, name, kwargs.pop(name))

        if kwargs:
            raise TypeError('Invalid argument(s): {}'.format(','.join(kwargs)))


if __name__ == '__main__':
    class Stock(Structure):
        _fields = ['name', 'shares', 'price']


    s1 = Stock('ACME', 50, 91.1)
    s2 = Stock('ACME', 50, price=91.1)
    s3 = Stock('ACME', shares=50, price=91.1)


# - 혹은 _fields 에 명시되지 않은 구조에 추가적인 속성을 추가하는 수단으로 키워드 매개변수를 사용할 수 있다.
class Structure:
    _fields = []

    def __init__(self, *args, **kwargs):
        if len(args) > len(self._fields):
            raise TypeError('Expected {} arguments'.format(len(self._fields)))

        for name, value in zip(self._fields, args):
            setattr(self, name, value)

        extra_args = kwargs.keys() - self._fields
        for name in extra_args:
            setattr(self, name, kwargs.pop(name))

        if kwargs:
            raise TypeError('Invalid argument(s): {}'.format(','.join(kwargs)))


if __name__ == '__main__':
    class Stock(Structure):
        _fields = ['name', 'shares', 'price']


    s1 = Stock('ACME', 50, 91.1)
    s2 = Stock('ACME', 50, 91.1, date='8/2/2012')


# ▣ 토론 : 제너럴한 목적으로 __init__() 메소드를 정의하는 기술은 규모가 작은 자료 구조를 대량으로 만드는 프로그램에 유용하다.
#            이 기술은 다음과 같이 일일이 __init__() 메소드를 작성하는 것에 비해 훨씬 코드의 양을 줄여 준다.

#   - 프레임 핵을 사용하면 인스턴스 변수를 자동으로 초기화할 수 있다.
def init_fromlocals(self):
    import sys
    locs = sys._getframe(1).f_locals
    for k, v in locs.items():
        if k != 'self':
            setattr(self, k, v)


class Stock:
    def __init__(self, name, shares, price):
        init_fromlocals(self)


# 8.12 인터페이스, 추상 베이스 클래스 정의
#  ▣ 문제 : 인터페이스나 추상 베이스 클래스 역할을 하는 클래스를 정의하고 싶다.
#            그리고 이 클래스는 타입 확인을 하고 특정 메소드가 서브 클래스에 구현되었는지 보장한다.
#  ▣ 해결 : 추상 베이스 클래스를 정의하려면 abc 모듈을 사용한다.
from abc import ABCMeta, abstractmethod


class IStream(metaclass=ABCMeta):
    @abstractmethod
    def read(self, maxbytes=-1):
        pass

    @abstractmethod
    def write(self, data):
        pass


# - 추상 베이스 클래스의 주요 기능은 직접 인스턴스화 할 수 없다는 점이다.
a = IStream()  # TypeError: Can't instantiate abstract class IStream with abstract methods read, write


class SocketStream(IStream):  # 추상 클래스를 상속받은 클래스는 추상 메소드를 반드시 오버라이딩 해야 한다.
    def read(self, maxbytes=-1):
        pass

    def write(self, data):
        pass


# - 추상 베이스 클래스는 특정 프로그래밍 인터페이스를 강요하고 싶을 때 주로 사용한다.
#     예를 들어 IStream 베이스 클래스는 데이터를 읽거나 쓰는 인터페이스에 대한 상위 레벨 스펙으로 볼 수 있다.
#     이런 인터페이스를 명시적으로 확인하는 코드는 다음과 같이 작성한다.
def serialize(obj, stream):
    if not isinstance(stream, IStream):
        raise TypeError('Expected an IStream')


# - 이런 타입 확인 코드는 서브클래싱과 추상 베이스 클래스에서만 동작할 것 같지만, ABC 는 다른 클래스가 특정 인터페이스를
#     구현했는지 확인하도록 허용한다.
import io

# 내장 I/O 클래스를 우리의 인터페이스를 지원하도록 등록
IStream.register(io.IOBase)

f = open('PythonCookBook/files/somefile.txt')
print(isinstance(f, IStream))

#   - @abstractmethod 를 스태틱 메소드, 클래스 메소드, 프로퍼티에도 적용할 수도 있다.
#     단지 함수 정의 직전에 @abstractmethod 를 적용해야 한다는 점만 기억하자.
from abc import ABCMeta, abstractmethod


class A(metaclass=ABCMeta):
    @property
    @abstractmethod
    def name(self):
        pass

    @name.setter
    @abstractmethod
    def name(self, value):
        pass

    @classmethod
    @abstractmethod
    def method1(cls):
        pass

    @staticmethod
    @abstractmethod
    def method2():
        pass


# ▣ 토론 : 파이썬 표준 라이브러리에서 추상 베이스 클래스(ABC)를 사용하는 경우가 많다.
#            collections 모듈은 콘테이너 이터레이터(sequence, mapping, set 등)에 ABC 를 정의하고 있고,
#            numbers 라이브러리는 숫자 관련 객체(integer, float, rational 등)에 ABC 를 정의하며, io 라이브러리는 I/O 처리에 ABC 를 정의한다.
import collections

x = []

if isinstance(x, collections.Sequence):
    pass

if isinstance(x, collections.Iterable):
    pass

if isinstance(x, collections.Sized):
    pass

if isinstance(x, collections.Mapping):
    pass

# - 특정 라이브러리 모듈은 ABC 를 활용하고 있지 않다.
from decimal import Decimal
import numbers

x = Decimal('3.4')
isinstance(x, numbers.Real)  # False


#   ※ 3.4는 엄밀히 따지면 실수이지만, 의도하지 않은 부동 소수점 숫자와 소수의 혼합 사용을 방지하기 위해 False 를 반환한다.
#      ABC 에 타입 확인 기능이 있지만, 프로그램에서 남용하지 않는 것이 좋다.
#      파이썬은 뛰어난 유연성을 자랑하는 역동적인 언어이다.
#      코드에서 매번 타입을 확인하려 들면 필요 이상으로 코드가 복잡해진다. 파이썬의 유연성을 받아들이도록 하자.


#  8.13 데이터 모델 혹은 타입 시스템 구현
#  ▣ 문제 : 여러 종류의 자료 구조를 정의하고 싶다. 이때 특정 값에 제약을 걸어 원하는 속성이 할당되도록 하고 싶다.
#  ▣ 해결 : 이번 문제의 경우, 기본적으로 특정 인스턴스 속성의 값을 설정할 때 확인을 하는 동작을 구현해야 한다.
#            이렇게 하려면 속성을 설정하는 부분을 속성 하나 단위로 커스터마이즈 해야 하고, 디스크립터로 해결 가능하다.
#            다음 코드는 디스크립터로 시스템 타입과 값 확인 프레임워크를 구현하는 방법을 보여준다.

# 베이스 클래스, 디스크립터로 값을 설정
class Descriptor:
    def __init__(self, name=None, **opts):
        self.name = name
        for key, value in opts.items():  # 처음 객체 생성시 해당 키 값으로 객체의 속성들을 초기화
            setattr(self, key, value)

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value


# 타입을 강제하기 위한 디스크립터
class Typed(Descriptor):
    expected_type = type(None)

    def __set__(self, instance, value):
        if not isinstance(value, self.expected_type):
            raise TypeError('expected ' + str(self.expected_type))
        super().__set__(instance, value)


# 값을 강제하기 위한 디스크립터
class Unsigned(Descriptor):
    def __set__(self, instance, value):
        if value < 0:
            raise ValueError('Expected >= 0')
        super().__set__(instance, value)


class MaxSized(Descriptor):
    def __init__(self, name=None, **opts):
        if 'size' not in opts:
            raise TypeError('missing size option')
        super().__init__(name, **opts)

    def __set__(self, instance, value):
        if len(value) >= self.size:
            raise ValueError('size must be < ' + str(self.size))
        super().__set__(instance, value)


# - 앞에 나온 클래스는 데이터 모델이나 타입 시스템을 만들 때 기반으로 사용하는 빌딩 블록으로 봐야 한다.
#     이제 서로 다른 데이터를 구현하는 코드를 보자.
class Integer(Typed):
    expected_type = int


class UnsignedInteger(Integer, Unsigned):
    pass


class Float(Typed):
    expected_type = float


class UnsignedFloat(Float, Unsigned):
    pass


class String(Typed):
    expected_type = str


class SizedString(String, MaxSized):
    pass


# - 타입 객체를 사용해서 다음과 같은 클래스를 정의할 수 있다.
class Stock:
    name = SizedString('name', size=8)
    shares = UnsignedInteger('shares')
    price = UnsignedFloat('price')

    def __init__(self, name, shares, price):
        self.name = name
        self.shares = shares
        self.price = price


s = Stock('ACME', 50, 91.1)
s.name
s.shares = 75
s.shares = -10  # ValueError: Expected >= 0
s.price = 'a lot'  # TypeError: expected <class 'float'>
s.name = 'ABRACADABRA'  # ValueError: size must be < 8


#   - 클래스의 제약 스펙을 단순화하는 몇 가지 기술이 있다. 우선 클래스 데코레이터를 사용하는 방식을 보자.
# 제약을 위한 클래스 데코레이터
def check_attributes(**kwargs):
    def decorate(cls):
        for key, value in kwargs.items():
            if isinstance(value, Descriptor):
                value.name = key
                setattr(cls, key, value)
            else:
                setattr(cls, key, value(key))
        return cls

    return decorate


@check_attributes(name=SizedString(size=8), shares=UnsignedInteger, price=UnsignedFloat)
class Stock:
    def __init__(self, name, shares, price):
        self.name = name
        self.shares = shares
        self.price = price


# - 또 다른 방식으로 메타클래스가 있다.
class checkedmeta(type):  # 추상 메서드 클래스로 사용
    def __new__(cls, clsname, bases, methods):
        # 디스크립터에 속성 이름 붙이기
        for key, value in methods.items():
            if isinstance(value, Descriptor):
                value.name = key
        return type.__new__(cls, clsname, bases, methods)


class Stock(metaclass=checkedmeta):
    name = SizedString(size=8)
    shares = UnsignedInteger()
    price = UnsignedFloat()

    def __init__(self, name, shares, price):
        self.name = name
        self.shares = shares
        self.price = price


# ▣ 토론 : 이번 레시피에서는 디스크립터, 믹스인 클래스(클래스에서 제공해야하는 추가적인 메서드만 정의하는 작은 클래스),
#            super() 활용, 클래스 데코레이터, 메타클래스 등 고급 기술을 다루었다.
#            이런 주제의 기본적인 사항을 여기서 모두 다루기에는 무리가 있지만 다른 레시피에서 예제를 볼 수는 있다.
#            (레시피 8.9, 8.18, 9.12, 9.19). 하지만 주목해야 할 점이 몇 가지 있다.
#            첫째, Descriptor 베이스 클래스에서, __set__() 메소드는 있는데 __get__() 이 없다는 것을 발견할 것이다.
#            디스크립터가 인스턴스 딕셔너리에서 동일한 이름의 값을 추출하는 것 이외에 다른 동작을 하지 않는다면, __get__() 을
#            정의할 필요가 없다.
#            사실 __get__() 을 정의하면 오히려 실행 속도가 느려진다. 따라서 이 레시피는 __set__() 구현에만 초점을 맞추었다.
#            여러 디스크립터 클래스의 전체적인 디자인은 믹스인 클래스에 기반하고 있다. 예를 들어 Unsigned 와 MaxSized 클래스는
#            Typed 에서 상속 받은 다른 디스크립터 클래스와 함께 사용하도록 디자인했다.
#            특정 데이터 타입을 처리하려면 다중 상속을 해야만 한다.
#            또한 디스크립터의 모든 __init__() 메소드는 키워드 매개변수 **opts 를 포함해서 동일한 시그니처를 가지도록 프로그램되어 있다는 점을 알 수 있다.
#            MaxSized 클래스는 opts 에서 요청한 속성을 찾지만, 이를 설정하는 Descriptor 베이스 클래스에 전달한다.
#            이런 클래스(특히 믹스인 관련)를 작성할 때 주의해야 할 점은, 클래스들이 어떻게 묶일지 혹은 어떤 super() 가 호출될지 알 수 없다는 것이다.
#            따라서 어떠한 조합으로도 잘 동작하도록 구현해야 한다.
#            Integer, Float, String 과 같은 타입 클래스 정의로부터 클래스 변수로 구현을 커스터마이즈하는 유용한 기술을 볼 수 있다.
#            Typed 디스크립터는 서브클래스가 제공한 expected_type 속성을 찾기만 한다.
#            클래스 데코레이터나 메타클래스를 사용하면 사용하면 사용자의 스펙을 단순화할 때 유용하다.
#            이런 예제에서 사용자가 속성의 이름을 한 번만 입력하면 된다는 점을 알 수 있다.

# 일반
class Point:
    x = Integer('x')
    y = Integer('y')


# 메타클래스
class Point(metaclass=checkedmeta):
    x = Integer()
    y = Integer()


# - 클래스 데코레이터와 메타클래스 코드는 단순히 클래스 딕셔너리에서 디스크립터를 찾는다.
#     디스크립터를 찾으면 키 값에 기반한 이름을 채워 넣는다.
#     앞에 나온 모든 방식 중에 클래스 데코레이터 방식이 가장 유연하고 안전하다.
#     우선 메타클래스와 같은 고급 기술에 의존하지 않는다. 그리고 데코레이션은 원할 때면 클래스 정의에 쉽게 추가하거나
#     제너할 수 있다.

#   - 마지막으로, 클래스 데코레이터 방식은 믹스인 클래스, 다중 상속, 복잡한 super() 대신 사용할 수 있다.
#     이번 레시피를 클래스 데코레이터를 사용해서 다시 작성해 보았다.

# 베이스 클래스, 값을 설정할 때 디스크립터를 사용
class Descriptor:
    def __init__(self, name=None, **opts):
        self.name = name
        for key, value in opts.items():
            setattr(self, key, value)

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value


# 타입 확인에 데코레이터 사용
def Typed(expected_type, cls=None):
    if cls is None:
        return lambda cls: Typed(expected_type, cls)

    super_set = cls.__set__

    def __set__(self, instance, value):
        if not isinstance(value, expected_type):
            raise TypeError('expected ' + str(expected_type))
        super_set(self, instance, value)

    cls.__set__ = __set__
    return cls


# unsigned 값에 데코레이터 사용
def Unsigned(cls):
    super_set = cls.__set__

    def __set__(self, instance, value):
        if value < 0:
            raise ValueError('Expected >= 0')
        super_set(self, instance, value)

    cls.__set__ = __set__
    return cls


# 크기 있는 값에 데코레이터 사용
def MaxSized(cls):
    super_init = cls.__init__

    def __init__(self, name=None, **opts):
        if 'size' not in opts:
            raise TypeError('missing size option')
        super_init(self, name, **opts)

    cls.__init__ = __init__

    super_set = cls.__set__

    def __set__(self, instance, value):
        if len(value) >= self.size:
            raise ValueError('size must be < ' + str(self.size))
        super_set(self, instance, value)

    cls.__set__ = __set__
    return cls


# 특별 디스크립터
@Typed(int)
class Integer(Descriptor):
    pass


@Unsigned
class UnsignedInteger(Integer):
    pass


@Typed(float)
class Float(Descriptor):
    pass


@Unsigned
class UnsignedFloat(Float):
    pass


@Typed(str)
class String(Descriptor):
    pass


@MaxSized
class SizedString(String):
    pass


# ※ 이번 코드에서 정의한 클래스는 어전 것과 완전히 동일하게 동작하고 오히려 실행 속도가 훨씬 빠르다.
#     예를 들어, 타입 속성에 값을 설정하는 간단한 실험을 통해 클래스 데코레이터 방식이 믹스인 방식보다 100% 빠르다는 것을
#     확인할 수 있다.


#  8.14 커스텀 컨테이너 구현
#  ▣ 문제 : 리스트나 딕셔너리와 같은 내장 컨테이너와 비슷하게 동작하는 커스텀 클래스를 구현하고 싶다.
#  ▣ 해결 : collections 라이브러리에 이 목적으로 사용하기 적절한 추상 베이스 클래스가 많이 정의되어 있다.
#            클래스에서 순환을 지원해야 한다고 가정해 보자. 이렇게 하려면 단순히 collections.Iterable 을 상속 받는 것으로 시작하면 된다.
import collections


class A(collections.Iterable):
    pass


# - collections.Iterable 을 상속 받으면 필요한 모든 특별 메소드를 구현하도록 보장해 준다.
#     메소드 구현을 잊으면 인스턴스화 과정에서 에러가 발생한다.
a = A()  # TypeError: Can't instantiate abstract class A with abstract methods __iter__

#   - collections 에 정의되어 있는 클래스에 또 주목할 만한 것으로 Sequence, Mutable Sequence, Mapping, MutableMapping, Set,
#     MutableSet 가 있다.
#     이 클래스 중 다수는 기능이 증가하는 체계를 형성한다.
#     이 클래스에 어떤 메소드를 구현해야 할지 확인하려면 단순히 인스턴스화해 보면 된다.
import collections

collections.Sequence()  # TypeError: Can't instantiate abstract class Sequence with abstract methods __getitem__, __len__

#   - 다음 코드는 필요한 메서드를 모두 구현해서 아이템을 정렬된 상태로 저장하는 시퀀스를 만든다.
import collections
import bisect


class SortedItems(collections.Sequence):
    def __init__(self, initial=None):
        self._items = sorted(initial) if initial is not None else []

    # 필요한 시퀀스 메소드
    def __getitem__(self, index):
        return self._items[index]

    def __len__(self):
        return len(self._items)

    # 올바른 장소에 아이템을 추가하기 위한 메소드
    def add(self, item):
        bisect.insort(self._items, item)


items = SortedItems([5, 1, 3])
list(items)
items[0]
items[-1]
items.add(2)
list(items)
items.add(-10)
list(items)
items[1:4]
3 in items
len(items)
for n in items:
    print(n)

# ※ SortedItems 의 인스턴스는 보통의 시퀀스와 동일한 동작을 하고, 인덱싱, 순환, len(), in 연산자, 자르기 등 일반적인
#     연산을 모두 지원한다.
#     참고로, 이번 레시피에서 사용한 bisect 모듈은 아이템을 정렬한 상태로 리스트에 보관할 때 매우 편리하다.

#  ▣ 토론 : collections 에 있는 추상 베이스 클래스를 상속 받으면 커스텀 컨테이너에 필요한 메소드를 모두 구현하도록 보장할 수 있다.
#            하지만 이 상속에는 타입 확인 기능도 있다.
#            예를 들어, 커스텀 컨테이너에 다음과 같은 타입 확인을 할 수 있다.
items = SortedItems()

import collections

isinstance(items, collections.Iterable)
isinstance(items, collections.Sequence)
isinstance(items, collections.Container)
isinstance(items, collections.Sized)
isinstance(items, collections.Mapping)


#   - collections 에 있는 추상 베이스 클래스는 일반적인 컨테이너 메소드의 기본 구현을 제공하는 것도 많다.
#     예를 들어 다음과 같이 collections.MutableSequence 에서 상속 받는 클래스가 있다고 가정해 보자.
class Items(collections.MutableSequence):
    def __init__(self, initial=None):
        self._items = list(initial) if initial is not None else []

    def __getitem__(self, index):
        print('Getting:', index)
        return self._items[index]

    def __setitem__(self, index, value):
        print('Setting:', index, value)
        self._items[index] = value

    def __delitem__(self, index):
        print('Deleting:', index)
        del self._items[index]

    def insert(self, index, value):
        print('Inserting:', index, value)
        self._items.insert(index, value)

    def __len__(self):
        print('Len')
        return len(self._items)


# - Items 의 인스턴스를 만들면, 리스트 메소드 중 중요한 것을 거의 다 지원한다는 것을 확인할 수 있다.
#     이런 메소드는 필요한 것만 사용하는 식으로 구현되어 있다.
a = Items([1, 2, 3])
len(a)
a.append(4)
a.append(2)
a.count(2)  # 시퀀스 객체 자체를 for 문으로 추출하므로 __getitem__() 메소드가 length 만큼 호출된다.
a.remove(3)


#  8.15 속성 접근 델리게이팅
#  ▣ 문제 : 인스턴스가 속성에 대한 접근을 내부 인스턴스로 델리게이트해서 상속의 대안으로 사용하거나 프록시 구현을 하고 싶다.
#  ▣ 해결 : 단순히 설명하자면, 델리게이트는 특정 동작에 대한 구현 책임을 다른 객체에게 미루는 프로그래밍 패턴이다.
class A:
    def spam(self, x):
        pass

    def foo(self):
        pass


class B:
    def __init__(self):
        self._a = A()

    def spam(self, x):
        return self._a.spam(x)

    def foo(self):
        return self._a.foo()

    def bar(self):
        pass


# - 델리게이트할 메소드가 몇 개 없으면, 주어진 코드를 그대로 작성해도 무방하다.
#     하지만 델리게이트해야 할 메소드가 많다면 또 다른 대안으로 다음과 같이 __getattr__() 메소드를 정의할 수 있다.
class A:
    def spam(self, x):
        pass

    def foo(self):
        pass


class B:
    def __init__(self):
        self._a = A()

    def bar(self):
        pass

    def __getattr__(self, name):  # 속성을 찾아보는 함수, 존재하지 않는 속성에 접근하려 할 때 호출.
        return getattr(self._a, name)


b = B()
b.bar()
b.spam()  # TypeError: spam() missing 1 required positional argument: 'x'


#   - 델리게이트의 또 다른 예제로 프록시 구현이 있다.
class Proxy:
    def __init__(self, obj):
        self._obj = obj

    # 속성 검색을 내부 객체로 델리게이트
    def __getattr__(self, name):
        print('getattr:', name)
        return getattr(self._obj, name)

    # 속성 할당 델리게이트
    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            print('setattr:', name, value)
            setattr(self._obj, name, value)

    # 속성 삭제 델리게이트
    def __delattr__(self, name):
        if name.startswith('_'):
            super().__delattr__(name)
        else:
            print('delattr:', name)
            delattr(self._obj, name)


class Spam:
    def __init__(self, x):
        self.x = x

    def bar(self, y):
        print('Spam.bar:', self.x, y)


s = Spam(2)
p = Proxy(s)

print(p.x)
p.bar(3)
p.x = 37


#  ※ 속성 접근 메소드 구현을 커스터마이징하면, 프록시가 다른 동작을 하도록 만들 수 있다.

#  ▣ 토론 : 델리게이트는 상속의 대안으로 사용하기도 한다.
#            예를 들어 다음과 같이 코드를 쓰지 않고,
class A:
    def spam(self, x):
        print('A.spam:', x)

    def foo(self):
        print('A.foo')


class B(A):
    def spam(self, x):
        print('B.spam')
        super().spam(x)

    def bar(self):
        print('B.bar')


# - 델리게이트를 활용해서 다음과 같이 작성할 수 있다.
class A:
    def spam(self, x):
        print('A.spam:', x)

    def foo(self):
        print('A.foo')


class B:
    def __init__(self):
        self._a = A()

    def spam(self, x):
        print('B.spam:', x)
        self._a.spam(x)

    def bar(self):
        print('B.bar')

    def __getattr__(self, name):
        return getattr(self._a, name)


# ※ 델리게이트를 사용하는 방식은 직접 상속이 어울리지 않거나 객체 간 관계를 더 조절하고 싶을 때(특정 메소드만 노출시키거나
#     인터페이스를 구현하는 등) 유용하다.
#     프록시를 구현하기 위해 델리게이트를 사용할 때, 기억해야 할 사항이 몇 가지 있다.
#     첫째, __getattr__() 메소드는 속성을 찾을 수 없을 때 한 번만 호출되는 fallback 메소드이다.
#     따라서 프록시 인스턴스의 속성 자체에 접근하는 경우, 이 메소드가 호출되지 않는다.
#     둘째, __setattr__() 과 __delattr__() 메소드는 프록시 인스턴스 자체와 내부 객체 _obj 속성의 개별 속성에 추가된 로직을 필요로 한다.
#     프록시에 대한 일반적인 관례는, 밑줄로 시작하지 않는 속성만 델리게이트하는 것이다.
#     또한 __getattr__() 메소드는 밑줄 두 개로 시작하는 대부분의 특별 메소드에 적용되지 않는다는 점도 중요하다.
class ListLike:
    def __init__(self):
        self._items = []

    def __getattr__(self, name):
        return getattr(self._items, name)


a = ListLike()
a.append(2)
a.insert(0, 1)
a.sort()
len(a)  # TypeError: 'ListLike' object is not iterable
a[0]  # TypeError: 'ListLike' object does not support indexing


#   - 서로 다른 연산을 지원하려면, 수동으로 관련된 특별 메소드를 델리게이트해야 한다.
class ListLike:
    def __init__(self):
        self._items = []

    def __getattr__(self, name):
        return getattr(self._items, name)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]

    def __setitem__(self, index, value):
        self._items[index] = value

    def __delitem__(self, index):
        del self._items[index]


# 8.16 클래스에 생성자 여러 개 정의
#  ▣ 문제 : 클래스를 작성 중인데, 사용자가 __init__() 이 제공하는 방식 이외에 여러 가지 방식으로 인스턴스를 생성할 수 있도록 하고 싶다.
#  ▣ 해결 : 생성자를 여러 개 정의하려면 클래스 메소드를 사용해야 한다.
import time

class Date:
    # 기본 생성자
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day

    # 대안 생성자
    @classmethod
    def today(cls):
        t = time.localtime()
        return cls(t.tm_year, t.tm_mon, t.tm_mday)

# - 두 번째 생성자를 사용하려면 Date.today() 와 같이 함수인 것처럼 호출하면 된다.
a = Date(2012, 12, 21)  # 기본
b = Date.today()  # 대안

#  ▣ 토론 : 클래스 메소드를 사용하는 주된 목적 중 하나가 바로 앞에 나온 것과 같은 생성자를 정의하는 것이다.
#            클래스를 첫 번째 인자(cls)로 받는 것이 클래스 메소드의 중요한 기능이다.
#            이 클래스는 메소드 내부에서 인스턴스를 생성하고 반환하기 위해 사용된다.
#            아주 미묘한 부분이지만, 바로 이 측면으로 인해 클래스 메소드가 상속과 같은 기능과도 잘 동작하게 된다.
class NewDate(Date):
    pass

c = Date.today()  # Date (cls=Date) 인스턴스 생성
d = NewDate.today()  # NewDate (cls=NewDate) 인스턴스 생성

#   - 생성자가 많은 클래스를 정의할 때, 주어진 값을 속성에 할당하는 이상 아무런 동작을 하지 않도록, 될 수 있으면 __init__()
#     을 최대한 단순하게 만들어야 한다.
#     그리고 대안이 되는 생성자에서 필요한 경우 더 고급 연산을 수행하면 된다.
#     개별적인 클래스 메소드를 정의하지 않고, __init__() 메소드에서 여러 동작을 호출하도록 하고 싶을 수도 있다.
class Date:
    def __init__(self, *args):
        if len(args) == 0:
            t = time.localtime()
            args = (t.tm_year, t.tm_mon, t.tm_mday)
        self.year, self.month, self.day = args

# - 이 기술이 제대로 동작하는 경우도 더러 있지만, 대개의 경우 이 코드는 이해하기도 어렵고 유지 보수도 쉽지 않다.
#     예를 들어 이 구현은 도움말을 제대로 보여주지 않는다.
#     그리고 Date 인스턴스를 만드는 코드도 명확하지 않게 된다.
a = Date(2012, 12, 21)
b = Date()

# class method version
c = Date.today()

#   - 앞선 코드에서 살펴본 대로 Date.today() 는 Date() 에 적절한 연, 월, 일을 인자로 넘겨 인스턴스를 만들며 기본 Date.__init__()
#     메소드를 호출한다.
#     필요하다면, __init__() 메소드를 호출하지 않고도 인스턴스를 생성할 수 있다.

#  8.17 init 호출 없이 인스턴스 생성
#  ▣ 문제 : 인스턴스를 생성해야 하는데, __init__() 메소드 호출을 피하고 싶다.
#  ▣ 해결 : 클래스의 __new__() 메소드를 호출해서 초기화하지 않은 인스턴스를 생성할 수 있다.
class Date:
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day

d = Date.__new__(Date)  # __init__() 호출 없이 Date 인스턴스를 생성.
d
d.year

#   - 앞에 나온 것처럼 생성된 인스턴스는 초기화되지 않았다. 따라서 적절한 인스턴스 변수를 설정하는 것은 사용자의 몫이다.
data = {'year': 2012, 'month': 8, 'day': 29}
for key, value in data.items():
    setattr(d, key, value)

d.year
d.month

#  ▣ 토론 : __init__() 을 생략하면 데이터 역직렬화나 대안 생성자로 정의한 클래스 메소드의 구현과 같이 비표준 방식으로
#            인스턴스를 생성할 때 문제가 발생하기도 한다.
#            예를 들어, Date 클래스에 today() 를 다음과 같이 정의할 수 있다.
from time import localtime

class Date:
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day

    @classmethod
    def today(cls):
        d = cls.__new__(cls)
        t = localtime()
        d.year = t.tm_year
        d.month = t.tm_mon
        d.day = t.tm_mday
        return d

# - 이와 유사하게, JSON 데이터를 역직렬화하면 다음과 같은 딕셔너리가 생성된다.
data = {'year': 2012, 'month': 8, 'day': 29}

#   - 비표준 방식으로 인스턴스를 생성할 때, 구현에 대해서 너무 많은 가정을 하지 않는 것이 좋다.
#     일반적으로 확실히 정의되어 있다고 보장되어 있지 않다면 인스턴스 딕셔너리 __dict__ 에 직접 접근해서 수정하는 코드는
#     작성하지 말아야 한다.
#     그렇지 않으면, 클래스가 __slots__, 프로퍼티, 디스크립터 등 고급 기술을 사용하는 경우에 코드에 문제가 생긴다.
#     setattr() 로 값을 설정하면 코드가 최대한 일반적인 목적이 된다.


#  8.20 문자열로 이름이 주어진 객체의 메소드 호출
#  ▣ 문제 : 문자열로 저장된 메소드 이름을 가지고 있고, 이 메소드를 실행하고 싶다.
#  ▣ 해결 : 간단한 경우, getattr() 를 사용하면 된다.
import math

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return 'Point({!r:}, {!r:})'.format(self.x, self.y)

    def distance(self, x, y):
        return math.hypot(self.x - x, self.y - y)

p = Point(2, 3)
d = getattr(p, 'distance')(0, 0)
print(d)

#   - 혹은 operator.methodcaller() 를 사용한다.
import operator
operator.methodcaller('distance', 0, 0)(p)

#   - 메소드를 이름으로 찾고 동일한 매개변수를 반복적으로 넣는 경우 operator.methodcaller() 를 사용하는 것이 좋다.
#     예를 들어 포인트 리스트를 정렬하고 싶다면 다음과 같이 한다.
points = [
    Point(1, 2),
    Point(3, 0),
    Point(10, -3),
    Point(-5, -7),
    Point(-1, 8),
    Point(3, 2)
]

points.sort(key=operator.methodcaller('distance', 0, 0))

#  ▣ 토론 : 메소드 호출의 과정은 실제로 속성 탐색과 함수 호출이라는 두 가지 과정으로 분리된다.
#            따라서 메소드를 호출하려면, 다른 속성과 마찬가지로 우선 getattr() 로 속성을 찾는다.
#            찾은 메소드를 호출하려면, 결과물을 함수로 여기면 된다.
#            operator.methodcaller() 는 호출 가능 객체를 생성하지만, 또한 메소드에 주어질 매개 변수를 고정시키는 역할도 한다.
#            우리는 올바른 self 인자를 제공하기만 하면 된다.
p = Point(3, 4)
d = operator.methodcaller('distance', 0, 0)
d(p)


#  8.24 비교 연산을 지원하는 클래스 만들기
#  ▣ 문제 : 표준 비교 연산자(>=, !=, <=) 를 사용해 클래스 인스턴스를 비교하고 싶다.
#            하지만 특별 메소드를 너무 많이 작성하고 싶지는 않다.
#  ▣ 해결 : 파이썬 클래스는 비교 연산을 위한 특별 메소드를 통해 인스턴스 간에 비교 기능을 지원한다.
#            예를 들어 >= 연산을 지원하려면 __ge__() 메소드를 정의하면 된다.
#            메소드 하나를 정의하는 것은 아무런 문제가 없지만 모든 비교 연산을 구현하려면 그 과정이 조금 귀찮아진다.
#            이때 functools.total_ordering 데코레이터를 사용하면 과정을 단순화할 수 있다.
#            클래스에 데코레이터를 붙이고 __eq__() 와 비교 메소드 (__lt__, __le__, __gt__, __ge__) 하나만 더 정의하면 된다.
#            그렇게 하면 데코레이터가 나머지 모든 메소드를 자동으로 추가해 준다.
from functools import total_ordering

class Room:
    def __init__(self, name, length, width):
        self.name = name
        self.length = length
        self.width = width
        self.square_feet = self.length * self.width

@total_ordering
class House:
    def __init__(self, name, style):
        self.name = name
        self.style = style
        self.rooms = list()

    @property
    def living_space_footage(self):
        return sum(r.square_feet for r in self.rooms)

    def add_room(self, room):
        self.rooms.append(room)

    def __str__(self):
        return '{} : {} square foot {}'.format(self.name, self.living_space_footage, self.style)

    def __eq__(self, other):
        return self.living_space_footage == other.living_space_footage

    def __lt__(self, other):
        return self.living_space_footage < other.living_space_footage

#   - 이 코드에서 House 클래스를 @total_ordering 으로 꾸몄다.
#     그리고 집의 전체 크기를 비교하는 __eq__() 와 __lt__() 를 정의했다.
#     이렇게 최소한의 정의만 하면 나머지 모든 비교 연산도 정상적으로 동작한다.
h1 = House('h1', 'Cape')
h1.add_room(Room('Master Bedroom', 14, 21))
h1.add_room(Room('Living Room', 18, 20))
h1.add_room(Room('Kitchen', 12, 16))
h1.add_room(Room('Office', 12, 12))

h2 = House('h2', 'Ranch')
h2.add_room(Room('Master Bedroom', 14, 21))
h2.add_room(Room('Living Room', 18, 20))
h2.add_room(Room('Kitchen', 12, 16))

h3 = House('h3', 'Split')
h3.add_room(Room('Master Bedroom', 14, 21))
h3.add_room(Room('Living Room', 18, 20))
h3.add_room(Room('Office', 12, 16))
h3.add_room(Room('Kitchen', 15, 17))
houses = [h1, h2, h3]

print('Is h1 bigger than h2?', h1 > h2)
print('Is h2 smaller than h3?', h2 < h3)
print('Is h2 greater than or equal to h1?', h2 >= h1)
print('Which one is biggest?', max(houses))
print('Which is smallest?', min(houses))

#  ▣ 토론 : 기본 비교 연산을 모두 지원하는 클래스를 작성해 본 적이 있다면 total_ordering 의 동작성을 이해할 수 있을 것이다.
#            이 데코레이터는 비교-지원 메소드에서 다른 모든 메소드로의 매핑을 정의한다.
#            따라서 __lt__() 를 클래스에 정의하면 다른 비교 연산이 이 메소드를 사용한다.
#            다음과 같이 클래스를 메소드로 채워 주는 것과 다를 바가 없는 것이다.
class House:
    def __eq__(self, other):
        pass

    def __lt__(self, other):
        pass

    # @total_ordering 이 생성한 메소드
    __le__ = lambda self, other: self < other or self == other
    __gt__ = lambda self, other: not (self < other or self == other)
    __ge__ = lambda self, other: not (self < other)
    __ne__ = lambda self, other: not self == other