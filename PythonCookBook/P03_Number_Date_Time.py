# Chapter 3. 숫자, 날짜, 시간
#  3.1 반올림
#  ▣ 문제 : 부동 소수점 값을 10진수로 반올림하고 싶다.
#  ▣ 해결 : 간단한 반올림은 내장 함수인 round(value, ndigits) 함수를 사용한다.
print(round(1.23, 1))  # 소수점 자리 반올림
print(round(1.27, 1))
print(round(-1.27, 1))
print(round(1.25361, 3))
print(round(2.5))

a = 1627731
print(round(a, -1))  # 정수 자리 반올림
print(round(a, -2))
print(round(a, -3))

#  ▣ 토론 : 반올림과 서식화를 헷갈리지 않도록 주의하자. 특정 자리수까지 숫자를 표현하는 것이 목적이라면 round() 를 사용하는 것이 아니라
#           서식화를 위한 자릿수를 명시하기만 하면 된다.
x = 1.23456
print(format(x, '0.2f'))
print(format(x, '0.3f'))
print('value is {:0.3f}'.format(x))

#   - 정확도 문제를 수정하려고 부동 소수점을 반올림하는 방법도 지양해야 한다.
a = 2.1
b = 4.2
c = a + b
print(c)
c = round(c, 2)
print(c)


#  3.2 정확한 10진수 계산
#  ▣ 문제 : 정확한 10진수 계산을 해야 하고, 부동 소수점을 사용할 때 발생하는 작은 오류를 피하고 싶다.
#  ▣ 해결 : 부동 소수점 사용 시 더 정확한 계산을 하고 싶다면, decimal 모듈을 사용해야 한다.
a = 4.2
b = 2.1
print(a + b)
print((a + b) == 6.3)

from decimal import Decimal
a = Decimal('4.2')
b = Decimal('2.1')
print(a+b)
print((a + b) == Decimal('6.3'))

#   - 반올림의 자릿수와 같은 계산적 측면을 조절할 수 있다. (localcontext())
from decimal import localcontext
a = Decimal('1.3')
b = Decimal('1.7')
print(a / b)
with localcontext() as ctx:
    ctx.prec = 3
    print(a / b)

with localcontext() as ctx:
    ctx.prec = 50
    print(a / b)

#  ▣ 토론 : 1. 과학이나 공학, 컴퓨터 그래픽 등 자연 과학 영역을 다룰 때는 부동 소수점 값을 사용하는 것이 더 일반적이다.
#           2. decimal 모듈에 비해 float 의 실행 속도가 확연히 빠르다.
nums = [1.23e+18, 1, -1.23e+18]
print(sum(nums))  # 1이 사라진다.
import math
print(math.fsum(nums))


#  3.3 출력을 위한 숫자 서식화
#  ▣ 문제 : 출력을 위해 자릿수, 정렬, 천 단위 구분 등 숫자를 서식화하고 싶다.
#  ▣ 해결 : 출력을 위해 숫자를 서식화하려면 내장 함수인 format() 을 사용한다.
#   - 소수점 둘째 자리 정확도
x = 1234.56789
print(format(x, '0.2f'))

#   - 소수점 한 자리 정확도로 문자 10개 기준 오른쪽에서 정렬
print(format(x, '>10.1f'))

#   - 왼쪽에서 정렬
print(format(x, '<10.1f'))

#   - 가운데 정렬
print(format(x, '^10.1f'))

#   - 천 단위 구분자 넣기
print(format(x, ','))
print(format(x, '0,.1f'))

#   - 지수 표현법 사용하려면 f 를 e나 E로 바꾸면 된다.
print(format(x, 'e'))
print(format(x, '0.2E'))

print('The Value is {:0,.2f}'.format(x))

#  ▣ 토론 : 출력을 위한 숫자 서식화는 대개 간단하다. 앞에 소개한 기술은 부동 소수점 값과 decimal 모듈의 숫자에 모두 잘 동작한다.
print(format(x, '0.1f'))
print(format(-x, '0.1f'))

#   - 지역 표기법을 따르기 위해 locale 모듈의 함수를 사용한다.
swap_separators = {ord('.'): ',', ord(','): '.'}
print(format(x, ',').translate(swap_separators))

#   - % 연산자로 서식화.
print('%0.2f' % x)
print('%10.1f' % x)
print('%-10.1f' % x)


#  3.4 2진수, 8진수, 16진수 작업
#  ▣ 문제 : 숫자를 2진수, 8진수, 16진수로 출력해야 한다.
#  ▣ 해결 : 정수를 2진수, 8진수, 16진수 문자열로 변환하려면 bin(), oct(), hex() 를 사용한다.
x = 1234
print(bin(x), oct(x), hex(x))
print(format(x, 'b'), format(x, 'o'), format(x, 'x'))

#   - 정수형은 부호가 있는 숫자이므로, 음수를 사용하면 결과물에도 부호가 붙는다.
x = -1234
print(format(x, 'b'), format(x, 'x'))

#   - 부호가 없는 값을 사용하려면 최대값을 더해서 비트 길이를 설정해야 한다.
x = -1234
print(format(2**32 + x, 'b'), format(2**32 + x, 'x'))

#   - 다른 진법의 숫자를 정수형으로 변환하려면 int() 함수에 적절한 진수를 전달한다.
print(int('4d2', 16), int('10011010010', 2))

#  ▣ 토론 : 8 진법을 사용할 때 프로그래머가 주의해야 할 점이 한 가지 있다.
import os
os.chmod('script.py', 0o755)  # 8진법 값 앞에는 0o 를 붙여야 한다. (chmod = 파일 권한 변환)


#  3.5 바이트에서 큰 숫자를 패킹/언패킹
#  ▣ 문제 : 바이트 문자열을 언패킹해서 정수 값으로 만들어야 한다. 혹은 매우 큰 정수 값을 바이트 문자열로 변환해야 한다.
#  ▣ 해결 : int.from_bytes() 메소드를 사용한다.
data = b'\x00\x124V\x00x\x90\xab\x00\xcd\xef\x01\x00#\x004'
print(len(data))
print(int.from_bytes(data, 'little'))
print(int.from_bytes(data, 'big'))

#   - 큰 정수 값을 바이트 문자열로 변환하려면 int.to_bytes() 메소드를 사용하고, 바이트 길이와 순서를 명시한다.
x = 94522842520747284487117727783387188
print(x.to_bytes(16, 'big'))
print(x.to_bytes(16, 'little'))

#  ▣ 토론 : 정수형 값과 바이트 문자열 간의 변환은 일반적인 작업이 아니다.
#           하지만 네트워크나 암호화가 필요한 특정 애플리케이션에서 사용하는 경우가 있다.
#           여기서 나온 방법 말고, struct 모듈을 사용할 수도 있다.

#   - struct 로 언패킹할 수 있는 정수형의 크기가 제한적이어서, 언팩을 여러 번 하고 결과 값을 합쳐야 한다.
import struct
hi, lo = struct.unpack('>QQ', data)
print((hi << 64) + lo)

#   - 바이트 순서는 정수형을 이루는 바이트가 가장 작은 것부터 표시되었는지 혹은 가장 큰 것부터 표시되었는지를 나타낸다.
x = 0x01020304
print(x.to_bytes(4, 'big'))
print(x.to_bytes(4, 'little'))

#   - 정수형 값을 바이트 문자열로 변환하려는데 지정한 길이에 다 들어가지 않는 경우에는 에러가 발생하므로, int.bit_length() 메소드로 확인한다.
x = 523 ** 23
print(x.bit_length())
nbytes, rem = divmod(x.bit_length(), 8)  # byte 자리수, 나머지 리턴
if rem:
    nbytes += 1
print(x.to_bytes(nbytes, 'little'))


#  3.6 복소수 계산
#  ▣ 문제 : 최신 웹 인증을 사용하는 코드를 작성하던 도중에 특이점을 발견하고 복소수 평면을 사용할 수 밖에 없는 상황에 처했다.
#           혹은 복소수를 사용하여 계산을 해야 한다.
#  ▣ 해결 : 복소수는 complex(real, imag) 함수를 사용하거나 j를 붙인 부동 소수점 값으로 표현할 수 있다.
a = complex(2, 4)
b = 3 - 5j
print(a, b)

#   - 실수, 허수, 켤레 복소수(허수 부분의 부호를 바꾼 복소수)를 구하는 방법.
print(a.real, a.imag, a.conjugate())

#   - 일반적인 수학 계산하는 방법.
print(a + b)
print(a * b)
print(a / b)
print(abs(a))  # 절대값

#   - 사인, 코사인, 제곱 등을 계산하려면 cmath 모듈을 사용한다.
import cmath
print(cmath.sin(a))
print(cmath.cos(a))
print(cmath.exp(a))

#  ▣ 토론 : 파이썬의 수학 관련 모듈은 대개 복소수를 인식한다. 예를 들어, numpy 를 사용하면 어렵지 않게 복소수 배열을 만들고 계산할 수 있다.
import numpy as np
a = np.array([2+3j, 4+5j, 6-7j, 8+9j])
print(a)
print(a+2)
print(np.sin(a))

#   - 하지만 파이썬의 표준 수학 함수는 기본적으로 복소수 값을 만들지 않는다.
#     따라서 코드에서 이런 값이 예상치 않게 발생하지는 않는다.
import math
print(math.sqrt(-1))
import cmath
print(cmath.sqrt(-1))  # cmath 모듈이 복소수에 대해 지원한다.


#  3.7 무한대와 NaN 사용
#  ▣ 문제 : 부동 소수점 값의 무한대, 음의 무한대, NaN(not a number)을 검사해야 한다.
#  ▣ 해결 : 이와 같은 특별한 부동 소수점 값을 표현하는 파이썬 문법은 없지만 float() 를 사용해서 만들 수는 있다.
a = float('inf')
b = float('-inf')
c = float('nan')
print(a, b, c)

#   - 값을 확인하기 위해 math.isinf() 와 math.isnan() 함수를 사용한다.
print(math.isinf(a), math.isinf(b))
print(math.isnan(c))

#  ▣ 토론 : 앞에 나온 특별한 부동 소수점 값에 대한 더 많은 정보를 원한다면 IEEE 754 스펙을 확인해야 한다.

#   - 무한대 값은 일반적인 수학 계산법을 따른다.
a = float('inf')
print(a + 45)
print(a * 10)
print(10 / a)

#   - 특정 연산자의 계산은 정의되어 있지 않고 NaN 을 발생시킨다.
print(a / a)
b = float('-inf')
print(a + b)

#   - NaN 값은 모든 연산자에 대해 예외를 발생시키지 않는다.
c = float('nan')
print(c + 23)
print(c / 2)
print(c * 2)
print(math.sqrt(c))

#   - NaN 에서 주의해야 할 점은, 이 값은 절대로 비교 결과가 일치하지 않는다는 점이다.
c = float('nan')
d = float('nan')
print(c == d)
print(c is d)


#  3.8 분수 계산
#  ▣ 문제 : 타임머신에 탑승했는데 갑자기 분수 계산을 하는 초등학생이 되어 버렸다.
#           혹은 목공소에서 만든 측량기에 관련된 계산을 하는 코드를 작성해야 한다.
#  ▣ 해결 : 분수 관련 계산을 위해 fractions 모듈을 사용한다.
from fractions import Fraction
a = Fraction(5, 4)
b = Fraction(7, 16)
print(a + b)
print(a * b)

#   - 분자 / 분모 구하기
c = a * b
print(c.numerator)  # 분자
print(c.denominator)  # 분모

#   - 소수로 변환
print(float(c))

#   - 분자를 특정 값으로 제한
print(c.limit_denominator(8))

#   - 소수를 분수로 변환
x = 3.75
y = Fraction(*x.as_integer_ratio())  # x.as_integer_ratio() : float 을 (분자, 분모) 를 쌍으로 가지는 튜플로 리턴한다.
print(y)

#  ▣ 토론 : 프로그램에서 치수 단위를 분수로 받아서 계산을 하는 것이 사용자가 소수로 직접 변환하고 계산하는 것보다 더 편리할 수 있다.


#  3.9 큰 배열 계산
#  ▣ 문제 : 배열이나 그리드와 같이 커다란 숫자 데이터 세트에 계산을 해야 한다.
#  ▣ 해결 : 배열이 관련된 커다란 계산을 하려면 NumPy 라이브러리를 사용한다.
#            NumPy 를 사용하면 표준 파이썬 리스트를 사용하는 것보다 수학 계산에 있어 훨씬 효율적이다.

#   - 파이썬 리스트
x = [1, 2, 3, 4]
y = [5, 6, 7, 8]
print(x * 2)
print(x + 10)  # 수행 안됨

#   - Numpy 배열
import numpy as np
ax = np.array([1, 2, 3, 4])
ay = np.array([5, 6, 7, 8])
print(ax*2)
print(ax + 10)
print(ax + ay)
print(ax * ay)

def f(x):
    return 3*x**2 - 2*x + 7
print(f(ax))

#   - Numpy 는 배열에 사용 가능한 "일반 함수"를 제공한다.
print(np.sqrt(ax))
print(np.cos(ax))
#    ※ 일반 함수는 배열 요소를 순환하며 요소마다 math 모듈 함수로 계산하는 것보다 수백 배 빠르다.

#   - 10,000 x 10,000 2차원 그리드를 만드는 경우.
grid = np.zeros(shape=(10000, 10000), dtype=float)
print(grid)
grid += 10
print(grid)
print(np.sin(grid))

#   - Numpy 는 다차원 배열의 인덱싱 기능을 확장하고 있다.
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(a)
print(a[1])
print(a[:, 1])
print(a[1:3, 1:3])
a[1:3, 1:3] += 10
print(a)

#   - 행 벡터를 모든 행 연산에 적용
print(a + [100, 101, 102, 103])
print(a)

#   - 조건이 있는 할당
print(np.where(a < 10, a, 10))

#  ▣ 토론 : Numpy 는 파이썬의 수많은 과학, 공학, 라이브러리의 기초가 된다.
#           또한 광범위하게 사용되는 모듈 중 가장 복잡하고 방대한 것 중 하나이다.


#  3.10 행렬과 선형 대수 계산
#  ▣ 문제 : 행렬 곱셈, 행렬식 찾기, 선형 방정식 풀기 등 행렬이나 선형 대수 계산을 해야 한다.
#  ▣ 해결 : Numpy 라이브러리에 이런 용도로 사용할 수 있는 matrix 객체가 있다.
import numpy as np
m = np.matrix([[1, -2, 3], [0, 4, 5], [7, 8, -9]])
print(m)

#   - 전치 행렬
print(m.T)

#   - 역행렬
print(m.I)

#   - 벡터를 만들고 곱하기
v = np.matrix([[2], [3], [4]])
print(v)
print(m * v)

import numpy.linalg

#   - 행렬식(determinant) : ad-bc
print(numpy.linalg.det(m))

#   - 고유값(eigenvalues) :  한 요인이 설명해 줄 수 있는 변수들의 분산 총합.
print(numpy.linalg.eigvals(m))

#   - mx = v 에서 x 풀기
x = numpy.linalg.solve(m, v)
print(x)
print(m * x)
print(v)

#  ▣ 토론 : 선형 대수의 범위는 너무 방대해서 이 책에서 다 다룰 수 없다.
#           하지만 행렬과 벡터를 다루어야 한다면 Numpy 부터 시작하도록 하자.


#  3.11 임의의 요소 뽑기
#  ▣ 문제 : 시퀀스에서 임의의 아이템을 고르거나 난수를 생성하고 싶다.
#  ▣ 해결 : random 모듈에는 이 용도에 사용할 수 있는 많은 함수가 있다.
#           예를 들어 시퀀스에서 임의의 아이템을 선택하려면 random.choice() 를 사용한다.
import random
values = [1, 2, 3, 4, 5, 6]
print(random.choice(values))

#   - 임의의 아이템을 N개 뽑아서 사용하고 버릴 목적이라면 random.sample() 을 사용한다.
print(random.sample(values, 2))

#   - 시퀀스의 아이템을 무작위로 섞으려면 random.shuffle() 을 사용한다.
random.shuffle(values)
print(values)

#   - 임의의 정수를 생성하려면 random.randint() 를 사용한다.
print(random.randint(0, 10))

#   - 0 과 1 사이의 균등 부동 소수점 값을 생성하려면 random.random() 을 사용한다.
print(random.random())

#   - N 비트로 표현된 정수를 만들기 위해서는 random.getrandbits() 를 사용한다.
print(random.getrandbits(200))

#  ▣ 토론 : random 모듈은 Mersenne Twister 알고리즘을 사용해 난수를 발생시킨다.
#           이 알고리즘은 정해진 것이지만, random.seed() 함수로 시드 값을 바꿀 수 있다.
#           random() 의 함수는 암호화 관련 프로그램에서 사용하지 말아야 한다.
#           그런 기능이 필요하다면 ssl 모듈을 사용해야 한다.
random.seed()  # 시스템 시간이나 os.urandom() 시드
random.seed(12345)  # 주어진 정수형 시드
random.seed(b'bytedata')  # 바이트 데이터 시드


#  3.12 시간 단위 변환
#  ▣ 문제 : 날짜를 초로, 시간을 분으로처럼 시간 단위 변환을 해야 한다.
#  ▣ 해결 : 단위 변환이나 단위가 다른 값에 대한 계산을 하려면 datetime 모듈을 사용한다.
#            예를 들어 시간의 간격을 나타내기 위해서는 timedelta 인스턴스를 생성한다.
from datetime import timedelta
a = timedelta(days=2, hours=6)  # days, hours 따로 동작한다.
b = timedelta(hours=4.5)
c = a + b
print(c.days, c.seconds, c.seconds/3600, c.total_seconds() / 3600)

#   - 특정 날짜와 시간을 표현하려면 datetime 인스턴스를 만들고 표준 수학 연산을 한다.
from datetime import datetime
a = datetime(2012, 9, 23)
print(a + timedelta(days=10))
b = datetime(2012, 12, 21)
d = b - a
print(d.days)
now = datetime.today()
print(now)
print(now + timedelta(minutes=10))

#   - datetime 이 윤년을 인식한다는 점에 주목하자.
a = datetime(2012, 3, 1)
b = datetime(2012, 2, 28)
print(a - b)
print((a - b).days)
c = datetime(2013, 3, 1)
d = datetime(2013, 2, 28)
print((c - d).days)

#  ▣ 토론 : 대부분의 날짜, 시간 계산 문제는 datetime 모듈로 해결할 수 있다.
#           시간대나, 퍼지 시계 범위, 공휴일 계산 등의 더욱 복잡한 날짜 계산이 필요하다면 dateutil 모듈을 알아보자.
a = datetime(2012, 9, 23)
print(a + timedelta(months=1))  # months 는 지원하지 않음

from dateutil.relativedelta import relativedelta
print(a + relativedelta(months=+1))
print(a + relativedelta(months=+4))

#   - 두 날짜 사이의 시간
b = datetime(2012, 12, 21)
d = b - a
print(d)
d = relativedelta(b, a)
print(d)
print(d.months, d.days)


#  3.13 마지막 금요일 날짜 구하기
#  ▣ 문제 : 한 주의 마지막에 나타난 날의 날짜를 구하는 일반적인 해결책을 만들고 싶다.
#           예를 들어 마지막 금요일이 며칠인지 궁금하다.
#  ▣ 해결 : 파이썬의 datetime 모듈에 이런 계산을 도와 주는 클래스와 함수가 있다.
from datetime import datetime, timedelta
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

def get_previous_byday(dayname, start_date=None):
    if start_date is None:
        start_date = datetime.today()
    day_num = start_date.weekday()
    day_num_target = weekdays.index(dayname)
    days_ago = (7 + day_num - day_num_target) % 7
    if days_ago == 0:
        days_ago = 7
    target_date = start_date - timedelta(days=days_ago)
    return target_date

print(datetime.today())
print(get_previous_byday('Monday'))
print(get_previous_byday('Tuesday'))
print(get_previous_byday('Friday'))
print(get_previous_byday('Sunday', datetime(2012, 12, 21)))

#  ▣ 토론 : 이번 레시피는 시작 날짜와 목표 날짜를 관련 있는 숫자 위치에 매핑하는 데에서 시작한다.
#            이와 같은 날짜 계산을 많이 수행한다면 python-dateutil 패키지를 설치하는 것이 좋다.
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dateutil.rrule import *
d = datetime.now()
print(d)

#   - 다음 금요일
print(d + relativedelta(weekday=FR))

#   - 마지막 금요일
print(d + relativedelta(weekday=FR(-1)))


#  3.14 현재 달의 날짜 범위 찾기
#  ▣ 문제 : 현재 달의 날짜를 순환해야 하는 코드가 있고, 그 날짜 범위를 계산하는 효율적인 방법이 필요하다.
#  ▣ 해결 : 날짜를 순환하기 위해 모든 날짜를 리스트로 만들 필요가 없고, 시작과 마지막 날짜만 계산하고 datetime.timedelta
#            객체를 사용해서 날짜를 증가시키면 된다.
from datetime import datetime, date, timedelta
import calendar

def get_month_range(start_date=None):
    if start_date is None:
        start_date = date.today().replace(day=1)
    _, days_in_month = calendar.monthrange(start_date.year, start_date.month)
    end_date = start_date + timedelta(days=days_in_month)
    return (start_date, end_date)

a_day = timedelta(days=1)
first_day, last_day = get_month_range()
while first_day < last_day:
    print(first_day)
    first_day += a_day

#  ▣ 토론 : 위처럼 구현할 수도 있지만 이상적으로는 generator 를 사용하면 아주 쉽게 구현할 수 있다.
def date_range(start, stop, step):
    while start < stop:
        yield start
        start += step

for d in date_range(datetime(2012, 9, 1), datetime(2012, 10, 1), timedelta(hours=6)):
    print(d)


#  3.15 문자열을 시간으로 변환
#  ▣ 문제 : 문자열 형식의 시간 데이터를 datetime 객체로 변환하고 싶다.
#  ▣ 해결 : 파이썬의 datetime 모듈을 사용하면 상대적으로 쉽게 이 문제를 해결할 수 있다.
from datetime import datetime
text = '2012-09-20'
y = datetime.strptime(text, '%Y-%m-%d')
z = datetime.now()
diff = z- y
print(diff)

#  ▣ 토론 : datetime.strptime() 메소드는 네 자리 연도 표시를 위한 %Y, 두 자리 월 표시를 위한 %m 과 같은 서식을 지원한다.

#   - datetime 객체를 생성하는 코드가 있는데, 이를 사람이 이해하기 쉬운 형태로 변환할 수 있다.
print(z)
nice_z = datetime.strftime(z, '%A %B %d, %Y')
print(nice_z)

#   - strptime() 함수는 순수 파이썬만을 사용해서 구현했고, 시스템 설정과 같은 세세한 부분을 모두 처리해야 하므로 예상보다 실행 속도가
#     느린 경우가 많다.
#     따라서 직접 구현하는 것이 속도 측면에서 훨씬 유리하다.
from datetime import datetime

def parse_ymd(s):
    year_s, mon_s, day_s = s.split('-')
    return datetime(int(year_s), int(mon_s), int(day_s))


#  3.16 시간대 관련 날짜 처리
#  ▣ 문제 : 시카고 시간으로 2012년 12월 21일 오전 9시 30분에 화상 회의가 예정되어 있다.
#           그렇다면 인도의 방갈로르에 있는 친구는 몇 시에 회의실에 와야 할까?
#  ▣ 해결 : 시간대와 관련 있는 거의 모든 문제는 pytz 모듈로 해결한다.
#           이 패키지는 많은 언어와 운영 체제에서 기본적으로 사용하는 Olson 시간대 데이터베이스를 제공한다.
#           pytz 는 주로 datetime 라이브러리에서 생성한 날짜를 현지화할 때 사용한다.
from datetime import datetime, timedelta
from pytz import timezone
d = datetime(2012, 12, 21, 9, 30, 0)
print(d)

#   - 시카고에 맞게 현지화.
central = timezone('US/Central')
loc_d = central.localize(d)
print(loc_d)

#   - 방갈로르 시간으로 변환.
bang_d = loc_d.astimezone(timezone('Asia/Kolkata'))
print(bang_d)

#   - 변환한 날짜에 산술 연산을 하려면 서머타임제 등을 알고 있어야 한다.
#     예를 들어 2013 년 미국에서 표준 서머타임은 3월 13일 오전 2시에 시작한다.
d = datetime(2013, 3, 10, 1, 45)
loc_d = central.localize(d)
print(loc_d)
later = loc_d + timedelta(minutes=30)
print(later)

#   - 서머타임을 고려하면 normalize() 메소드를 사용한다.
from datetime import timedelta
later = central.normalize(loc_d + timedelta(minutes=30))
print(later)

#  ▣ 토론 : 현지화된 날짜를 조금 더 쉽게 다루기 위한 한 가지 전략으로, 모든 날짜를 UTC(세계 표준 시간) 시간으로 변환해 놓고 사용하는 것이 있다.
import pytz
print(loc_d)
utc_d = loc_d.astimezone(pytz.utc)
print(utc_d)

#   - 현지화 시간을 원할 경우 모든 계산을 마친 후에 원하는 시간대로 변환한다.
later_utc = utc_d + timedelta(minutes=30)
print(later_utc.astimezone(central))

#   - 특정 국가 시간대 이름 출력. (pytz.country_timezones)
#     키 값으로 ISO 3166 국가 코드를 사용.
print(pytz.country_timezones['IN'])