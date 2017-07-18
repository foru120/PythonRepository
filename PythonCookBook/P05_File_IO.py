# Chapter 5. 파일과 입출력
#  5.1 텍스트 데이터 읽고 쓰기
#  ▣ 문제 : 텍스트 데이터를 읽거나 써야 하는데 ASCII, UTF-8, UTF-16 과 같이 서로 다른 인코딩을 사용해야 한다.
#  ▣ 해결 :
with open('files/somefile.txt', 'rt') as f:  # 파일 전체를 하나의 문자열로 읽음
    data = f.read()

with open('files/somefile.txt', 'rt') as f:  # 파일의 줄을 순환
    for line in f:
        print(line)

with open('files/somefile.txt', 'wt') as f:
    f.write('text1')
    f.write('text2')

with open('files/somefile.txt', 'wt') as f:
    print('text1', file=f)
    print('text2', file=f)
#   - 기본적으로 파일을 읽고 쓸 때 sys.getdefaultencoding() 으로 확인할 수 있는 시스템 기본 인코딩을 사용한다.

#  ▣ 토론 : 예제에서 사용한 with 문이 파일을 사용할 콘텍스트를 만든다.
#           컨트롤이 with 블록을 떠나면 파일이 자동으로 닫힌다.
#           with 문을 꼭 사용하지 않아도 되지만, 그럴 때는 반드시 파일을 닫아야 한다.
f = open('files/somefile.txt', 'rt')
data = f.read()
f.close()

with open('files/somefile.txt', 'rt', newline='') as f:  # 줄 바꿈 변환 없이 읽기
    f.read()

#   - 인코딩 에러가 나는 경우(errors 로 처리)
f = open('files/somefile.txt', 'rt', encoding='ascii', errors='replace')  # errors='replace' : 치환
f = open('files/somefile.txt', 'rt', encoding='ascii', errors='ignore')  # errors='ignore' : 무시


#  5.2 파일에 출력
#  ▣ 문제 : print() 함수의 결과를 파일에 출력하고 싶다.
#  ▣ 해결 : print() 에 file 키워드 인자를 사용한다.
with open('PythonCookBook/files/somefile.txt', 'wt') as f:
    print('Hello World!', file=f)

#  ▣ 토론 : 파일을 텍스트 모드로 열었는지 꼭 확인해야 한다.
#           바이너리 모드로 파일을 열면 출력에 실패한다.


#  5.3 구별자나 종단 부호 바꾸기
#  ▣ 문제 : print() 를 사용해 데이터를 출력할 때 구분자나 종단 부호를 바꾸고 싶다.
#  ▣ 해결 : print() 에 sep 과 end 키워드 인자를 사용한다.
print('ACME', 50, 91.5)
print('ACME', 50, 91.5, sep=',')
print('ACME', 50, 91.5, sep=',', end='!!\n')

#   - 출력의 개행 문자를 바꿀 때도 end 인자를 사용한다.
for i in range(5):
    print(i)

for i in range(5):
    print(i, end=' ')

#  ▣ 토론 : print() 로 출력 시 아이템을 구분하는 문자를 스페이스 공백문 이외로 바꾸는 가장 쉬운 방법은 구별자를 지정하는 것이다.
print(','.join(['ACME', '50', '91.5']))  # str.join() 은 문자열에만 동작한다는 문제점이 있다.

#   - 문자열이 아닌 데이터에 사용하는 경우
row = ('ACME', 50, 91.5)
print(','.join(row))
print(','.join(str(x) for x in row))
print(*row, sep=',')  # 구별자를 사용


#  5.4 바이너리 데이터 읽고 쓰기
#  ▣ 문제 : 이미지나 사운드 파일 등 바이너리 데이터를 읽고 써야 한다.
#  ▣ 해결 : open() 함수에 rb 와 wb 모드를 사용해서 바이너리 데이터를 읽거나 쓴다.
with open('files/somefile.bin', 'rb') as f:  # 파일 전체를 하나의 바이트 문자열로 읽기
    data = f.read()

with open('files/somefile.bin', 'wb') as f:
    f.write(b'Hello World')

#  ▣ 토론 : 바이너리 데이터를 읽을 때, 바이너리 문자열과 텍스트 문자열 사이에 미묘한 문법 차이가 있다.
#           데이터에 인덱스나 순환으로 반환한 값은 바이트 문자열이 아닌 정수 바이트 값이 된다.
#   - 텍스트 문자열
t = 'Hello World'
print(t[0])  # 문자

for c in t:
    print(c)

#   - 바이트 문자열
b = b'Hello World'
print(b[0])  # 정수 바이트

for c in b:
    print(c)

#   - 바이너리 모드 파일로부터 텍스트를 읽거나 쓰려면 인코딩이나 디코딩 과정이 필요하다.
with open('files/somefile.bin', 'rb') as f:
    data = f.read(16)
    text = data.decode('utf-8')  #

with open('files/somefile.bin', 'wb') as f:
    text = 'Hello World'
    f.write(text.encode('utf-8'))

#   - 배열이나 C 구조체와 같은 객체를 bytes 객체로 변환하지 않고 바로 사용
import array
nums = array.array('i', [1, 2, 3, 4])
with open('PythonCookBook/files/data.bin', 'wb') as f:
    f.write(nums)

import array
a = array.array('i', [0, 0, 0, 0, 0, 0, 0, 0])
with open('PythonCookBook/files/data.bin', 'rb') as f:
    print(f.readinto(a))
print(a)


#  5.5 존재하지 않는 파일에 쓰기
#  ▣ 문제 : 파일이 파일 시스템에 존재하지 않을 때, 데이터를 파일에 쓰고 싶다.
#  ▣ 해결 : open() 에 x 모드를 사용해서 해결할 수 있다. w 모드와 다르게 x 모드는 잘 알려져 있지 않다.
with open('files/somefile.txt', 'wt') as f:
    f.write('Hello\n')

try:
    with open('files/somefile.txt', 'xt') as f:  # 존재하면 write 가 안된다.
        f.write('Hello\n')
except Exception as e:
    print(e)

#  ▣ 토론 : 이 레시피는 파일을 쓸 때 발생할 수 있는 문제점(실수로 파일을 덮어쓰는 등)을 아주 우아하게 피해 가는 법을 알려준다.
#           혹은 파일을 쓰기 전에 파일이 있는지 확인하는 방법도 있다.
import os
if not os.path.exists('files/somefile.txt'):
    with open('files/somefile.txt', 'wt') as f:
        f.write('Hello\n')
else:
    print('File already exists!')


#  5.6 문자열에 입출력 작업하기
#  ▣ 문제 : 파일 같은 객체에 동작하도록 작성한 코드에 텍스트나 바이너리 문자열을 제공하고 싶다.
#  ▣ 해결 : io.StringIO() 와 io.BytesIO() 클래스로 문자열 데이터에 동작하는 파일 같은 객체를 생성한다.
import io
s = io.StringIO()
s.write('Hello World\n')
print('This is a test', file=s)
print(s.getvalue())  # 기록한 모든 데이터 얻기

s = io.StringIO('Hello\nWorld\n')
print(s.read(4))
print(s.read())

#   - io.StringIO() 클래스는 텍스트에만 사용해야 한다. 바이너리 데이터를 다룰 때는 io.BytesIO() 클래스를 사용한다.
s = io.BytesIO()
s.write(b'binary data')
print(s.getvalue())

#  ▣ 토론 : 일반 파일 기능을 흉내 내려 할 때 StringIO() 와 BytesIO() 클래스가 가장 유용하다.
#           예를 들어 유닛 테스트를 할 때, StringIO() 로 테스트 데이터를 담고 있는 객체를 만들어 일반 파일에 동작하는 함수에 사용할 수 있다.
#           StringIO 와 BytesIO 인스턴스가 올바른 정수 파일 디스크립터를 가지고 있지 않다는 점을 기억하자.
#           따라서 file, pipe, socket 등 실제 시스템 레벨 파일을 요구하는 코드에는 사용할 수 없다.


#  5.7 압축된 데이터 파일 읽고 쓰기
#  ▣ 문제 : gzip 이나 bz2 로 압축한 파일을 읽거나 써야 한다.
#  ▣ 해결 : gzip 과 bz2 모듈을 사용하면 간단히 해결 가능하다.
#           이 모듈은 open() 을 사용하는 구현법의 대안을 제공한다.
#   - gzip 압축 데이터 읽기
import gzip
with gzip.open('files/somefile.gz', 'rt') as f:
    text = f.read()

#   - bz2 압축 데이터 읽기
import bz2
with bz2.open('files/somefile.bz2', 'rt') as f:
    text = f.read()

#   - gzip 압축 데이터 쓰기
import gzip
with gzip.open('files/somefile.gz', 'wt') as f:
    f.write(text)

#   - bz2 압축 데이터 읽기
import bz2
with bz2.open('files/somefile.bz2', 'wt') as f:
    f.write(text)

#  ▣ 토론 : 압축한 데이터를 읽거나 쓰기가 어렵지는 않다. 하지만, 올바른 파일 모드를 선택하는 것은 상당히 중요하다.
#           모드를 명시하지 않으면 기본적으로 바이너리 모드가 된다.
#           gzip.open() 과 bz2.open() 은 encoding, errors, newline 과 같이 내장 함수 open() 과 동일한 인자를 받는다.
#           압축한 데이터를 쓸 때는 compresslevel 인자로 압축 정도를 지정할 수 있다.
#           기본 레벨은 9로, 가장 높은 압축률을 가리킨다. 레벨을 내리면 속도는 더 빠르지만 압축률은 떨어진다.

#   - gzip.open() 과 bz2.open() 을 기존에 열려 있는 바이너리 파일의 상위에 위치시킨다.
#     gzip 과 bz2 모듈이 파일 같은 객체와 같이 작업할 수 있다.
import gzip

f = open('files/somefile.gz', 'rb')
with gzip.open(f, 'rt') as g:
    text = g.read()


#  5.8 고정 크기 레코드 순환
#  ▣ 문제 : 파일을 줄 단위로 순환하지 않고, 크기를 지정해서 그 단위별로 순환하고 싶다.
#  ▣ 해결 : iter() 함수와 functools.partial() 을 사용한다.
from functools import partial
RECORD_SIZE = 32
with open('PythonCookBook/files/winter.txt', 'rt') as f:
    records = iter(partial(f.read, RECORD_SIZE), '')  # partial 함수를 통해 RECORD_SIZE 만큼 부분적으로 만든다음
    for r in records:
        print(r)

#  ▣ 토론 : iter() 함수에 잘 알려지지 않은 기능으로, 호출 가능 객체와 종료 값을 전달하면 이터레이터를 만드는 것이 있다.
#           그 이터레이터는 제공 받은 호출 가능 객체를 반복적으로 호출하며 종료 값을 반환할 때 순환을 멈춘다.
#           고정 크기 단위로 파일을 읽는 작업은 주로 바이너리 모드에서 사용한다.


#  5.9 바이너리 데이터를 수정 가능한 버퍼에 넣기
#  ▣ 문제 : 바이너리 데이터를 읽어 수정 가능 버퍼에 넣을 때 어떠한 복사 과정도 거치고 싶지 않다.
#           그리고 그 데이터를 변형한 후 파일에 다시 써야 할지도 모른다.
#  ▣ 해결 : 데이터를 읽어 수정 가능한 배열에 넣으려면 readinto() 메소드를 사용한다.
import os.path

def read_into_buffer(filename):
    buf = bytearray(os.path.getsize(filename))
    with open(filename, 'rb') as f:
        f.readinto(buf)
    return buf

with open('PythonCookBook/files/sample.bin', 'wb') as f:
    f.write(b'Hello World')

buf = read_into_buffer('PythonCookBook/files/sample.bin')
buf[0:5] = b'Hallo'
print(buf)

with open('PythonCookBook/files/newsample.bin', 'wb') as f:
    f.write(buf)

#  ▣ 토론 : readinto() 메소드를 사용해서 미리 할당해 놓은 배열에 데이터를 채워 넣을 수 있다.
#           이때 array 모듈이나 numpy 와 같은 라이브러리를 사용해서 생성한 배열을 사용할 수도 있다.
#           새로운 객체를 할당하고 반환하는 일반적인 read() 메소드와는 다르게 readinto() 메소드는 기존의 버퍼에 내용을 채워 넣는다.
#           따라서 불필요한 메모리 할당을 피할 수 있다. 예를 들어 레코드 크기가 고정적인 바이너리 파일을 읽는다면 다음과 같은 코드를 작성할 수 있다.
record_size = 32

buf = bytearray(record_size)
with open('PythonCookBook/files/sample.bin', 'rb') as f:
    while True:
        n = f.readinto(buf)
        print(buf)

        if n < record_size:
            break

#   - 기존 버퍼의 제로-카피 조각을 만들 수 있고 기존의 내용은 수정하지 않는 메모리뷰를 사용
print(buf)
m1 = memoryview(buf)
m2 = m1[-5:]
print(m2)
m2[:] = b'WORLD'
print(buf)
#   - f.readinto() 를 사용할 때 반환 코드를 반드시 확인해야 한다. 반환 코드는 실제로 읽은 바이트 수가 된다.
#     "into" 형식의 다른 함수에도 관심을 갖도록 하자.(recv_into(), pack_into() 등)
#     파이썬에는 readinto() 외에도 직접 입출력 혹은 배열, 버퍼를 채우거나 수정하는 데 사용할 수 있도록 데이터에 대한 접근을 지원하는 것이 많다.


#  5.10 바이너리 파일 메모리 매핑
#  ▣ 문제 : 바이너리 파일을 수정 가능한 바이트 배열에 매핑하고, 내용에 접근하거나 수정하고 싶다.
#  ▣ 해결 : mmap 모듈을 사용해서 파일을 메모리 매핑한다.
import os
import mmap

def memory_map(filename, access=mmap.ACCESS_WRITE):  # 이 함수를 사용하려면, 데이터로 채워진 파일이 있어야 한다.
    size = os.path.getsize(filename)
    fd = os.open(filename, os.O_RDWR)
    return mmap.mmap(fd, size, access=access)

size = 1000000
with open('PythonCookBook/files/sample.bin', 'wb') as f:
    f.seek(size-1)
    f.write(b'\x00')

m = memory_map('PythonCookBook/files/sample.bin')  # memory_map() 함수로 메모리 매핑 수행
print(len(m))
print(m[0:10], m[0])
m[0:11] = b'Hello World'  # 슬라이스 재할당
m.close()

with open('PythonCookBook/files/sample.bin', 'rb') as f:
    print(f.read(11))

with memory_map('PythonCookBook/files/sample.bin') as m:  # 컨텍스트 매니저를 사용
    print(len(m))
    print(m[0:10])

m1 = memory_map('PythonCookBook/files/sample.bin', mmap.ACCESS_READ)  # 읽기 전용으로 파일 오픈
m2 = memory_map('PythonCookBook/files/sample.bin', mmap.ACCESS_COPY)  # 지역 레벨에서 수정하고, 원본에는 영향을 주고 싶지 않을 경우

#  ▣ 토론 : mmap 으로 파일을 메모리에 매핑하면 파일 내용에 매우 효율적으로 무작위로 접근할 수 있다.
#           예를 들어 파일을 열고 seek(), read(), write() 호출을 번갈아 가며 해야 할 일을 파일에 매핑해 놓고 자르기 연산으로 쉽게 해결할 수 있다.
m = memory_map('PythonCookBook/files/sample.bin')
v = memoryview(m).cast('I')  # unsigned integer 의 메모리뷰
v[0] = 7
print(m[0:4])
m[0:4] = b'\x07\x01\x00\x00'
print(v[0])


#  5.11 경로 다루기
#  ▣ 문제 : 기본 파일 이름, 디렉터리 이름, 절대 경로 등을 찾기 위해 경로를 다루어야 한다.
#  ▣ 해결 : 경로를 다루기 위해서 os.path 모듈의 함수를 사용한다. 몇몇 기능을 예제를 통해 살펴보자
import os
path = '/Users/beazley/Data/data.csv'
print(os.path.basename(path))  # 경로의 마지막 부분 구하기
print(os.path.dirname(path))  # 디렉터리 이름 구하기
print(os.path.join('tmp', 'data', os.path.basename(path)))  # 각 부분을 합치기
path = '~/Data/data.csv'
print(os.path.expanduser(path))  # 사용자의 홈 디렉토리 펼치기
print(os.path.splitext(path))  # 파일 확장자 나누기
print(os.path.split(path))  # 디렉토리와 파일 나누기

#  ▣ 토론 : 파일 이름을 다루기 위해서 문자열에 관련된 코드를 직접 작성하지 말고 os.path 모듈을 사용해야 한다.
#           os.path 모듈은 Unix 와 Windows 의 차이점을 알고 Data/data.csv 와 Data\data.csv 의 차이점을 자동으로 처리한다.


#  5.12 파일 존재 여부 확인
#  ▣ 문제 : 파일이나 디렉터리가 존재하는지 확인해야 한다.
#  ▣ 해결 : 파일이나 디렉터리의 존재 여부를 확인하기 위해서 os.path 모듈을 사용한다.
import os
print(os.path.exists('PythonCookBook/files/somefile.txt'))  # 파일 존재 여부
print(os.path.exists('PythonCookBook/files'))  # 디렉터리 존재 여부

#   - 파일의 종류가 무엇인지 확인
print(os.path.isfile('PythonCookBook/files/somefile.txt'))  # 일반 파일인지 확인
print(os.path.isdir('PythonCookBook/files/somefile.txt'))  # 디렉터리인지 확인
print(os.path.islink('PythonCookBook/files/somefile.txt'))  # 심볼릭 링크인지 확인
print(os.path.realpath('PythonCookBook/files/somefile.txt'))  # 절대 경로 얻기

#   - 메타데이터(파일 크기, 수정 날짜) 등이 필요할 때도 os.path 모듈을 사용한다.
print(os.path.getsize('PythonCookBook/files/somefile.txt'))  # 파일 크기
print(os.path.getmtime('PythonCookBook/files/somefile.txt'))  # 수정 날짜

import time
print(time.ctime(os.path.getmtime('PythonCookBook/files/somefile.txt')))

#  ▣ 토론 : os.path 를 사용하면 파일 테스팅은 그리 어렵지 않다. 유의해야 할 점은 아마도 파일 권한에 관련된 것뿐이다.


#  5.13 디렉터리 리스팅 구하기
#  ▣ 문제 : 디렉터리나 파일 시스템 내부의 파일 리스트를 구하고 싶다.
#  ▣ 해결 : os.listdir() 함수로 디렉터리 내에서 파일 리스트를 얻는다.
import os
names = os.listdir('PythonCookBook/files/')  # listdir() : 디렉터리 내에서 파일 리스트를 출력
print(names)

#   - 데이터를 걸러 내야 한다면 os.path 라이브러리의 파일에 리스트 컴프리헨션을 사용한다.
import os.path
names = [name for name in os.listdir('PythonCookBook/files/') if os.path.isfile(os.path.join('PythonCookBook/files/', name))]  # 일반 파일 모두 구하기
print(names)

dirnames = [name for name in os.listdir('PythonCookBook/') if os.path.isdir(os.path.join('PythonCookBook', name))]  # 디렉터리 모두 구하기
print(dirnames)

#   - 문자열의 startswith() 와 endswith() 메소드를 사용하면 디렉터리의 내용을 걸러 내기 유용하다.
pyfiles = [name for name in os.listdir('PythonCookBook/files/') if name.endswith('.py')]

#   - 파일 이름 매칭을 하기 위해 glob 이나 fnmatch 모듈을 사용한다.
import glob
binfiles = glob.glob('PythonCookBook/files/*.bin')
print(binfiles)

from fnmatch import fnmatch
pyfiles = [name for name in os.listdir('PythonCookBook/') if fnmatch(name, '*.py')]
print(pyfiles)


#  5.14 파일 이름 인코딩 우회
#  ▣ 문제 : 시스템의 기본 인코딩으로 디코딩 혹은 인코딩되지 않은 파일 이름에 입출력 작업을 수행해야 한다.
#  ▣ 해결 : 기본적으로 모든 파일 이름은 sys.getfilesystemencoding() 이 반환하는 텍스트 인코딩 값으로 디코딩 혹은 인코딩 되어 있다.
#           하지만 이 인코딩을 우회하길 바란다면 raw 바이트 문자열로 파일 이름을 명시해야 한다.
import sys
print(sys.getfilesystemencoding())

#   - 유니코드로 파일 이름을 쓴다.
with open('PythonCookBook/files/jalape\xf1o.txt', 'w', encoding='utf-8') as f:
    f.write('Spicy!')

#   - 디렉터리 리스트 (디코딩됨)
import os
print(os.listdir('PythonCookBook/files/'))

#   - 디렉터리 리스트 (디코딩되지 않음)
print(os.listdir(b'PythonCookBook/files/'))

#   - raw 파일 이름으로 파일 열기
with open(b'PythonCookBook/files/.txt') as f:
    print(f.read())

#  ※ 마지막 두 작업에 나온 것처럼, open() 이나 os.listdir() 와 같은 파일 관련 함수에 바이트 문자열을 넣었을 때 파일 이름 처리는 거의 변하지 않는다.

#  ▣ 토론 : 파일 이름과 디렉터리를 읽을 때 디코딩되지 않은 raw 바이트를 이름으로 사용하면 이런 문제점을 피해 갈 수 있다.


#  5.15 망가진 파일 이름 출력
#  ▣ 문제 : 프로그램에서 디렉터리 리스트를 받아 파일 이름을 출력하려고 할 때, UnicodeEncodeError 예외와 "surrogates not allowed" 메시지가
#           발생하면서 프로그램이 죽어 버린다.
#  ▣ 해결 : 출처를 알 수 없는 파일 이름을 출력할 때, 다음 코드로 에러를 방지한다.
def bad_filename(filename):
    return repr(filename)[1:-1]

filenames = os.listdir('PythonCookBook/files/')
for filename in filenames:
    try:
        print(filename)
    except UnicodeEncodeError:
        print(bad_filename(filename))
#  ※ os.listdir() 와 같은 명령을 실행할 때, 망가진 파일 이름을 사용하면 파이썬에 문제가 생긴다.
#     해결책은 디코딩할 수 없는 바이트 값 \xhh 를 Unicode 문자 \udchh 로 표현하는 소위 "대리 인코딩"으로 매핑하는 것이다.

#  ▣ 토론 : UTF-8 이 아닌 Latin-1 으로 인코딩한 bad.txt 를 포함한 디렉터리 리스트가 어떻게 보이는지 예제를 보자.
filename = 'bad.txt'.encode('Latin-1')
with open(b'PythonCookBook/files/'+filename, 'wt', encoding='Latin-1') as f:
    f.write('test')

import os
files = os.listdir('PythonCookBook/files/')
print(files)
#   ※ Latin-1 로 인코딩한 bad.txt 를 출력하려할때 프로그램이 비정상적으로 종료된다.
#      따라서 아래와 같이 출력해야한다.
filenames = os.listdir('PythonCookBook/files/')
for filename in filenames:
    try:
        print(filename)
    except UnicodeEncodeError:
        print(bad_filename(filename))

#   - 아래와 같이 bad_filename() 함수안에서 잘못된 인코딩 된 값을 재인코딩할 수 있다.
def bad_filename(filename):
    temp = filename.encode(sys.getfilesystemencoding(), errors='surrogateescape')
    return temp.decode('latin-1')

for filename in filenames:
    try:
        print(filename)
    except UnicodeEncodeError:
        print(bad_filename(filename))


#  5.16 이미 열려 있는 파일의 인코딩을 수정하거나 추가하기
#  ▣ 문제 : 이미 열려 있는 파일을 닫지 않고 Unicode 인코딩을 추가하거나 변경하고 싶다.
#  ▣ 해결 : 바이너리 모드로 이미 열려 있는 파일 객체를 닫지 않고 Unicode 인코딩/디코딩을 추가하고 싶다면 그 객체를 io.TextIOWrapper()
#           객체로 감싼다.
import urllib.request
import io

u = urllib.request.urlopen('http://www.python.org')
f = io.TextIOWrapper(u, encoding='utf-8')  # 기존 byte 형태인 것을 utf-8 로 변경
text = f.read()
print(text)

#   - 텍스트 모드로 열린 파일의 인코딩을 변경하려면 detach() 메소드로 텍스트 인코딩 레이어를 제거하고 다른 것으로 치환한다.
import sys
print(sys.stdout.encoding)
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='latin-1')
print(sys.stdout.encoding)
#   ※ 위 코드를 실행하면 터미널의 출력이 망가질 수도 있다.

#  ▣ 토론 : I/O 시스템은 여러 레이어로 만들어져 있다. 다음 간단한 코드를 통해 레이어를 볼 수 있다.
f = open('PythonCookBook/files/sample.txt', 'w', encoding='utf-8')
print(f)  # io.TextIOWrapper 는 Unicode 를 인코딩/디코딩하는 텍스트 처리 레이어
print(f.buffer)  # io.BufferedWriter 는 바이너리 데이터를 처리하는 버퍼 I/O 레이어
print(f.buffer.raw)  # io.FileIO 는 운영체제에서 하위 레벨 파일 디스크립터를 표현하는 raw file

#   - 일반적으로 앞에 나타난 속성에 접근해 레이어를 직접 수정하는 것은 안전하지 않다.
print(f)
f = io.TextIOWrapper(f.buffer, encoding='latin-1')
print(f)
f.write('Hello')  # ValueError: I/O operation on closed file.
#   ※ f의 원본 값이 파괴되고 프로세스의 기저 파일을 닫았기 때문에 제대로 동작하지 않는다.
#      detach() 메소드는 파일의 최상단 레이어를 끊고 그 다음 레이어를 반환한다.
#      그 다음에 상단 레이어를 더 이상 사용할 수 없다.
f = open('PythonCookBook/files/sample.txt', 'w', encoding='utf-8')
print(f)
b = f.detach()
print(b)
f.write('hello')  # ValueError: underlying buffer has been detached

#   - 하지만 연결을 끊은 후에는, 반환된 결과에 새로운 상단 레이어를 추가할 수 있다.
f = io.TextIOWrapper(b, encoding='latin-1')
print(f)

#   - 인코딩을 변경하는 방법을 보였지만, 이 기술을 라인 처리, 에러 규칙 등 파일 처리의 다른 측면에 활용할 수 있다.
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='ascii', errors='xmlcharrefreplace')
print('Jalape\u00f1o')


#  5.17 텍스트 파일에 바이트 쓰기
#  ▣ 문제 : 텍스트 모드로 연 파일에 로우 바이트를 쓰고 싶다.
#  ▣ 해결 : 단순히 바이트 데이터를 buffer 에 쓴다.
import sys
sys.stdout.write(b'Hello\n')
sys.stdout.buffer.write(b'Hello\n')
#  ※ 이와 유사하게, 텍스트 파일의 buffer 속성에서 바이너리 데이터를 읽을 수도 있다.

#  ▣ 토론 : I/O 시스템은 레이어로부터 만들어진다.
#           텍스트 파일은 버퍼 바이너리 모드 파일 상단에 Unicode 인코딩/디코딩 레이어를 추가해서 생성된다.
#           buffer 속성은 바로 이 파일 아래 부분을 가리킨다.
#           여기에 접근하면 텍스트 인코딩/디코딩 레이어를 우회할 수 있다.


#  5.18 기존 파일 디스크립터를 파일 객체로 감싸기
#  ▣ 문제 : 운영 체제 상에 이미 열려 있는 I/O 채널에 일치하는 정수형 파일 디스크립터를 가지고 있고(file, pipe, socket 등), 이를
#           상위 레벨 파이썬 파일 객체로 감싸고 싶다.
#  ▣ 해결 : 파일 디스크립터는 운영 체제가 할당한 정수형 핸들로 시스템 I/O 채널 등을 참조하기 위한 목적으로써 일반 파일과는 다르다.
#           파일 디스크립터가 있을 때 open() 함수를 사용해 파이썬 파일 객체로 감쌀 수 있다.
#           하지만 이때 파일 이름 대신 정수형 파일 디스크립터를 먼저 전달해야 한다.

# ★ 파일 디스크립터
#  - 파일을 관리하기 위해 운영체제가 필요로 하는 파일의 정보를 가지고 있는 것이다.
#    FCB(File Control Block)이라고 하며 FCB 에는 다음과 같은 정보들이 저장되어 있다.
#   1. 파일 이름
#   2. 보조기억장치에서의 파일 위치
#   3. 파일 구조 : 순차 파일, 색인 순차 파일, 색인 파일
#   4. 액세스 제어 정보
#   5. 파일 유형
#   6. 생성 날짜와 시간, 제거 날짜와 시간
#   7. 최종 수정 날짜 및 시간
#   8. 액세스한 횟수
#  - 결론은 파일 디스크립터란 운영체제가 만든 파일 또는 소켓의 지칭을 편히 하기 위해서 부여된 숫자이다.
#  - 기본적으로 파일 디스크립터는 정수형으로 차례로 넘버링 되고 0,1,2 는 이미 할당되어 있어서 3 부터 디스크립터를 부여한다.

#   - 하위 레벨 파일 디스크립터 열기
import os
fd = os.open('PythonCookBook/files/somefile.txt', os.O_WRONLY | os.O_CREAT)

#   - 올바른 파일로 바꾸기
f = open(fd, 'wt')
f.write('hello world\n')
f.close()

#   - 상위 레벨 파일 객체가 닫혔거나 파괴되었다면, 그 하단 파일 디스크립터 역시 닫힌다.
#     이런 동작을 원하지 않는다면 closefd=False 인자를 open() 에 전달해야 한다.
f = open(fd, 'wt', closefd=False)

#  ▣ 토론 : Unix 시스템 상에서 이 기술을 사용하면 기존의 I/O 채널(pipe, socket 등)을 감싸 파일과 같은 인터페이스로 사용할 수 있는
#           쉬운 길이 열린다.
from socket import socket, AF_INET, SOCK_STREAM

def echo_client(client_sock, addr):
    print('Got connection from', addr)

    # 읽기/쓰기를 위해 소켓에 대한 텍스트 모드 파일 래퍼(wrapper)를 만든다.
    client_in = open(client_sock.fileno(), 'rt', encoding='latin-1', closefd=False)
    client_out = open(client_sock.fileno(), 'wt', encoding='latin-1', closefd=False)

    # 파일 I/O를 사용해 클라이언트에 라인을 에코한다.
    for line in client_in:
        client_out.write(line)
        client_out.flush()
    client_sock.close()

def echo_server(address):
    sock = socket(AF_INET, SOCK_STREAM)
    sock.bind(address)
    sock.listen(1)
    while True:
        client, addr = sock.accept()
        echo_client(client, addr)
#  ※ 앞에 나온 예제는 내장 함수 open() 의 기능을 보이기 위한 목적으로 작성한 것이고 Unix 기반 시스템에서만 동작한다.
#     소켓에 대한 파일 같은 인터페이스가 필요하고 크로스 플랫폼 코드가 필요하다면 소켓의 makefile() 메소드를 사용해야 한다.
#     하지만 이식성을 신경쓰지 않는다면 makefile() 을 사용하는 것보다 앞에 나온 예제가 성능 면에서 훨씬 뛰어나다.

#   - stdout 에 바이너리 데이터를 넣기 위한 파일 객체를 만드는 방법
import sys
bstdout = open(sys.stdout.fileno(), 'wb', closefd=False)
bstdout.write(b'Hello World\n')
bstdout.flush()
#   ※ 기존 파일 디스크립터를 파일로 감싸는 것도 가능하지만, 모든 파일 모드를 지원하지 않을 수 있고 이런 파일 디스크립터에 예상치 못한 부작용이 생길 수 있다.
#      또한 동작성이 운영 체제에 따라 달라지기도 한다.
#      예를 들어 앞에 나온 모든 예제는 Unix 가 아닌 시스템에서 아마도 동작하지 않을 것이다.


#  5.19 임시 파일과 디렉터리 만들기
#  ▣ 문제 : 임시 파일이나 디렉터리를 만들어 프로그램에 사용해야 한다.
#           그 후에 파일이나 디렉터리는 아마도 파기할 생각이다.
#  ▣ 해결 : tempfile 모듈에 이런 목적의 함수가 많이 있다. 이름 없는 임시 파일을 만들기 위해서 tempfile.TemporaryFile 을 사용한다.
from tempfile import TemporaryFile

with TemporaryFile('w+t') as f:  # with 문 종료 시 임시 파일은 파기된다.
    # 파일에서 읽기/쓰기
    f.write('Hello World\n')
    f.write('Testing\n')

    # 처음으로 이동해 데이터를 읽는다.
    f.seek(0)
    data = f.read()

#   - 원한다면 다음과 같이 파일을 사용할 수도 있다.
f = TemporaryFile('w+t')
f.close()

with TemporaryFile('w+t', encoding='utf-8', errors='ignore') as f:  # TemporaryFile() 은 추가적으로 내장 함수 open() 과 동일한 인자를 받는다
    f.write('aaa')

#   - 대개 Unix 시스템에서 TemporaryFile() 로 생성한 파일에 이름이 없고 디렉터리 엔트리도 갖지 않는다.
#     이 제한을 없애고 싶으면 NamedTemporaryFile() 을 사용한다.
from tempfile import NamedTemporaryFile

with NamedTemporaryFile('w+t') as f:
    print('filename is : ', f.name)

#   - 자동으로 tempfile 이 삭제되는 걸 원하지 않는 경우 delete=False 키워드 인자를 사용한다.
with NamedTemporaryFile('w+t', delete=False) as f:
    print('filename is:', f.name)

#   - 임시 디렉토리를 만들기 위해서는 tempfile.TemporaryDirectory() 를 사용한다.
from tempfile import TemporaryDirectory
with TemporaryDirectory() as dirname:
    print('dirname is :', dirname)

#  ▣ 토론 : 임시 파일과 디렉터리를 만들 때 TemporaryFile(), NamedTemporaryFile(), TemporaryDirectory() 함수가 가장 쉬운 방법이다.
#            이 함수는 생성과 추후 파기까지 모두 자동으로 처리해 준다.
#            더 하위 레벨로 내려가면 mkstemp() 와 mkdtemp() 로 임시 파일과 디렉터리를 만들 수 있다.
import tempfile
print(tempfile.mkstemp())
print(tempfile.mkdtemp())
#  ※ mkstemp() 함수는 단순히 raw OS 파일 디스크립터를 반환할 뿐 이를 올바른 파일로 바꾸는 것은 프로그래머의 역할로 남겨 둔다.
#     이와 유사하게 파일을 제거하는 것도 독자에게 달려 있다.

#   - 일반적으로 임시 파일은 /var/tmp 와 같은 시스템의 기본 위치에 생성된다.
#     실제 위치를 찾으려면 tempfile.gettempdir() 함수를 사용한다.
print(tempfile.gettempdir())

#   - 모든 임시 파일 관련 함수는 디렉터리와 이름 규칙을 오버라이드 할 수 있도록 한다.
#     prefix, suffix, dir 키워드 인자를 사용하면 된다.
f = NamedTemporaryFile(prefix='mytemp', suffix='.txt', dir='C:\\Users\\kyh\\AppData\\Local\\Temp\\')
print(f.name)

#   - 마지막으로 tempfile() 은 가장 안전한 방식으로 파일을 생성한다는 점을 기억하자.
#     예를 들어 파일에 접근할 수 있는 권한은 현재 사용자에게만 주고, 파일 생성에서 레이스 컨디션이 발생하지 않도록 한다.


#  5.20 시리얼 포트와 통신
#  ▣ 문제 : 시리얼 포트를 통해 하드웨어 디바이스(로봇, 센서 등)와 통신하고 싶다.
#  ▣ 해결 : 파이썬의 내장 기능으로 직접 해결할 수도 있지만, 그보다는 pySerial 패키지를 사용하는 것이 더 좋다.
# import serial
# ser = serial.Serial('/dev/tty.usbmodem641',
#                     baudrate=9600,
#                     bytesize=8,
#                     parity='N',
#                     stopbits=1)
#  ※ 디바이스 이름은 종류나 운영 체제에 따라 달라진다. 예를 들어 Windows 에서 0, 1 등을 사용해서 "COM0", "COM1" 과 같은 포트를 연다.
#     열고 나서 read(), readline(), write() 호출로 데이터를 읽고 쓴다.
# ser.write(b'G1 X50 Y50\r\r')
# resp = ser.readline()

#  ▣ 토론 : 겉보기에는 시리얼 통신이 간단해 보이지만 때로 복잡해지는 경우가 있다.
#            pySerial 과 같은 패키지를 사용해야 하는 이유로 고급 기능(타임 아웃, 컨트롤 플로우, 버퍼 플러싱, 핸드셰이킹 등)을 지원한다는 점이 있다.
#            시리얼 포트와 관련된 모든 입출력은 바이너리임을 기억하자.
#            따라서 코드를 작성할 때 텍스트가 아닌 바이트를 사용하도록 해야 한다.
#            그리고 바이너리 코드 명령이나 패킷을 만들 때 struct 모듈을 사용하면 편리하다.


#  5.21 파이썬 객체를 직렬화 하기
#  ▣ 문제 : 파이썬 객체를 바이트 스트림에 직렬화시켜 파일이나 데이터베이스에 저장하거나 네트워크를 통해 전송하고 싶다.
#  ▣ 해결 : 데이터 직렬화를 위한 가장 일반적인 접근은 pickle 모듈을 사용하는 것이다.
import pickle

data = ['test', 'test1', 'test2', 'test3', 'test4']
f = open('PythonCookBook/files/somefile.bin', 'wb')
pickle.dump(data, f)

#   - 객체를 문자열에 덤프하려면 pickle.dumps() 를 사용한다.
s = pickle.dumps(data)
print(s)

#   - 바이트 스트림으로부터 객체를 다시 만들기 위해서 pickle.load() 나 pickle.loads() 함수를 사용한다.
f = open('PythonCookBook/files/somefile.bin', 'rb')
data = pickle.load(f)
print(data)

#   - 문자열에서 불러들이기
data = pickle.loads(s)
print(data)

#  ▣ 토론 : 대부분의 프로그램에서 pickle 을 효율적으로 사용하기 위해서는 dump() 와 load() 함수만 잘 사용하면 된다.
#            파이썬 객체를 데이터베이스에 저장하거나 불러오고, 네트워크를 통해 전송하는 라이브러리를 사용한다면 내부적으로
#            pickle 을 사용하고 있을 확률이 크다.

#   - 다중 객체와 작업
import pickle
f = open('PythonCookBook/files/somefile.bin', 'wb')
pickle.dump([1, 2, 3, 4], f)
pickle.dump('hello', f)
pickle.dump({'Apple', 'Pear', 'Banana'}, f)
f.close()
f = open('PythonCookBook/files/somefile.bin', 'rb')
print(pickle.load(f))
print(pickle.load(f))
print(pickle.load(f))

#   - 함수, 클래스, 인스턴스를 피클할 수 있지만 결과 데이터는 코드 객체와 관련 있는 이름 참조만 인코드한다.
import math
print(pickle.dumps(math.cos))

#  ※ pickle.load() 는 믿을 수 없는 데이터에 절대 사용하면 안 된다.
#     로딩의 부작용으로 pickle 이 자동으로 모듈을 불러오고 인스턴스를 만든다.
#     하지만 악의를 품은 사람이 이 동작을 잘못 사용하면 일종의 바이러스 코드를 만들어 파이썬이 자동으로 실행하도록 할 수 있다.
#     따라서 서로 인증을 거친 믿을 수 있는 시스템끼리 내부적으로만 pickle 을 사용하는 것이 좋다.

#  ※ 피클할 수 없는 객체
#   - 파일, 네트워크 연결, 스레드, 프로세스, 스택 프레임 등 외부 시스템 상태와 관련 있는 것들이 포함.

#   - 사용자 정의 클래스에 __getstate__() 와 __setstate__() 메소드를 제공하면 이런 제약을 피해 갈 수 있다.
#     정의를 했다면 pickle.dump() 는 __getstate__() 를 호출해 피클할 수 있는 객체를 얻는다.
#     마찬가지로 __setstate__() 는 언피클을 유발한다.

#   - 클래스 내부에 스레드를 정의하지만 피클/언피클 할 수 있는 예제를 보자.
from PythonCookBook import countdown
c = countdown.Countdown(30)

f = open('PythonCookBook/files/cstate.p', 'wb')
import pickle
pickle.dump(c, f)
f.close()

#   - 이제 파이썬을 종료하고 재시작한 후에 다음을 실행한다.
f = open('PythonCookBook/files/cstate.p', 'rb')
pickle.load(f)
#  ※ 쓰레드가 다시 살아나서 처음으로 피클했을 때 종료했던 곳부터 시작하는 것을 볼 수 있다.
#     pickle 은 array 모듈이나 numpy 와 같은 라이브러리가 만들어 낸 거대한 자료 구조에 사용하기에 효율적인 인코딩 방식이 아니다.
#     pickle 에는 아주 많은 옵션이 있고 주의해야 할 점도 많다.
#     일반적인 경우에 이런 것을 걱정할 필요는 없지만 직렬화를 위해 pickle 을 사용하는 큰 애플리케이션을 만든다면 공식 문서를 통해
#     이와 같은 내용을 잘 숙지하도록 하자.