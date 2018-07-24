# Chapter 2. 문자열과 텍스트
#  2.1 여러 구분자로 문자열 나누기
#  ▣ 문제 : 문자열을 필드로 나누고 싶지만 구분자가 문자열에 일관적이지 않다.
#  ▣ 해결 : 문자열 객체의 split() 메소드는 아주 간단한 상황에 사용하도록 설계되었고 여러 개의 구분자나 구분자 주변의
#            공백까지 고려하지는 않는다. 좀 더 유연해져야 할 필요가 있다면 re.split() 메소드를 사용한다.
line = 'asdf fjdk; afed,  fjek,asdf,      foo'
import re
print(re.split(r'[;,\s]\s*', line))

#  ▣ 토론 : re.split() 함수는 분리 구문마다 여러 패턴을 명시할 수 있다는 점이 유리하다.
#   - re.split() 을 사용할 때는 괄호 안에 묶인 정규 표현식 패턴이 캡처 그룹이 된다.
fields = re.split(r'(;|,|\s)\s*', line)
print(fields)

#   - 구분 문자만 출력하는 경우.
values = fields[::2]
delimiters = fields[1::2] + ['']
print(values, delimiters)

#   - 동일한 구분자로 라인을 구성한다.
print(''.join(v+d for v, d in zip(values, delimiters)))

#   - 분리 구문을 결과에 포함시키고 싶지 않지만 정규 표현식에 괄호를 사용해야 할 필요가 있다면 논캡처 그룹을 사용한다.
print(re.split('(?:,|;|\s)\s*', line))


#  2.2 문자열 처음이나 마지막에 텍스트 매칭
#  ▣ 문제 : 문자열의 처음이나 마지막에 파일 확장자, URL scheme 등 특정 텍스트 패턴이 포함되었는지 검사하고 싶다.
#  ▣ 해결 : 문자열의 처음이나 마지막에 패턴이 포함되었는지 확인하는 간단한 방법으로 str.startswith() 나 str.endswith() 메소드가 있다.
filename = 'spam.txt'
print(filename.endswith('.txt'))
print(filename.startswith('file:'))
url = 'http://www.python.org'
print(url.startswith('http:'))

#   - 여러 개의 선택지를 검사해야 한다면 검사하고 싶은 값을 튜플에 담아 startswith() 나 endswith() 에 전달한다.
import os
filenames = os.listdir('.')
print(filenames)
print([name for name in filenames if name.endswith(('.c', '.h'))])

from urllib.request import urlopen

def read_data(name):
    if name.startswith(('http:', 'https:', 'ftp:')):
        return urlopen(name).read()
    else:
        with open(name) as f:
            return f.read()

#   - startswith() 메소드는 튜플만을 입력으로 받는다.
choices = ['http:', 'ftp:']
url = 'http://www.python.org'
url.startswith(choices)
url.startswith(tuple(choices))

#  ▣ 토론 : startswith() 와 endswith() 메소드는 접두어와 접미어를 검사할 때 매우 편리하다.
#            슬라이스를 사용하면 비슷한 동작을 할 수 있지만 코드의 가독성이 많이 떨어진다.
filename = 'spam.txt'
print(filename[-4:] == '.txt')
url = 'http://www.python.org'
print(url[:5] == 'http:' or url[:6] == 'https:' or url[:4] == 'ftp:')

#   - 정규 표현식을 사용해도 된다.
import re
url = 'http://www.python.org'
print(re.match('http:|https:|ftp:', url))

#   - startswith() 와 endswith() 메소드는 일반적인 데이터 감소와 같은 다른 동작에 함께 사용하기에도 좋다.
if any(name.endswith(('.c', '.h')) for name in os.listdir('.')):
    pass


#  2.3 쉘 와일드카드 패턴으로 문자열 매칭
#  ▣ 문제 : Unix 쉘에 사용하는 것과 동일한 와일드카드 패턴을 텍스트 매칭에 사용하고 싶다.(예: *.py, Dat[0-9]*.csv 등)
#  ▣ 해결 : fnmatch 모듈에 두 함수 fnmatch() 와 fnmatchcase() 를 사용하면 된다.
from fnmatch import fnmatch, fnmatchcase
print(fnmatch('foo.txt', '*.txt'))
print(fnmatch('foo.txt', '?oo.txt'))
print(fnmatch('Dat45.csv', 'Dat[0-9]*'))
names = ['Dat1.csv', 'Dat2.csv', 'config.ini', 'foo.py']
print([name for name in names if fnmatch(name, 'Dat*.csv')])

#   - 일반적으로 fnmatch() 는 시스템의 파일 시스템과 동일한 대소문자 구문 규칙을 따른다.
print(fnmatch('foo.txt', '*.TXT'))  # Mac    : True
print(fnmatch('foo.txt', '*.TXT'))  # window : True

#   - 이런 차이점이 없는 것을 사용하려면 fnmatchcase() 를 사용한다.
print(fnmatchcase('foo.txt', '*.TXT'))

#   - 파일 이름이 아닌 데이터 프로세싱에도 사용할 수 있다.
addresses = ['5412 N CLARK ST',
             '1060 W ADDISON ST',
             '1039 W GRANVILLE AVE',
             '2122 N CLARK ST',
             '4802 N BROADWAY']
from fnmatch import fnmatchcase
print([addr for addr in addresses if fnmatchcase(addr, '* ST')])
print([addr for addr in addresses if fnmatchcase(addr, '54[0-9][0-9] *CLARK*')])

#  ▣ 토론 : fnmatch 가 수행하는 매칭은 간단한 문자열 메소드의 기능과 정규 표현식의 중간쯤 위치하고 있다.
#           데이터 프로세싱을 할 때 간단한 와일드카드를 사용할 생각이라면 이 함수를 사용하는 것이 괜찮은 선택이다.


#  2.4 텍스트 패턴 매칭과 검색
#  ▣ 문제 : 특정 패턴에 대한 텍스트 매칭이나 검색을 하고 싶다.
#  ▣ 해결 : 매칭하려는 텍스트가 간단하다면 str.find(), str.endswith(), str.startswith() 와 같은 기본적인 문자열 메소드만으로도
#           충분하다.
text = 'yeah, but no, but yeah, but no, but yeah'

#   - 정확한 매칭
print(text == 'yeah')

#   - 처음이나 끝에 매칭
print(text.startswith('yeah'))
print(text.endswith('no'))

#   - 처음 나타난 곳 검색
print(text.find('no'))  # 해당 값이 처음 나온 인덱스를 리턴.

#   - 좀 더 복잡한 매칭을 위해 정규 표현식과 re 모듈을 사용한다.
#     match() : 항상 문자열 처음에서 찾기를 시도.
#     findall() : 텍스트 전체에 걸쳐 찾기를 시도.
text1 = '11/27/2012'
text2 = 'Nov 27, 2012'

import re
#   - 간단한 매칭: \d+는 하나 이상의 숫자를 의미
if re.match(r'\d+/\d+/\d+', text1):
    print('yes')
else:
    print('no')

if re.match(r'\d+/\d+/\d+', text2):
    print('yes')
else:
    print('no')

#   - 동일한 패턴으로 매칭을 많이 수행할 예정이라면 정규 표현식을 미리 컴파일해서 패턴 객체로 만들어 놓는다.
datepat = re.compile(r'\d+/\d+/\d+')
if datepat.match(text1):
    print('yes')
else:
    print('no')

if datepat.match(text2):
    print('yes')
else:
    print('no')

#   - 텍스트 전체에 걸쳐 패턴을 찾으려면 findall() 메소드를 사용한다.
text = 'Today is 11/27/2012. PyCon starts 3/13/2013.'
print(datepat.findall(text))

#   - 정규 표현식을 정의할 때 괄호를 사용해 캡처 그룹을 만드는 것이 일반적이다.
datepat = re.compile(r'(\d+)/(\d+)/(\d+)')
m = datepat.match('11/27/2012')
print(m.group(0), m.group(1), m.group(2), m.group(3), m.groups())  # 그룹별로 출력 가능 (0 은 전체 출력)
month, day, year = m.groups()
for month, day, year in datepat.findall(text):
    print('{}-{}-{}'.format(year, month, day))

#   - 한 번에 결과를 얻지 않고 텍스트를 순환하며 찾으려면 finditer() 를 사용한다.
for m in datepat.finditer(text):
    print(m.groups())

#  ▣ 토론 : 핵심이 되는 기능은 re.compile() 을 사용해 패턴을 컴파일하고 그것을 match(), findall(), finditer() 등에 사용한다.
#           패턴을 명시할 때 r'(\d+)/(\d+)/(\d+)'와 같이 raw string 을 그대로 쓰는것이 일반적이다.
#           이 형식은 백슬래시 문자를 해석하지 않고 남겨 두기 때문에 정규 표현식과 같은 곳에 유용하다.

#   - match() 메소드는 문자열의 처음만 확인하므로, 예상치 못한 것에 매칭할 확률도 있다.
m = datepat.match('11/27/2012abcdef')
print(m)
print(m.group())

datepat = re.compile(r'(\d+)/(\d+)/(\d+)$')
print(datepat.match('11/27/2012abcdef'))
print(datepat.match('11/27/2012'))


#  2.5 텍스트 검색과 치환
#  ▣ 문제 : 문자열에서 텍스트 패턴을 검색하고 치환하고 싶다.
#  ▣ 해결 : 간단한 패턴이라면 str.replace() 메소드를 사용한다.
#           조금 더 복잡한 패턴을 사용하려면 re 모듈의 sub() 함수/메소드를 사용한다.
text = 'yeah, but no, but yeah, but no, but yeah'
print(text.replace('yeah', 'yep'))

text = 'Today is 11/27/2012. PyCon starts 3/13/2013.'
import re
print(re.sub(r'(\d+)/(\d+)/(\d+)', r'\3-\1-\2', text))

#   - 동일한 패턴을 사용한 치환을 계속해야 한다면 성능 향상을 위해 컴파일링을 고려해 보는 것이 좋다.
import re
datepat = re.compile(r'(\d+)/(\d+)/(\d+)')
print(datepat.sub(r'\3-\1-\2', text))

#   - 더 복잡한 치환을 위한 콜백 함수 명시.
from calendar import month_abbr
def change_date(m):
    mon_name = month_abbr[int(m.group(1))]
    return '{} {} {}'.format(m.group(2), mon_name, m.group(3))

print(datepat.sub(change_date, text))

#   - 치환이 몇 번 발생했는지 알고 싶다면 re.subn() 을 사용한다.
newtext, n = datepat.subn(r'\3-\1-\2', text)
print(newtext, n)

#  ▣ 토론 : 앞서 살펴본 sub() 메소드에 정규 표현식 검색과 치환 이외에 어려운 것은 없다.


#  2.6 대소문자를 구별하지 않는 검색과 치환
#  ▣ 문제 : 텍스트를 검색하고 치환할 때 대소문자를 구별하지 않고 싶다.
#  ▣ 해결 : 텍스트 관련 작업을 할 때 대소문자를 구별하지 않기 위해서는 re 모듈을 사용해야 하고 re.IGNORECASE 플래그를 지정해야 한다.
text = 'UPPER PYTHON, lower python, Mixed Python'
print(re.findall('python', text, flags=re.IGNORECASE))
print(re.sub('python', 'snake', text, flags=re.IGNORECASE))

def matchcase(word):
    def replace(m):
        text = m.group()
        if text.isupper():
            return word.upper()
        elif text.islower():
            return word.lower()
        elif text[0].isupper():
            return word.capitalize()
        else:
            return word
    return replace

print(re.sub('python', matchcase('snake'), text, flags=re.IGNORECASE))

#  ▣ 토론 : 대개의 경우 re.IGNORECASE 를 사용하는 것만으로 대소문자를 무시한 텍스트 작업에 무리가 없다.
#            하지만 유니코드가 포함된 작업을 하기에는 부족할 수도 있다.


#  2.7 가장 짧은 매칭을 위한 정규 표현식
#  ▣ 문제 : 정규 표현식을 사용한 텍스트 매칭을 하고 싶지만 텍스트에서 가장 긴 부분을 찾아낸다.
#            만약 가장 짧은 부분을 찾아내고 싶다면 어떻게 해야 할까?
#  ▣ 해결 : * 뒤에 ? 를 붙이면 된다.
import re
str_pat = re.compile(r'\"(.*)\"')
text1 = 'Computer says "no."'
print(str_pat.findall(text1))
text2 = 'Computer says "no." Phone says "yes."'
print(str_pat.findall(text2))
str_pat = re.compile(r'\"(.*?)\"')
print(str_pat.findall(text2))

#  ▣ 토론 : 패턴에서 점은 개행문을 제외한 모든 문제에 매칭하므로, ? 를 * 나 + 뒤에 붙여준다.


#  2.8 여러 줄에 걸친 정규 표현식 사용
#  ▣ 문제 : 여러 줄에 걸친 정규 표현식 매칭을 사용하고 싶다.
#  ▣ 해결 : 패턴에서 (?:.|\n)인 논캡처 그룹을 명시한다.
comment = re.compile(r'/\*(.*?)\*/')
text1 = '/* this is a comment */'
text2 = '''/* this is a
              multiline comment */
'''
print(comment.findall(text1))
print(comment.findall(text2))
comment = re.compile(r'/\*((?:.|\n)*?)\*/')
print(comment.findall(text2))

#  ▣ 토론 : re.compile() 함수에 re.DOTALL 이라는 유용한 플래그를 사용할 수 있다.
comment = re.compile(r'/\*(.*?)\*/', re.DOTALL)
print(comment.findall(text2))


#  2.9 유니코드 텍스트 노멀화
#  ▣ 문제 : 유니코드 모든 문자열에 동일한 표현식을 갖도록 보장해주자.
#  ▣ 해결 : unicodedata 모듈로 텍스트를 노멀화해서 표준 표현식으로 바꿔야 한다.
s1 = 'Spicy Jalape\u00f1o'
s2 = 'Spicy Jalapen\u0303o'
print(s1, s2)
print(s1 == s2, len(s1), len(s2))

#   - 위의 경우 같은 문자이지만 표현 방식이 달라 다른 문자열로 인식하였다.
#     따라서. 텍스트 노멀화를 통해 표준 표현식으로 변경해주어야 한다.
import unicodedata
t1 = unicodedata.normalize('NFC', s1)
t2 = unicodedata.normalize('NFC', s2)
print(t1 == t2)
print(ascii(t1))

t3 = unicodedata.normalize('NFD', s1)
t4 = unicodedata.normalize('NFD', s2)
print(t3 == t4)
print(ascii(t3))

#   - 파이썬은 특정 문자를 다룰 수 있도록 추가적인 호환성을 부여하는 NFKC 와 NFKD 노멀화도 지원한다.
s = '\ufb01'
print(s)
print(unicodedata.normalize('NFD', s))
print(unicodedata.normalize('NFKD', s))  # 개별 문자로 분리
print(unicodedata.normalize('NFKC', s))  # 개별 문자로 분리

#  ▣ 토론 : 일관적이고 안전한 유니코드 텍스트 작업을 위해서 노멀화는 아주 중요하다.
#            특히 인코딩을 조절할 수 없는 상황에서 사용자에게 문자열 입력을 받는 경우에는 특히 조심해야 한다.
t1 = unicodedata.normalize('NFD', s1)
print(''.join(c for c in t1 if not unicodedata.combining(c)))  # combining() 함수는 문자가 결합 문자인지 확인한다.


#  2.10 정규 표현식에 유니코드 사용
#  ▣ 문제 : 텍스트 프로세싱에 정규 표현식을 사용 중이다. 하지만 유니코드 문자 처리가 걱정된다.
#  ▣ 해결 : 기본적인 유니코드 처리를 위한 대부분의 기능을 re 모듈이 제공한다.
import re
num = re.compile('\d+')
print(num.match('123'))  # 아스키 숫자
print(num.match('\u0661\u0662\u0663'))  # 아라비아 숫자

#   - 특정 유니코드 문자를 패턴에 포함하고 싶으면, 유니코드 문자에 이스케이프 시퀀스를 사용한다.
arabic = re.compile('[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]+')

#   - 대소문자를 무시하는 매칭에 대소문자 변환을 합친 코드는 다음과 같다.
pat = re.compile('stra\u00dfe', re.IGNORECASE)
s = 'straße'
print(pat.match(s))
print(pat.match(s.upper()), s.upper())

#  ▣ 토론 : 유니코드와 정규 표현식을 같이 사용하려면 서드파티 regex 라이브러리를 설치하고 유니코드 대소문자 변환 등을 기본으로
#            제공하는 많은 기능을 이용하는 것이 좋다.


#  2.11 문자열에서 문자 잘라내기
#  ▣ 문제 : 텍스트의 처음, 끝, 중간에서 원하지 않는 공백문 등을 잘라내고 싶다.
#  ▣ 해결 : strip() 메소드를 사용하면 문자열의 처음과 끝에서 문자를 잘라낼 수 있다.
#            기본적으로 공백이나 \n 을 잘라내지만 원하는 문자를 지정할 수도 있다.
s = '     hello world \n'
print(s.strip())
print(s.lstrip())
print(s.rstrip())
t = '-----hello====='
print(t.strip('-'))
print(t.strip('='))
print(t.strip('-='))

#  ▣ 토론 : 데이터를 보기 좋게 만들기 위한 용도로 여러 strip() 메소드를 일반적으로 사용한다.
#            하지만 텍스트의 중간에서 잘라내기를 할 수는 없다.
s = ' hello        world    \n'
print(s.strip())

#   - 중간의 공백을 없애기 위해서는 replace() 메소드나 정규 표현식의 치환과 같은 다른 기술을 사용해야 한다.
print(s.replace(' ', ''))
import re
print(re.sub('\s+', ' ', s).strip())

with open('files\\somefile.txt') as f:  # 파일 전체 compile 해야 경로 인식.
    lines = (line.strip() for line in f)
    for line in lines:
        print(re.sub('\s+', ' ', line).strip())


#  2.12 텍스트 정리
#  ▣ 문제 : 당신의 웹 페이지에 어떤 사람이 장난스럽게 "python" 이라는 특수문자를 입력했다. 이를 정리하고 싶다.
#  ▣ 해결 : 특정 범위의 문자나 발음 구별 구호를 없애려고 할 때는 str.translate() 메소드를 사용해야 한다.
s = 'pýtĥöñ\fis\tawesome\r\n'
print(s)

#   - 우선 문자열에서 공백문을 잘라내기 위해 작은 변환 테이블을 만들어 놓고 translate() 를 사용한다.
remap = {
    ord('\t'): ' ',  # ord() : 하나의 문자열에 대해 유니코드를 나타내는 의미로 변환.
    ord('\f'): ' ',
    ord('\r'): None  # 삭제됨
}
a = s.translate(remap)
print(a)

#   - 결합 문자를 없애는 방법.
import unicodedata
import sys
cmb_chrs = dict.fromkeys(c for c in range(sys.maxunicode) if unicodedata.combining(chr(c)))  # combining() : 유니코드 중에 조합된 것을 추출.
print(cmb_chrs)
b = unicodedata.normalize('NFD', a)
print(b)
print(b.translate(cmb_chrs))

#   - 유니코드 숫자 문자를 이와 관련 있는 아스키 숫자에 매핑하도록 변환 테이블을 작성한다.
digitmap = {
    c: ord('0') + unicodedata.digit(chr(c)) for c in range(sys.maxunicode) if unicodedata.category(chr(c)) == 'Nd'
}
print(len(digitmap), digitmap)
x = '\u0661\u0662\u0663'
print(x.translate(digitmap))

#   - 또 다른 텍스트 정리 기술로 I/O 인코딩, 디코딩 함수가 있다. 이 방식은 텍스트를 우선 정리해 놓고 encode() 나 decode() 를
#     실행해서 잘라내거나 변경한다.
print(a)
b = unicodedata.normalize('NFD', a)
print(b.encode('ascii', 'ignore').decode('ascii'))  # ascii 형태로 인코딩 및 디코딩

#  ▣ 토론 : 텍스트 정리를 하다 보면 실행 성능 문제에 직면하기도 한다.
#            간단한 치환을 위해서는 str.replace() 함수를 사용하는 것이 빠르고 복잡한 치환을 하는 경우에는 str.translate() 함수를
#            사용하는 것이 좋다.


#  2.13 텍스트 정렬
#  ▣ 문제 : 텍스트를 특정 형식에 맞추어 정렬하고 싶다.
#  ▣ 해결 : ljust(), rjust(), center() 등이 있다.
text = 'Hello World'
print(text.ljust(20))
print(text.rjust(20))
print(text.center(20))

#   - 세 개의 메소드는 별도의 채워 넣기 문자를 사용할 수 있다.
print(text.rjust(20, '='))
print(text.center(20, '*'))

#   - format() 함수를 사용하면 인자로 <, >, ^ 를 적절하게 사용해 주면 된다.
print(format(text, '>20'))  # rjust 와 동일
print(format(text, '<20'))  # ljust 와 동일
print(format(text, '^20'))  # center 와 동일

#   - 공백 대신 특정 문자를 채워 넣고 싶다면 정렬 문자 앞에 그 문자를 지정한다.
print(format(text, '=>20'))
print(format(text, '*^20'))

#   - 포맷 코드는 format() 메소드에 사용해 여러 값을 서식화할 수도 있다.
print('{:>10} {:>10}'.format('Hello', 'World'))

#   - format() 을 사용하면 문자열뿐만 아니라 숫자 값 등 모든 값에 동작한다.
x = 1.2345
print(format(x, '>10'))  # str 타입으로 변환
print(format(x, '^10.2f'))  # 소수점 자리수 지정 가능

#  ▣ 토론 : 오래된 코드를 보면 % 연산자를 사용해 텍스트를 서식화하기도 했다.
print('%-20s ' % text)
print('%20s ' % text)


#  2.14 문자열 합치기
#  ▣ 문제 : 작은 문자열 여러 개를 합쳐 하나의 긴 문자열을 만들고 싶다.
#  ▣ 해결 : 합치고자 하는 문자열이 시퀀스나 순환 객체 안에 있다면 join() 메소드를 사용하는 것이 가장 빠르다.
parts = ['Is', 'Chicago', 'Not', 'Chicago?']
print(' '.join(parts))
print(','.join(parts))
print(''.join(parts))

#   - 합치려고 하는 문자열의 수가 아주 적다면 + 를 사용하면 된다.
a = 'Is Chicago'
b = 'Not Chicago?'
print(a + ' ' + b)

#   - + 연산자는 조금 더 복잡한 문자열 서식 연산에 사용해도 잘 동작한다.
print('{} {}'.format(a, b))
print(a + ' ' + b)

a = 'Hello' 'World'
print(a)

#  ▣ 토론 : 명심해야 할 부분은, + 연산자로 많은 문자열을 합치려고 하면 메모리 복사와 가비지 컬렉션으로 인해 매우 비효율적이라는 점이다.
s = ''
for p in parts:
    s += p

#   - 생성자 표현식으로 합치는 방법이 있다.
data = ['ACME', 50, 91.1]
print(' '.join(str(v) for v in data))

#   - 불필요한 문자열 합치기를 하고 있지 않은지도 주의하자.
a = 'qwer'
b = 'asdf'
c = 'zxcv'
print(a + ':' + b + ':' + c)  # 좋지 않은 방식
print(':'.join([a, b, c]))  # 개선된 방식
print(a, b, c, sep=':')  # 좋은 방식

#   - 수많은 짧은 문자열을 하나로 합쳐 문자열을 만드는 코드를 작성한다면, yeild 를 사용한 생성자 함수를 고려하자.
def sample():
    yield 'Is'
    yield 'Chicago'
    yield 'Not'
    yield 'Chicago?'

text = ''.join(sample())
print(text)

#   - 문자열을 입출력(I/O)으로 리다이렉트 할 수 있다.
for part in sample():
    f.write(part)

#   - 입출력을 조합한 하이브리드 방식 구현도 가능하다.
f = open('D:\\KYH\\02.PYTHON\\data\\combine.txt', 'a')

def combine(source, maxsize):
    parts = []
    size = 0
    for part in source:
        parts.append(part)
        size += len(part)
        if size >= maxsize:
            yield ''.join(parts)
            parts = []
            size = 0
    yield ''.join(parts)

for part in combine(sample(), 32768):
    f.write(part)


#  2.15 문자열에 변수 사용
#  ▣ 문제 : 문자열에 변수를 사용하고 이 변수에 맞는 값을 채우고 싶다.
#  ▣ 해결 : 파이썬 문자열에 변수 값을 치환하는 간단한 방법은 존재하지 않는다.
#            하지만 format() 메소드를 사용하면 비슷하게 흉내 낼 수 있다.
s = '{name} hash {n} messages.'
print(s.format(name='Guido', n=37))

#   - 치환할 값이 변수에 들어 있다면 format_map() 과 vars() 를 함께 사용하면 된다.
name = 'Guido'
n = 37
print(s.format_map(vars()))

#   - vars() 에는 인스턴스를 사용할 수도 있다.
class Info:
    def __init__(self, name, n):
        self.name = name
        self.n = n

a = Info('Guido', 37)
print(s.format_map(vars(a)))

#   - format() 또는 format_map() 사용시 빠진 값은 __missing__() 메소드가 있는 딕셔너리 클래스를 정의해서 피할 수 있다.
class safesub(dict):
    def __missing__(self, key):
        return '{' + key + '}'

del n
print(s.format_map(safesub(vars())))

#   - 코드에서 변수 치환을 빈번히 사용할 것 같다면 치환하는 작업을 유틸리티 함수에 모아 놓고 소위 "프레임 핵(frame hack)"으로 사용할 수 있다.
import sys

def sub(text):
    return text.format_map(safesub(sys._getframe(1).f_locals))

name = 'Guido'
n = 37
print(sub('Hello {name}'))
print(sub('You have {n} messages.'))
print(sub('Your favorite color is {color}'))

#  ▣ 토론 : 파이썬 자체에서 변수 보간법이 존재하지 않아서 다양한 대안이 생겼다.
name = 'Guido'
n = 37
print('%(name) has %(n) messages.' % vars())

import string
s = string.Template('$name has $n messages.')
print(s.substitute(vars()))


#  2.16 텍스트 열의 개수 고정
#  ▣ 문제 : 긴 문자열의 서식을 바꿔 열의 개수를 조절하고 싶다.
#  ▣ 해결 : textwrap 모듈을 사용해서 텍스트를 재서식화 한다.
s = 'Look into my eyes, look into my eyes, the eyes, the eyes,' \
    "the eyes, not around the eyes, don't look around the eyes," \
    "look into my eyes, you're under."
import textwrap
print(textwrap.fill(s, 70))
print(textwrap.fill(s, 40))
print(textwrap.fill(s, 40, initial_indent='       '))
print(textwrap.fill(s, 40, subsequent_indent='       '))

#  ▣ 토론 : 텍스트를 출력하기 전에 textwrap 모듈을 사용하면 깔끔하게 서식을 맞출 수 있다.
#           특히 터미널에 사용할 텍스트에 적합하다.
#           터미널의 크기를 얻으려면 os.get_terminal_size() 를 사용한다.
import os
print(os.get_terminal_size().columns)


#  2.17 HTML 과 XML 엔티티 처리
#  ▣ 문제 : &entity; 나 &#code; 와 같은 HTML, XML 엔터티를 이에 일치하는 문자로 치환하고 싶다.
#           혹은 텍스트를 생성할 때 특정 문자(<, >, & 등)를 피하고 싶다.
#  ▣ 해결 : 텍스트를 생성할 때 <나>와 같은 특수 문자를 치환하는 것은 html.escape() 함수를 사용하면 상대적으로 간단히 처리할 수 있다.
s = 'Elements are written as "<tag>text</tag>".'
import html
print(s)
print(html.escape(s))

#   - 따옴표는 남겨 두도록 지정
print(html.escape(s, quote=False))

#   - 텍스트를 아스키로 만들고 캐릭터 코드를 아스키가 아닌 문자에 끼워 넣고 싶으면 errors='xmlcharrefreplace' 인자를 입출력 관련 함수에 사용한다.
s = 'Spicy Jalapeño'
print(s.encode('ascii', errors='xmlcharrefreplace'))

#   - 수동으로 치환을 해야 한다면 HTML, XML 파서에 내장되어 있는 여러 유틸리티 함수나 메소드를 사용한다.
s = 'Spicy &quot;Jalape&#241;o&quot.'
from html.parser import HTMLParser
p = HTMLParser()
print(p.unescape(s))  # 파이썬 3.5 버전부터 deprecated 됨.
print(html.unescape(s))

t = 'The prompt is &gt;&gt;&gt;'
from xml.sax.saxutils import unescape
print(unescape(t))

#  ▣ 토론 : HTML, XML 을 생성할 때 특수 문자를 제대로 이스케이핑하는 과정을 간과하기 쉽다.
#            print() 로 결과물을 생성하거나 기본적인 문자열 서식 기능을 사용할 때 특히 더 그렇다.
#            가장 쉬운 해결책은 html.escape() 와 같은 유틸리티 함수를 사용하는 것이다.


#  2.18 텍스트 토큰화
#  ▣ 문제 : 문자열을 파싱해서 토큰화하고 싶다.
#  ▣ 해결 : 정규 표현식과 scanner() 메소드를 사용한다.

#   - 정규 표현식을 사용.(이름 있는 캡처 그룹)
text = 'foo = 23 + 42 * 10'

import re
NAME = r'(?P<NAME>[a-zA-Z_][a-zA-Z_0-9]*)'
NUM = r'(?P<NUM>\d+)'
PLUS = r'(?P<PLUS>\+)'
TIMES = r'(?P<TIMES>\*)'
EQ = r'(?P<EQ>=)'
WS = r'(?P<WS>\s+)'

master_pat = re.compile('|'.join([NAME, NUM, PLUS, TIMES, EQ, WS]))

#   - scanner() 메소드를 사용.
scanner = master_pat.scanner('foo = 42')
scanner.match()
print(_.lastgroup, _.group())  # _ 는 파이썬 2.x 버전에서 사용 가능함.
scanner.match()
print(_.lastgroup, _.group())
scanner.match()
print(_.lastgroup, _.group())
scanner.match()
print(_.lastgroup, _.group())
scanner.match()
print(_.lastgroup, _.group())

#   - 이 기술을 사용해 간결한 생성자를 만들 수 있다.
from collections import namedtuple
Token = namedtuple('Token', ['type', 'value'])

def generate_tokens(pat, text):
    scanner = pat.scanner(text)
    for m in iter(scanner.match, None):
        yield Token(m.lastgroup, m.group())

for tok in generate_tokens(master_pat, 'foo = 42'):
    print(tok)

tokens = (tok for tok in generate_tokens(master_pat, text) if tok.type != 'WS')
for tok in tokens:
    print(tok)

#  ▣ 토론 : 보통 더 복잡한 텍스트 파싱이나 처리를 하기 전에 토큰화를 한다.
#            매칭할 때 re 는 명시한 순서대로 패턴을 매칭한다. 따라서 한 패턴이 다른 패턴의 부분이 되는 경우가 있다면
#            항상 더 긴 패턴을 먼저 넣어야 한다.
LT = r'(?P<LT><)'
LE = r'(?P<LE><=)'
EQ = r'(?P<EQ>=)'

master_pat = re.compile('|'.join([LE, LT, EQ]))  # 올바름
master_pat = re.compile('|'.join([LT, LE, EQ]))  # 틀림

for tok in generate_tokens(master_pat, '<='):
    print(tok)

#   - 패턴이 부분 문자열을 형성하는 경우도 조심해야 한다.
PRINT = r'(?P<PRINT>print)'
NAME = r'(?P<NAME>[a-zA-Z_][a-zA-Z_0-9]*)'

master_pat = re.compile('|'.join([PRINT, NAME]))

for tok in generate_tokens(master_pat, 'printer'):
    print(tok)


#  2.19 간단한 재귀 파서 작성
#  ▣ 문제 : 주어진 문법 규칙에 따라 텍스트를 파싱하고 동작을 수행하거나 입력된 텍스트를 추상 신택스 트리로 나타내야 한다.
#            문법은 간단하지만 프레임워크를 사용하지 않고 파서를 직접 작성하고 싶다.
#  ▣ 해결 : 이 문제는 특정 문법에 따라 텍스트를 파싱하는 데 집중한다.
#            우선 문법의 정규 스펙을 BNF 나 EBNF 로 하는 데서 시작한다.
import re
import collections

#   - 토큰 스펙화.
NUM    = r'(?P<NUM>\d+)'
PLUS   = r'(?P<PLUS>\+)'
MINUS  = r'(?P<MINUS>-)'
TIMES  = r'(?P<TIMES>\*)'
DIVIDE = r'(?P<DIVIDE>/)'
LPAREN = r'(?P<LPAREN>\()'
RPAREN = r'(?P<RPAREN>\))'
WS     = r'(?P<WS>\s+)'

master_pat = re.compile('|'.join([NUM, PLUS, MINUS, TIMES, DIVIDE, LPAREN, RPAREN, WS]))

#   - 토큰화.
Token = collections.namedtuple('Token', ['type', 'value'])

def generate_tokens(text):
    scanner = master_pat.scanner(text)
    for m in iter(scanner.match, None):
        tok = Token(m.lastgroup, m.group())
        if tok.type != 'WS':
            yield tok

#   - 파서.
class ExpressionEvaluator:
    '''
        재귀 파서 구현, 모든 메소드는 하나의 문법 규칙을 구현한다.
        현재 룩어헤드 토큰을 받고 테스트하는 용도로 ._accept()를 사용한다.
        입력 받은 내역에 완벽히 매칭하고 다음 토큰을 무시할 때는
        ._expect()를 사용한다. (혹시 매칭하지 않는 경우에는 SyntaxError 를 발생한다.)
    '''

    def parse(self, text):
        self.tokens = generate_tokens(text)
        self.tok = None        # 마지막 심볼 소비
        self.nexttok = None    # 다음 심볼 토큰화
        self._advance()         # 처음 룩어헤드 토큰 불러오기
        return self.expr()

    # generate_tokens() 메서드에서 가져온 토큰을 순차적으로 설정하는 함수
    def _advance(self):
        'Advance one token ahead'
        self.tok, self.nexttok = self.nexttok, next(self.tokens, None)  # next 함수는 iterator 를 순차적으로 리턴시켜주는 함수

    # 다음 토큰이 원하는 토큰인 경우 self.tok 에 다음 토큰을 담고 True 를 리턴하는 함수
    def _accept(self, toktype):
        'Test and consume the next token if it matches toktype'
        if self.nexttok and self.nexttok.type == toktype:
            self._advance()
            return True
        else:
            return False

    def _expect(self, toktype):
        'Consume next token if it matches toktype or raise SyntaxError'
        if not self._accept(toktype):
            raise SyntaxError('Expected ' + toktype)

    #   - 문법 규칙.
    def expr(self):
        "expression ::= term { ('+'|'-') term }*"

        exprval = self.term()
        while self._accept('PLUS') or self._accept('MINUS'):
            op = self.tok.type
            right = self.term()
            if op == 'PLUS':
                exprval += right
            elif op == 'MINUS':
                exprval -= right
        return exprval

    def term(self):
        "term ::= factor { ('*'|'/') factor }*"

        termval = self.factor()
        while self._accept('TIMES') or self._accept('DIVIDE'):
            op = self.tok.type
            right = self.factor()
            if op == 'TIMES':
                termval *= right
            elif op == 'DIVIDE':
                termval /= right
        return termval

    def factor(self):
        "factor ::= NUM | ( expr )"

        if self._accept('NUM'):
            return int(self.tok.value)
        elif self._accept('LPAREN'):
            exprval = self.expr()
            self._expect('RPAREN')
            return exprval
        else:
            raise SyntaxError('Expected NUMBER or LPAREN')

e = ExpressionEvaluator()
print(e.parse('2'))
print(e.parse('2 + 3'))
print(e.parse('2 + 3 * 4'))
print(e.parse('2 + (3 + 4) * 5'))
# print(e.parse('2 + (3 + * 4)'))


class ExpressionTreeBuilder(ExpressionEvaluator):
    def expr(self):
        "expression ::= term { ('+'|'-') term }"

        exprval = self.term()
        while self._accept('PLUS') or self._accept('MINUS'):
            op = self.tok.type
            right = self.term()
            if op == 'PLUS':
                exprval = ('+', exprval, right)
            elif op == 'MINUS':
                exprval = ('-', exprval, right)
        return exprval

    def term(self):
        "term ::= factor { ('*'|'/') factor }"

        termval = self.factor()
        while self._accept('TIMES') or self._accept('DIVIDE'):
            op = self.tok.type
            right = self.factor()
            if op == 'TIMES':
                termval = ('*', termval, right)
            elif op == 'DIVIDE':
                termval = ('/', termval, right)
        return termval

    def factor(self):
        'factor ::= NUM | ( expr )'

        if self._accept('NUM'):
            return int(self.tok.value)
        elif self._accept('LPAREN'):
            exprval = self.expr()
            self._expect('RPAREN')
            return exprval
        else:
            raise SyntaxError('Expected NUMBER or LPAREN')

e = ExpressionTreeBuilder()
print(e.parse('2 + 3'))
print(e.parse('2 + 3 * 4'))
print(e.parse('2 + (3 + 4) * 5'))
print(e.parse('2 + 3 + 4'))

#  ▣ 토론 : 파싱은 컴파일러 과목에서 3주 이상을 할애해서 배우는 쉽지 않은 주제이다.
#            파싱 알고리즘이나 문법과 같은 기본적인 지식을 좀 더 알고 싶다면 우선 컴파일러 책을 한 권 읽어야 한다.
#            재귀 파서의 한 가지 제약으로 좌측 재귀가 포함된 어떠한 문법 규칙에 대해서도 사용할 수 없다.

#   - 정말 복잡한 문법이 있다면 PyParsing 이나 PLY 와 같은 파싱 도구를 사용하는 것이 더 좋다.
from ply.lex import lex
from ply.yacc import yacc

# 토큰 리스트
tokens = ['NUM', 'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'LPAREN', 'RPAREN']

# 무시 문자
t_ignore = '\t\n'

# 토큰 스펙 (정규 표현식으로)
t_PLUS = r'\+'
t_MINUS = r'-'
t_TIMES = r'\*'
t_DIVIDE = r'/'
t_LPAREN = r'\('
t_RPAREN = r'\)'

# 토큰화 함수
def t_NUM(t):
    r'\d+'
    t.value = int(t.value)
    return t

# 에러 핸들러
def t_error(t):
    print('Bad character: {!r}'.format(t.value[0]))
    t.skip(1)

# 렉서(lexer) 만들기
lexer = lex()

# 문법 규칙과 핸들러 함수
def p_expr(p):
    '''
        expr : expr PLUS term
             | expr MINUS term
    '''
    if p[2] == '+':
        p[0] = p[1] + p[3]
    elif p[2] == '-':
        p[0] = p[1] - p[3]

def p_expr_term(p):
    '''
        expr : term
    '''
    p[0] = p[1]

def p_term(p):
    '''
        term : term TIMES factor
             | term DIVIDE factor
    '''
    if p[2] == '*':
        p[0] = p[1] * p[3]
    elif p[2] == '/':
        p[0] = p[1] / p[3]

def p_term_factor(p):
    '''
        term : factor
    '''
    p[0] = p[1]

def p_factor(p):
    '''
        factor : NUM
    '''
    p[0] = p[1]

def p_factor_group(p):
    '''
        factor : LPAREN expr RPAREN
    '''
    p[0] = p[2]

def p_error(p):
    print('Syntax error')

parser = yacc()
print(parser.parse('2'))


#  2.20 바이트 문자열에 텍스트 연산 수행
#  ▣ 문제 : 바이트 문자열(byte string)에 일반적인 텍스트 연산(잘라내기, 검색, 치환 등)을 수행하고 싶다.
#  ▣ 해결 : 바이트 문자열도 텍스트 문자열과 마찬가지로 대부분의 연산을 내장하고 있다.
data = b'Hello World'
print(data[0:5])
print(data.startswith(b'Hello'))
print(data.split())
print(data.replace(b'Hello', b'Hello Cruel'))

#   - 바이트 배열에도 사용 가능하다.
data = bytearray(b'Hello World')
print(data[0:5])
print(data.startswith(b'Hello'))
print(data.split())
print(data.replace(b'Hello', b'Hello Cruel'))

#   - 바이트 배열에서도 정규 표현식이 가능하다.
data = b'FOO:BAR,SPAM'
import re
print(re.split(b'[:,]', data))  # 패턴도 바이트로 나타내야 한다.

#  ▣ 토론 : 대개의 경우 텍스트 문자열에 있는 연산 기능은 바이트 문자열에도 내장되어 있다.
#            하지만 주의해야 할 차이점이 몇 가지 있다.

#   - 첫째. 바이트 문자열에 인덱스를 사용하면 개별 문자가 아니라 정수를 가리킨다.
a = 'Hello World'
print(a[0], a[1])

b = b'Hello World'
print(b[0], b[1])

#   - 둘째. 바이트 문자열은 보기 좋은 표현식을 지원하지 않으며, 텍스트 문자열로 변환하지 않으면 깔끔하게 출력할 수도 없다.
s = b'Hello World'
print(s)
print(s.decode('ascii'))

#   - 셋째. 바이트 문자열은 서식화를 지원하지 않는다.
# print(b'%10s %10d %10.2f' %(b'ACME', 100, 490.1))
# print(b'{} {} {}'.format(b'ACME', 100, 490.1))
print('{:10s} {:10d} {:10.2f}'.format('ACME', 100, 490.1).encode('ascii'))

#   - 넷째. 바이트 문자열을 사용하면 특정 연산의 문법에 영향을 주기도 한다.
with open('files/somefile.txt', 'w') as f:
    f.write('spicy')

import os
print(os.listdir('.'))
print(os.listdir(b'.'))
#   ※ 바이트 데이터가 성능상 더 빠르더라도, 코드가 매우 지저분하고 이해하기 어려워지므로 텍스트 데이터를 사용하는 것이 좋다.