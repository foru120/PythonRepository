# Chapter 6. 데이터 인코딩과 프로세싱
#  6.1 CSV 데이터 읽고 쓰기
#  ▣ 문제 : CSV 파일로 인코딩된 데이터를 읽거나 쓰고 싶다.
#  ▣ 해결 : 대부분의 CSV 데이터는 csv 라이브러리를 사용한다.
import csv
with open('PythonCookBook/files/emp.csv') as f:
    f_csv = csv.reader(f)
    headers = next(f_csv)
    print(headers)
    for row in f_csv:
        print(row[0], row[1])

#   - 인덱스 사용이 때때로 헷갈리기 때문에 네임드 튜플을 고려한다.
#     이렇게 하면 row.empno, row.ename 과 같이 열 헤더를 사용할 수 있다.
from collections import namedtuple
with open('PythonCookBook/files/emp.csv') as f:
    f_csv = csv.reader(f)
    headings = next(f_csv)
    Row = namedtuple('Row', headings)
    for r in f_csv:
        row = Row(*r)
        print(row.empno, row.ename)

#   - 또 다른 대안으로 데이터를 딕셔너리 시퀀스로 읽을 수도 있다.
import csv
with open('PythonCookBook/files/emp.csv') as f:
    f_csv = csv.DictReader(f)
    for row in f_csv:
        print(row['empno'], row['ename'])

#   - CSV 데이터를 쓰려면, csv 모듈을 사용해서 쓰기 객체를 생성한다.
headers = ['Symbol', 'Price', 'Date', 'Time', 'Change', 'Volume']
rows = [('AA', 39.48, '6/11/2007', '9:36am', -0.18, 181800),
        ('AIG', 71.38, '6/11/2007', '9:36am', -0.15, 195500),
        ('AXP', 62.58, '6/11/2007', '9:36am', -0.46, 935000)]
with open('PythonCookBook/files/stocks.csv', 'w') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(rows)

#   - 데이터를 딕셔너리 시퀀스로 가지고 있는 경우
headers = ['Symbol', 'Price', 'Date', 'Time', 'Change', 'Volume']
rows = [{'Symbol': 'AA', 'Price': 39.48, 'Date': '6/11/2007', 'Time': '9:36am', 'Change': -0.18, 'Volumn': 181800},
        {'Symbol': 'AIG', 'Price': 71.38, 'Date': '6/11/2007', 'Time': '9:36am', 'Change': -0.15, 'Volumn': 195500},
        {'Symbol': 'AXP', 'Price': 62.58, 'Date': '6/11/2007', 'Time': '9:36am', 'Change': -0.46, 'Volumn': 935000}]
with open('PythonCookBook/files/stocks.csv', 'w') as f:
    f_csv = csv.DictWriter(f, headers)
    f_csv.writeheader()
    f_csv.writerows(rows)

#  ▣ 토론 : CSV 데이터를 수동으로 다루는 프로그램을 작성하기보다는 csv 모듈을 사용하는 것이 훨씬 나은 선택이다.
#   - 구분자가 tab 으로 나누어진 데이터를 읽는 경우
with open('PythonCookBook/files/stocks.csv') as f:
    f_tsv = csv.reader(f, delimiter='\t')
    for row in f_tsv:
        print(row)

#   - CSV 파일 헤더에 유효하지 않은 식별 문자가 들어있는 경우
import re
with open('PythonCookBook/files/stocks.csv') as f:
    f_csv = csv.reader(f)
    headers = [re.sub('[^a-zA-Z_]', '_', h) for h in next(f_csv)]
    Row = namedtuple('Row', headers)
    for r in f_csv:
        row = Row(*r)

#   - CSV 데이터에 대해서 추가적인 형식 변환을 하는 경우
col_types = [str, float, str, str, float, int]
with open('PythonCookBook/files/stocks.csv') as f:
    f_csv = csv.reader(f)
    headers = next(f_csv)
    for row in f_csv:
        r = tuple(convert(value) for convert, value in zip(col_types, row))
        print(r)

#   - 딕셔너리에서 선택한 필드만 변환하는 경우
print('Reading as dicts with type conversion')
field_types = [('Price', float), ('Change', float), ('Volume', int)]
with open('PythonCookBook/files/stocks.csv') as f:
    for row in csv.DictReader(f):
        row.update((key, conversion(row[key])) for key, conversion in field_types)
        print(row)


#  6.2 JSON 데이터 읽고 쓰기
#  ▣ 문제 : JSON 으로 인코딩된 데이터를 읽거나 쓰고 싶다.
#  ▣ 해결 : JSON 으로 데이터를 인코딩, 디코딩하는 쉬운 방법은 json 모듈을 사용하는 것이다.
#            주요 함수는 json.dumps() 와 json.loads() 이고, pickle 과 같은 직렬화 라이브러리에서 사용한 것과 인터페이스는 동일하다.
import json

data = {'name': 'ACME', 'shares': 100, 'price': 542.23}
json_str = json.dumps(data)
data = json.loads(json_str)
print(data, type(data))

#   - 문자열이 아닌 파일로 작업한다면 json.dump() 와 json.load() 를 사용해서 JSON 데이터를 인코딩/디코딩한다.
with open('PythonCookBook/files/data.json', 'w') as f:
    json.dump(data, f)

with open('PythonCookBook/files/data.json', 'r') as f:
    data = json.load(f)
    print(data)

#  ▣ 토론 : JSON 인코딩은 None, bool, int, float, str 과 같은 기본 타입과 함께 리스트, 튜플, 딕셔너리와 같은 컨테이너 타입을 지원한다.
#            딕셔너리의 경우 키는 문자열로 가정한다.
#            JSON 인코딩 포맷은 약간의 차이점을 제외하고는 파이썬 문법과 거의 동일하다.
#            예를 들어 True 는 true 로 False 는 false 로 None 은 null 로 매핑된다.
print(json.dumps(False))
d = {'a': True, 'b': 'Hello', 'c': None}
print(json.dumps(d))

#   - 데이터에 중첩이 심하게 된 구조체가 포함된 경우 pprint 모듈의 pprint() 함수를 사용해 보자.
#     이 함수는 키를 알파벳 순으로 나열하고 딕셔너리를 좀 더 보기 좋게 출력한다.
from urllib.request import urlopen
import json
u = urlopen('http://search.twitter.com/search.json?q=python&rpp=5')
resp = json.loads(u.read().decode('utf-8'))
from pprint import pprint
pprint(resp)

#   - 일반적으로 JSON 디코딩은 제공 받은 데이터로부터 딕셔너리나 리스트를 생성한다.
#     다른 종류의 객체를 만들고 싶다면 json.loads() 에 object_pairs_hook 나 object_hook 를 넣는다.
s = '{"name": "ACME", "shares": 50, "price": 490.1}'
from collections import OrderedDict
data = json.loads(s, object_pairs_hook=OrderedDict)
print(data, data['name'])

#   - JSON 딕셔너리를 파이썬 객체로 바꾸는 예시
class JSONObject:
    def __init__(self, d):
        self.__dict__ = d

data = json.loads(s, object_hook=JSONObject)
print(data.name, data.shares, data.price)

#   - 출력을 더 보기 좋게 하기위해 json.dumps() 에 indent 인자를 사용한다.
print(json.dumps(data))
print(json.dumps(data, indent=4))  # indent : 들여쓰는 역할

#   - 출력에서 키를 정렬하는 경우
print(json.dumps(data, sort_keys=True))

#   - 인스턴스는 일반적으로 JSON 으로 직렬화하지 않는다.
#     직렬화하고 싶다면 인스턴스를 입력으로 받아 직렬화 가능한 딕셔너리를 반환하는 함수를 제공해야 한다.
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def serialize_instance(obj):
    d = {'__classname__': type(obj).__name__}
    d.update(vars(obj))  # vars : 해당 객체에 대한 변수 정보 출력 -> dict
    return d

classes = {'Point': Point}

def unserialize_object(d):
    clsname = d.pop('__classname__', None)
    if clsname:
        cls = classes[clsname]
        obj = cls.__new__(cls)
        for key, value in d.items():
            setattr(obj, key, value)
        return obj
    else:
        return d

p = Point(2, 3)
s = json.dumps(p, default=serialize_instance)
print(s)
a = json.loads(s, object_hook=unserialize_object)
print(a, a.x, a.y)


#  6.3 단순한 XML 데이터 파싱
#  ▣ 문제 : 단순한 XML 문서에서 데이터를 얻고 싶다.
#  ▣ 해결 : 단순한 XML 문서에서 데이터를 얻기 위해 xml.etree.ElementTree 모듈을 사용하면 된다.
from urllib.request import urlopen
from xml.etree.ElementTree import parse

u = urlopen('http://planet.python.org/rss20.xml')  # RSS 피드를 다운로드하고 파싱한다.
doc = parse(u)

for item in doc.iterfind('channel/item'):
    title = item.findtext('title')
    date = item.findtext('pubDate')
    link = item.findtext('link')

    print(title)
    print(date)
    print(link)
    print()

#  ▣ 토론 : 많은 애플리케이션에서 XML 로 인코딩된 데이터를 다룬다.
#            인터넷 상에서 데이터를 주고 받을 때 XML 을 사용하는 곳이 많기도 하지만, 애플리케이션 데이터를 저장할 때도
#            일반적으로 사용하는 형식이다.

#   - ElementTree 모듈이 나타내는 모든 요소는 파싱에 유용한 요소와 메소드를 약간 가지고 있다.
#     tag 요소에는 태그의 이름, text 요소에는 담겨 있는 텍스트가 포함되어 있고 필요한 경우 get() 메소드로 요소를 얻을 수 있다.
print(doc)
e = doc.find('channel/title')
print(e)
print(e.tag, e.text, type(e))
e1 = doc.find('channel')
print(e1.get('name'))  # get() :  태그 속성을 가져옴

#   - XML 파싱에 xml.etree.ElementTree 말고 다른 것을 사용할 수도 있다.
#     임포트 구문만 from lxml.etree import parse 로 바꾸면 된다.
#     lxml 은 XML 표준과 완벽히 동일한 혜택을 제공한다. 또한 매우 빠르고 검증, XSLT, XPath 와 같은 모든 기능을 제공한다.


#  6.4 매우 큰 XML 파일 증분 파싱하기
#  ▣ 문제 : 매우 큰 XML 파일에서 최소의 메모리만 사용하여 데이터를 추출하고 싶다.
#  ▣ 해결 : 증분 데이터 처리에 직면할 때면 언제나 이터레이터와 제너레이터를 떠올려야 한다.
#            여기 아주 큰 XML 파일을 증분적으로 처리하며 메모리 사용은 최소로 하는 함수를 보자.
from xml.etree.ElementTree import iterparse

def parse_and_remove(filename, path):
    path_parts = path.split('/')
    doc = iterparse(filename, ('start', 'end'))
    next(doc)

    tag_stack = []
    elem_stack = []

    for event, elem in doc:
        if event == 'start':
            tag_stack.append(elem.tag)
            elem_stack.append(elem)
        elif event == 'end':
            if tag_stack == path_parts:
                yield elem
                elem_stack[-2].remove(elem)  # 앞에서 나온 요소를 부모로부터 제거하는 역할 (Tag 삭제)
            try:
                tag_stack.pop()
                elem_stack.pop()
            except IndexError:
                pass

#   - 파일 전체를 읽어 메모리에 넣고 수행하는 코드
from xml.etree.ElementTree import iterparse
from collections import Counter
potholes_by_zip = Counter()
doc = parse('PythonCookBook/files/potholes.xml')
for pothole in doc.iterfind('row/row'):
    potholes_by_zip[pothole.findtext('zip')] += 1

for zipcode, num in potholes_by_zip.most_common():
    print(zipcode, num)

#   - 파일의 특정 부분을 가지고 메모리에 넣고 수행하는 코드
from collections import Counter
potholes_by_zip = Counter()

data = parse_and_remove('PythonCookBook/files/potholes.xml', 'row/row')
for pothole in data:
    potholes_by_zip[pothole.findtext('zip')] += 1

for zipcode, num in potholes_by_zip.most_common():
    print(zipcode, num)

#  ▣ 토론 : iterparse() 메소드로, XML 문서를 증분 파싱
#             - iterparse() 가 생성한 이터레이터는 (event, elem) 으로 구성된 튜플을 만든다.
#             - start 이벤트는 요소가 처음 생성되었지만 다른 데이터를 만들지 않았을 때 생성된다.
#             - end 이벤트는 요소를 마쳤을 때 생성된다.
#             - start-ns 와 end-ns 이벤트는 XML 네임스페이스 선언을 처리하기 위해 사용한다.
#             - elem_stack[-2].remove(elem) 을 통해 부모 태그로부터 특정 태그(elem)를 제거한다.


#  6.5 딕셔너리를 XML 로 바꾸기
#  ▣ 문제 : 파이썬 딕셔너리 데이터를 받아서 XML 로 바꾸고 싶다.
#  ▣ 해결 : xml.etree.ElementTree 라이브러리는 파싱에 일반적으로 사용하지만, XML 문서를 생성할 때 사용하기도 한다.
from xml.etree.ElementTree import Element

def dict_to_xml(tag, d):
    elem = Element(tag)

    for key, val in d.items():
        child = Element(key)
        child.text = str(val)
        elem.append(child)
    return elem

s = {'name': 'GOOG', 'shares':100, 'price': 490.1}
e = dict_to_xml('stock', s)
print(e)

#   - I/O 를 위해서 xml.etree.ElementTree 의 tostring() 함수로 이를 바이트 문자열로 변환
from xml.etree.ElementTree import tostring
print(tostring(e))

#   - 요소에 속성을 넣고 싶으면 set() 메소드를 사용한다.
e.set('_id', '1234')
print(tostring(e))

#  ▣ 토론 : XML 을 생성할 때 단순히 문자열을 사용하고 싶을 수도 있다.
def dict_to_xml_str(tag, d):
    parts = ['<{}>'.format(tag)]
    for key, val in d.items():
        parts.append('<{0}>{1}</{0}>'.format(key, val))
    parts.append('</{}>'.format(tag))
    return ''.join(parts)

#   - 딕셔너리에 다음과 같이 특별 문자가 포함되어 있는 경우
d = {'name': '<spam>'}
print(dict_to_xml_str('item', d))
print(tostring(dict_to_xml('item', d)))

#   - 마지막 예제에서 < 와 > 문자가 &lt; 와 &gt; 로 치환되었다.
#     이런 문자를 수동으로 이스케이핑하고 싶다면 xml.sax.saxutils 의 escape() 와 unescape() 함수를 사용한다.
from xml.sax.saxutils import escape, unescape
print(escape('<spam>'))
print(unescape('<spam>'))


#  6.6 XML 파싱, 수정, 저장
#  ▣ 문제 : XML 문서를 읽고, 수정하고, 수정 내용을 XML 에 반영하고 싶다.
#  ▣ 해결 : xml.etree.ElementTree 모듈로 이 문제를 간단히 해결할 수 있다.
#            우선 일반적인 방식으로 문서 파싱부터 시작한다.
from xml.etree.ElementTree import parse, Element
doc = parse('PythonCookBook/files/pred.xml')
root = doc.getroot()
print(root)

#   - 요소 몇 개 제거하기
root.remove(root.find('sri'))
root.remove(root.find('cr'))

#   - 특정 태그 뒤에 요소 몇개 삽입하기
print(root.getchildren().index(root.find('nm')))
e = Element('spam')
e.text = 'This is a test'
root.insert(2, e)

#   - 파일에 쓰기
doc.write('PythonCookBook/files/newpred.xml', xml_declaration=True)

#  ▣ 토론 : XML 문서의 구조를 수정하는 것은 어렵지 않지만 모든 수정 사항은 부모 요소에도 영향을 미쳐 리스트인 것처럼 다루어진다는 점을 기억해야 한다.
#            그리고 모든 요소는 element[i] 또는 element[i:j] 와 같이 인덱스와 슬라이스 명령으로도 접근할 수 있다.


#  6.7 네임스페이스로 XML 문서 파싱
#  ▣ 문제 : XML 문서를 파싱할 때 XML 네임스페이스를 사용하고 싶다.
#  ▣ 해결 : 다음과 같이 네임스페이스를 사용하는 문서를 고려해 보자.
from xml.etree.ElementTree import Element, parse
doc = parse('PythonCookBook/files/sample.xml')
root = doc.getroot()

#   - 동작하는 쿼리
print(doc.findtext('author'))
print(doc.find('content'))

#   - 네임스페이스 관련 쿼리(동작하지 않음)
print(doc.find('content/html'))

#   - 조건에 맞는 경우에만 동작
print(doc.find('content/{http://www.w3.org/1999/xhtml}html'))

#   - 동작하지 않음
print(doc.findtext('content/{http://www.w3.org/1999/xhtml}html/head/title'))

#   - 조건에 일치함
print(doc.findtext('content/{http://www.w3.org/1999/xhtml}html/{http://www.w3.org/1999/xhtml}head/{http://www.w3.org/1999/xhtml}title'))

#   - 유틸리티 클래스로 네임스페이스를 감싸 주면 문제를 더 단순화할 수 있다.
class XMLNamespaces:
    def __init__(self, **kwargs):
        self.namespaces = {}
        for name, uri in kwargs.items():
            self.register(name, uri)

    def register(self, name, uri):
        self.namespaces[name] = '{'+uri+'}'

    def __call__(self, path):  # __call__() : 클래스 인스턴스명 자체로 함수처럼 사용할 때 호출된다. 해당 메서드를 정의한 클래스의 인스ㅓㄴ스는 callable(인스턴스) 에 True 를 리턴한다.
        return path.format_map(self.namespaces)  # format_map() : 키 값에 따른 포맷팅 형식을 맞춰준다.

ns = XMLNamespaces(html='http://www.w3.org/1999/xhtml')
print(doc.find(ns('content/{html}html')))
print(doc.findtext(ns('content/{html}html/{html}head/{html}title')))

#  ▣ 토론 : 네임스페이스를 포함한 XML 문서를 파싱하기는 꽤나 복잡하다.
#            XMLNamespaces 클래스는 짧게 줄인 네임스페이스 이름을 쓸 수 있도록 해서 코드를 정리해 줄 뿐이다.
#            하지만 iterparse() 함수를 사용한다면 네임스페이스 처리의 범위에 대해서 정보를 조금 더 얻을 수는 있다.
from xml.etree.ElementTree import iterparse
for evt, elem in iterparse('PythonCookBook/files/sample.xml', ('end', 'start-ns', 'end-ns')):
    print(evt, elem)

#  ※ 파싱하려는 텍스트가 네임스페이스나 여타 고급 XML 기능을 사용한다면 ElementTree 보다는 lxml 라이브러리를 사용하는 것이 좋다.


#  6.8 관계형 데이터베이스 작업
#  ▣ 문제 : 관계형 데이터베이스에 선택, 삽입, 행 삭제 등의 작업을 하고 싶다.
#  ▣ 해결 : 파이썬에서 데이터 행을 나타내는 표준은 튜플 시퀀스이다.
stocks = [('GOOG', 100, 490.1), ('AAPL', 50, 545.75), ('FB', 150, 7.45), ('HPQ', 75, 33.2)]

#   - sqlite3 데이터베이스 연결
import sqlite3
db = sqlite3.connect('database.db')

#   - 데이터 관련 작업을 위한 커서 생성 및 쿼리 실행
c = db.cursor()
c.execute('create table portfolio (symbol text, shares integer, price real)')
db.commit()

#   - 데이터에 행의 시퀀스를 삽입하려면 다음 구문을 사용한다.
c.executemany('insert into portfolio values (?,?,?)', stocks)
db.commit()

#   - 값 추출
for row in db.execute('select * from portfolio'):
    print(row)

#   - 사용자가 입력한 파라미터를 받는 쿼리를 수행하려면 ? 를 사용해 파라미터를 이스케이핑 해야 한다.
min_price = 100
for row in db.execute('select * from portfolio where price >= ?', (min_price,)):
    print(row)

#  ▣ 토론 : 날짜와 같은 자료를 저장할 때 datetime 모듈의 datetime 인스턴스나 타임스탬프를 사용하는 것이 일반적이다.
#            그리고 금융 자료와 같이 숫자를 저장할 때는 decimal 모듈의 Decimal 인스턴스를 사용하는 경우가 많다.
#            하지만 이에 대한 정확한 매핑은 데이터베이스 백엔드에 따라 달라지기 때문에 관련 문서를 잘 읽어 봐야 한다.
#            또, 절대로 파이썬의 서식화 연산자(% 등)나 .format() 메소드로 문자열을 만들면 안 된다.
#            서식화 연산자에 전달된 값이 사용자의 입력에서 오는 것이라면 SQL 주입 공격을 당할 수 있다.


#  6.9 16 진수 인코딩, 디코딩
#  ▣ 문제 : 문자열로 된 16진수를 바이트 문자열로 디코딩하거나, 바이트 문자열을 16진법으로 인코딩해야 한다.
#  ▣ 해결 : 문자열을 16진수로 인코딩하거나 디코딩하려면 binascii 모듈을 사용한다.
s = b'Hello'

import binascii
h = binascii.b2a_hex(s)  # 16진법으로 인코딩
print(h)
print(binascii.a2b_hex(h))  # 바이트로 디코딩

#   - base64 모듈에도 유사한 기능이 있다.
import base64
h = base64.b16encode(s)  # 16진법으로 인코딩
print(h)
print(base64.b16decode(h))  # 바이트로 디코딩

#  ▣ 토론 : 두 기술의 차이점은 바로 대소문자 구분에 있다.
#            base64.b15decode() 와 base64.b16encode() 함수는 대문자에만 동작하지만 binascii 는 대소문자를 가리지 않는다.

#   - 인코딩 함수가 만들 출력물은 언제나 바이트 문자열이지만 반드시 유니코드를 사용해야 한다면 디코딩 과정을 하나 더 추가해야 한다.
h = base64.b16encode(s)
print(h)
print(h.decode('ascii'))
#  ※ 16진수를 디코딩할 때 b16decode() 와 a2b_hex() 함수는 바이트 혹은 유니코드 문자열을 받는다.
#     하지만 이 문자열에는 반드시 ASCII 로 인코딩한 16진수가 포함되어 있어야 한다.


#  6.10 Base64 인코딩, 디코딩
#  ▣ 문제 : Base64 를 사용한 바이너리 데이터를 인코딩, 디코딩해야 한다.
#  ▣ 해결 : base64 모듈에 b64encode() 와 b64decode() 함수를 사용하면 이 문제를 해결할 수 있다.
s = b'hello'

import base64
a = base64.b64encode(s)  # Base64 로 인코딩
print(a)
print(base64.b64decode(a))  # Base64 를 디코딩

#  ▣ 토론 : Base64 인코딩은 바이트 문자열과 바이트 배열과 같은 바이트 데이터에만 사용하도록 디자인되었다.
#            또한 인코딩의 결과물은 항상 바이트 문자열이 된다. Base64 인코딩 데이터와 유니코드 텍스트를 함께 사용하려면
#            추가적인 디코딩 작업을 거쳐야 한다.
a = base64.b64encode(s).decode('ascii')
print(a)


#  6.13 데이터 요약과 통계 수행
#  ▣ 문제 : 커다란 데이터세트를 요약하거나 통계를 내고 싶다.
#  ▣ 해결 : 통계, 시계열 등과 연관 있는 데이터 분석을 하려면 Pandas 라이브러리를 알아봐야 한다.
import pandas

emp = pandas.read_csv('PythonCookBook/files/emp.csv', skip_footer=1)
print(emp)

#   - 특정 필드에 대해 값의 범위를 조사한다.
print(emp['sal'].unique())  # 특정 필드에 대해 유니크한 값을 리스트로 출력

#   - 데이터 필터링
emp_clerks = emp[emp['job'] == 'CLERK']
print(emp_clerks)

#   - 가장 많은 deptno 2개 추출
emp_dept = emp['deptno'].value_counts()[:2]
print(emp_dept)

#   - hiredate 로 그룹 짓기
dates = emp.groupby('hiredate')
print(dates)

#   - 각 날짜에 대한 카운트 얻기
date_counts = dates.size()
print(date_counts[0:10], type(date_counts))

#   - 카운트 정렬
date_counts.sort()  # sort() 는 deprecated 됨
date_counts.sort_values()
print(date_counts[-10:])

#  ▣ 토론 : 웨스 멕킨리의 Python for Data Analysis 에서 더 많은 정보를 얻을 수 있으니 참고하도록 하자.