#todo chapter5. pandas 시작하기
from pandas import Series, DataFrame
import pandas as pd
import numpy as np

#todo 5.1 pandas 자료 구조 소개
'''
    pandas 에 대해서 알아보려면 Series 와 DataFrame, 이 두 가지 자료 구조에 익숙해질 필요가 있다.
    이 두 가지 자료 구조로 모든 문제를 해결할 수는 없지만 대부분의 애플리케이션에서 사용하기 쉽고 탄탄한 기반을 제공한다.
'''
#todo 5.1.1 Series
'''
    Series 는 일련의 객체를 담을 수 있는 1차원 배열 같은 자료 구조다. (어떤 NumPy 자료형이라도 담을 수 있다.)
    그리고 색인이라고 하는 배열의 데이터에 연관된 이름을 가지고 있다.
    가장 간단한 Series 객체는 배열 데이터로부터 생성할 수 있다.
'''
obj = Series([4, 7, -5, 3])
obj
'''
    Series 객체의 문자열 표현은 왼쪽에 색인을 보여주고 오른쪽에 해당 색인의 값을 보여준다.
    앞의 예제에서는 데이터의 색인을 지정하지 않았으니 기본 색인인 정수 0 에서 N-1(N은 데이터의 길이)까지의 숫자가 표시된다.
    Series 의 배열과 색인 객체는 각각 values 와 index 속성을 통해 얻을 수 있다.
'''
obj.values
obj.index
'''
    각각의 데이터를 지칭하는 색인을 지정해 Series 객체를 생성해야 할 때는 다음처럼 생성한다.
'''
obj2 = Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
obj2
obj2.index
'''
    배열에서 값을 선택하거나 대입할 때는 색인을 이용해서 접근한다.
'''
obj2['a']
obj2['d'] = 6
obj2[['c', 'a', 'd']]
'''
    불리언 배열을 사용해서 값을 걸러내거나 산술 곱셈을 수행하거나 또는 수학 함수를 적용하는 등 NumPy 배열연산을 수행해도
    색인-값 연결은 유지된다.
'''
obj2[obj2 > 0]
obj2 * 2
np.exp(obj2)
'''
    Series 를 이해하는 다른 방법은 고정 길이의 정렬된 사전형이라고 이해하는 것이다.
    Series 는 색인 값에 데이터 값을 매핑하고 있으므로 파이썬의 사전형과 비슷하다.
    Series 객체는 파이썬의 사전형을 인자로 받아야 하는 많은 함수에서 사전형을 대체하여 사용할 수 있다.
'''
'b' in obj2
'e' in obj2
'''
    파이썬 사전형에 데이터를 저장해야 한다면 파이썬 사전 객체로부터 Series 객체를 생성할 수 있다.
'''
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = Series(sdata)
obj3
'''
    사전 객체만 가지고 Series 객체를 생성하면 생성된 Series 객체의 색인은 사전의 키 값이 순서대로 들어간다.
'''
states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = Series(sdata, index=states)
obj4
'''
    이 예제를 보면 sdata 에 있는 값 중 3 개만 확인할 수 있는데, 이는 'California'에 대한 값을 찾을 수 없기 때문이다.
    이 값은 NaN(not a number)으로 표시되고 pandas 에서는 누락된 값 혹은 NA 값으로 취급된다.
    나는 앞으로 '누락된'과 'NA'를 누락된 데이터를 지칭하는 데 사용하도록 하겠다.
    pandas 의 isnull 과 notnull 함수는 누락된 데이터를 찾을 때 사용된다.
'''
pd.isnull(obj4)
pd.notnull(obj4)
'''
    이 메서드는 Series 의 인스턴스 메서드이기도 하다.
'''
obj4.isnull()
'''
    누락된 데이터를 처리하는 방법은 이 장의 끝부분에서 좀 더 자세히 살펴보기로 하자.
    가장 중요한 Series 의 기능은 다르게 색인된 데이터에 대한 산술연산이다.
'''
obj3
obj4
obj3 + obj4  # 둘 중에 하나만 존재하는 키에 대해서는 NaN 값이 출력된다.
'''
    Series 객체와 Series 의 색인은 모두 name 속성이 있는데, 이 속성은 pandas 의 기능에서 중요한 부분을 차지하고 있다.
'''
obj4.name = 'population'
obj4.index.name = 'state'
obj4
'''
    Series 의 색인은 대입을 통해 변경할 수 있다.
'''
obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
obj

#todo 5.1.2 DataFrame
'''
    DataFrame 은 표 같은 스프레드시트 형식의 자료 구조로 여러 개의 칼럼이 있는데, 각 칼럼은 서로 다른 종류의 값(숫자, 문자열,
    불리언 등)을 담을 수 있다.
    DataFrame 은 로우와 칼럼에 대한 색인이 있는데, 이 DataFrame 은 색인의 모양이 같은 Series 객체를 담고 있는 파이썬 사전으로
    생각하면 편하다.
    R 의 data.frame 같은 다른 DataFrame 과 비슷한 자료 구조와 비교했을 때, DataFrame 에서의 로우 연산과 칼럼 연산은 거의
    대칭적으로 취급된다.
    내부적으로 데이터는 리스트나 사전 또는 1차원 배열을 담고 있는 다른 컬렉션이 아니라 하나 이상의 2차원 배열에 저장된다.
    구체적인 DataFrame 의 내부 구조는 이 책에서 다루는 내용에서 벗어나므로 생략하겠다.
'''
'''
    DataFrame 객체는 다양한 방법으로 생성할 수 있지만 가장 흔하게 사용되는 방법은 같은 길이의 리스트에 담긴 사전을 이용하거나
    NumPy 배열을 이용하는 방법이다.
'''
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
frame = DataFrame(data)
'''
    만들어진 DataFrame 의 색인은 Series 와 같은 방식으로 자동으로 대입되며 칼럼은 정렬되어 저장된다.
'''
frame
'''
    원하는 순서대로 columns 를 지정하면 원하는 순서를 가진 DataFrame 객체가 생성된다.
'''
DataFrame(data, columns=['year', 'state', 'pop'])
'''
    Series 와 마찬가지로 data 에 없는 값을 넘기면 NA 값이 저장된다.
'''
frame2 = DataFrame(data, columns=['year', 'state', 'pop', 'debt'], index=['one', 'two', 'three', 'four', 'five'])
frame2
frame2.columns
'''
    DataFrame 의 칼럼은 Series 처럼 사전 형식의 표기법으로 접근하거나 속성 형식으로 접근할 수 있다.
'''
frame2['state']
frame2.year
'''
    반환된 Series 객체가 DataFrame 같은 색인을 가지면 알맞은 값으로 name 속성이 채워진다.
    로우는 위치나 ix 같은 몇 가지 메서드를 통해 접근할 수 있다.
'''
frame2.ix['three']  # .ix is deprecated.
frame2.loc['three']  # .loc for label based indexsing or
frame2.iloc[2]  # .iloc for positional indexing
'''
    칼럼은 대입이 가능하다.
    예를 들면 현재 비어있는 'debt' 칼럼에 스칼라 값이나 배열의 값을 대입할 수 있다.
'''
frame2['debt'] = 16.5  # 전체 행에 대해 같은 값이 들어간다.
frame2
frame2['debt'] = np.arange(5.)
frame2
'''
    리스트나 배열을 칼럼에 대입할 때는 대입하려는 값의 길이가 DataFrame 의 크기와 같아야 한다.
    Series 를 대입하려면 DataFrame 의 색인에 따라 값이 대입되며 없는 색인에는 값이 대입되지 않는다.
'''
val = Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
frame2['debt'] = val
frame2
'''
    없는 칼럼을 대입하면 새로운 칼럼이 생성된다.
    파이썬 사전형에서와 마찬가지로 del 예약어를 사용해서 칼럼을 삭제할 수 있다.
'''
frame2['eastern'] = frame2.state == 'Ohio'
frame2
del frame2['eastern']
frame2.columns
'''
    DataFrame 의 색인을 이용해서 생성된 칼럼은 내부 데이터에 대한 뷰이며 복사가 이루어지지 않는다.
    따라서 이렇게 얻은 Series 객체에 대한 변경은 실제 DataFrame 에 반영된다.
    복사본이 필요할 때는 Series 의 copy 메서드를 이용하자.
'''
'''
    또한 중첩된 사전을 이용해서 데이터를 생성할 수 있는데, 다음과 같은 중첩된 사전이 있다면
'''
pop = {'Nevada': {2001: 2.4, 2002: 2.9}, 'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
'''
    바깥에 있는 사전의 키 값이 칼럼이 되고 안에 있는 키는 로우가 된다.
'''
frame3 = DataFrame(pop)
frame3
'''
    NumPy 에서와 마찬가지로 결과 값의 순서를 뒤집을 수 있다.
'''
frame3.T
'''
    중첩된 사전을 이용해서 DataFrame 을 생성할 때 안쪽에 있는 사전 값은 키 값별로 조합되어 결과의 색인이 되지만 색인을
    직접 지정한다면 지정된 색인으로 DataFrame 을 생성한다.
'''
DataFrame(pop, index=[2001, 2002, 2003])
'''
    Series 객체를 담고 있는 사전 데이터도 같은 방식으로 취급된다.
'''
pdata = {'Ohio': frame3['Ohio'][:-1], 'Nevada': frame3['Nevada'][:2]}
DataFrame(pdata)
'''
    DataFrame 생성자에 넘길 수 있는 자료형의 목록은 [표 5-1]을 참고하자.
'''
frame3.index.name = 'year'
frame3.columns.name = 'state'
frame3
'''
    Series 와 유사하게 values 속성은 DataFrame 에 저장된 데이터를 2차원 배열로 반환한다.
'''
frame3.values
'''
    DataFrame 의 칼럼에 서로 다른 dtype 이 있다면 모든 칼럼을 수용하기 위해 그 칼럼 배열의 dtype 이 선택된다.
'''
frame2.values

#todo 5.1.3 색인 객체
'''
    pandas 의 색인 객체는 표 형식의 데이터에서 각 로우와 칼럼에 대한 이름과 다른 메타데이터(축의 이름 등)를 저장하는
    객체다.
    Series 나 DataFrame 객체를 생성할 때 사용되는 배열이나 혹은 다른 순차적인 이름은 내부적으로 색인으로 변환된다.
'''
obj = Series(range(3), index=['a', 'b', 'c'])
index = obj.index
index
index[1:]
'''
    색인 객체는 변경할 수 없다.
'''
index[1] = 'd'
'''
    색인 객체는 변경할 수 없기에 자료 구조 사이에서 안전하게 공유될 수 있다.
'''
index = pd.Index(np.arange(3))
obj2 = Series([1.5, -2.5, 0], index=index)
obj2.index is index
'''
    [표 5-2]에 pandas 에서 사용하는 내장 색인 클래스가 정리되어 있다.
    특수한 목적으로 축을 색인하는 기능을 개발하기 위해 Index 클래스의 서브 클래스를 만들 수 있다.
'''
'''
    또한 배열과 유사하게 Index 객체도 고정 크기로 동작한다.
'''
frame3
'Ohio' in frame3.columns
2003 in frame3.index
'''
    각각의 색인은 담고 있는 데이터에 대한 정보를 취급하는 여러 가지 메서드와 속성을 가지고 있다.
'''

#todo 5.2 핵심 기능
'''
    이 절에서는 Series 나 DataFrame 에 저장된 데이터를 다루는 기본 방법을 설명한다.
    앞으로 pandas 를 이용한 데이터 분석과 조작에 관한 좀 더 자세한 내용을 살펴볼 것이다.
    이 책은 pandas 라이브러리에 대한 완전한 설명은 자제하고 중요한 기능에만 초점을 맞추고 있다.
    잘 사용하지 않는 내용에 대한 학습은 독자의 몫으로 남겨둔다.
'''

#todo 5.2.1 재색인
'''
    pandas 객체의 기막힌 기능 중 하나인 reindex 는 새로운 색인에 맞도록 객체를 새로 생성하는 기능이다.
'''
obj = Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
obj
'''
    이 Series 객체에 대해 reindex 를 호출하면 데이터를 새로운 색인에 맞게 재배열하고, 없는 색인 값이 있다면 비어있는 값을
    새로 추가한다.
'''
obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
obj2
obj.reindex(['a', 'b', 'c', 'd', 'e'], fill_value=0)
'''
    시계열 같은 순차적인 데이터를 재색인할 때 값을 보간하거나 채워 넣어야 할 경우가 있다.
    이런 경우 method 옵션을 이용해서 해결할 수 있으며, ffill 메서드를 이용하면 앞의 값으로 누락된 값을 채워 넣을 수 있다.
'''
obj3 = Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
obj3.reindex(range(6), method='ffill')  # 앞의 값으로 누락된 값을 채움
obj3.reindex(range(6), method='pad')  # 앞의 값으로 누락된 값을 채움

obj3.reindex(range(6), method='bfill')  # 뒤의 값으로 누락된 값을 채움
obj3.reindex(range(6), method='backfill')  # 뒤의 값으로 누락된 값을 채움
'''
    DataFrame 에 대한 reindex 는 (로우)색인, 칼럼 또는 둘 다 변경이 가능하다.
    그냥 순서만 전달하면 로우가 재색인된다.
'''
frame = DataFrame(np.arange(9).reshape((3, 3)), index=['a', 'c', 'd'], columns=['Ohio', 'Texas', 'California'])
frame
frame2 = frame.reindex(['a', 'b', 'c', 'd'])
frame2
'''
    열은 columns 예약어를 사용해서 재색인할 수 있다.
'''
states = ['Texas', 'Utah', 'California']
frame.reindex(columns=states)
'''
    로우와 칼럼을 모두 한 번에 재색인할 수 있지만 보간은 로우에 대해서만 이루어진다. (axis 0)
'''
frame.reindex(index=['a', 'b', 'c', 'd'], method='ffill').reindex(columns=states)
'''
    재색인은 ix 를 이용해서 라벨로 색인하면 좀 더 간결하게 할 수 있다.
'''
frame.ix[['a', 'b', 'c', 'd'], states]  # iloc(integer position), loc(label-based indexing), ix(iloc or loc)

#todo 5.2.2 하나의 로우 또는 칼럼 삭제하기
'''
    색인 배열 또는 삭제하려는 로우나 칼럼이 제외된 리스트를 이미 가지고 있다면 로우나 칼럼을 쉽게 삭제할 수 있는데, 이 
    방법은 데이터의 모양을 변경하는 작업이 필요하다.
    drop 메서드를 사용하면 선택한 값이 삭제된 새로운 객체를 얻을 수 있다.
'''
obj = Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
new_obj = obj.drop('c')
new_obj
obj.drop(['d', 'c'])
'''
    DataFrame 에서는 로우와 칼럼 모두에서 값을 삭제할 수 있다.
'''
data = DataFrame(np.arange(16).reshape((4, 4)), index=['Ohio', 'Colorado', 'Utah', 'New York'], columns=['one', 'two', 'three', 'four'])
data.drop(['Colorado', 'Ohio'])  # 해당 로우의 값을 삭제
data.drop('two', axis=1)  # 2차원 배열 기준 (axis=0 : 행, axis=1 : 열)
data.drop(['two', 'four'], axis=1)

#todo 5.2.3 색인하기, 선택하기, 거르기
'''
    Series 의 색인(obj[...])은 NumPy 배열의 색인과 유사하게 동작하는데, Series 의 색인은 정수가 아니어도 된다는 점이 다르다.
    몇 가지 예제를 살펴보자.
'''
obj = Series(np.arange(4.), index=['a', 'b', 'c', 'd'])
obj['b']
obj[1]
obj[2:4]
obj[['b', 'a', 'd']]
obj[[1, 3]]
obj[obj<2]
'''
    라벨 이름으로 슬라이싱하는 것은 시작점과 끝점을 포함한다는 점이 일반 파이썬에서의 슬라이싱과 다른 점이다.
'''
obj['b':'c']
'''
    슬라이싱 문법으로 선택된 영역에 값을 대입하는 것은 예상한 대로 동작한다.
'''
obj['b':'c'] = 5
obj
'''
    앞에서 확인했듯이 색인으로 DataFrame 에서 칼럼의 값을 하나 이상 가져올 수 있다.
'''
data = DataFrame(np.arange(16).reshape((4, 4)), index=['Ohio', 'Colorado', 'Utah', 'New York'], columns=['one', 'two', 'three', 'four'])
data
data['two']
data[['three', 'one']]
'''
    슬라이싱으로 로우를 선택하거나 불리언 배열로 로우를 선택할 수도 있다.
'''
data[:2]
data[data['three'] > 5]
'''
    이 문법에 모순이 있다고 생각하는 분이 있을지도 모르겠지만 이 문법은 실용성에 기인한 것일 뿐이다.
    또 다른 사례는 스칼라 비교를 통해 생성된 불리언 DataFrame 을 사용해서 값을 선택하는 것이다.
'''
data < 5
data[data < 5] = 0
data
'''
    이 예제는 DataFrame 을 ndarray 와 문법적으로 비슷하게 보이도록 의도한 것이다.
    DataFrame 의 칼럼에 대해 라벨로 색인하는 방법으로, 특수한 색인 필드인 ix 를 소개한다.
    ix 는 NumPy 와 비슷한 방식에 추가적으로 축의 라벨을 사용하여 DataFrame 의 로우와 칼럼을 선택할 수 있도록 한다.
    앞에서 언급했듯이 이 방법은 재색인을 좀 더 간단하게 할 수 있는 방법이다.
'''
data.ix['Colorado', ['two', 'three']]  # ix is deprecated. label based indexing -> .loc, positional indexing -> .iloc
data.ix[['Colorado', 'Utah'], [3, 0, 1]]  # ix 메서드는 label + positional indexing 을 같이 사용할 때 사용한다.
data.loc[['Colorado', 'Utah'], ['four', 'one', 'two']]
data.iloc[2]
data.loc[:'Utah', 'two']
data.ix[data.three > 5, :3]

#todo 5.2.4 산술연산과 데이터 정렬
'''
    pandas 에서 중요한 기능은 색인이 다른 객체 간의 산술연산이다.
    객체를 더할 때 짝이 맞지 않는 색인이 있다면 결과에 두 색인이 통합된다.
'''
s1 = Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
s2 = Series([-2.1, 3.6, -1.5, 4, 3.1], index=['a', 'c', 'e', 'f', 'g'])
s1 + s2
'''
    서로 겹치는 색인이 없다면 데이터는 NA 값이 된다.
    산술연산 시 누락된 값은 전파되며, DataFrame 에서는 로우와 칼럼 모두에 적용된다.
'''
df1 = DataFrame(np.arange(9.).reshape((3, 3)), columns=list('bcd'), index=['Ohio', 'Texas', 'Colorado'])
df2 = DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])
df1
df2
df1 + df2

#todo 산술연산 메서드에 채워 넣을 값 지정하기
'''
    서로 다른 색인을 가지는 객체 간의 산술연산에서 존재하지 않는 축의 값을 특수한 값(0 같은)으로 지정하고 싶을 때는 다음과
    같이 할 수 있다.
    add : 덧셈을 위한 메서드
    sub : 뺄셈을 위한 메서드
    div : 나눗셈을 위한 메서드
    mul : 곱셈을 위한 메서드
'''
df1 = DataFrame(np.arange(12.).reshape((3, 4)), columns=list('abcd'))
df2 = DataFrame(np.arange(20.).reshape((4, 5)), columns=list('abcde'))
df1 + df2
'''
    df1 의 add 메서드로 df2 와 fill_value 값을 인자로 전달한다.
'''
df1.add(df2, fill_value=0)  # fill_value 는 한 쪽이라도 존재하지 않는 값은 해당 DataFrame 에서 0 으로 설정한다.
'''
    Series 나 DataFrame 을 재색인할 때 역시 fill_value 를 지정할 수 있다.
'''
df1.reindex(columns=df2.columns, fill_value=0)

#todo DataFrame 과 Series 간의 연산
'''
    NumPy 배열의 연산처럼 DataFrame 과 Series 간의 연산도 잘 정의되어 있다.
    먼저 2차원 배열과 그 배열 중 한 칼럼의 차이에 대해서 생각할 수 있는 예제를 살펴보자.
'''
arr = np.arange(12.).reshape((3, 4))
arr
arr[0]
arr - arr[0]
'''
    이 예제는 브로드캐스팅에 대한 예제로, 자세한 내용은 12 장에서 살펴볼 것이다.
    DataFrame 과 Series 간의 연산은 이와 유사하다.
'''
frame = DataFrame(np.arange(12.).reshape((4, 3)), columns=list('bde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])
series = frame.ix[0]
frame
series
'''
    기본적으로 DataFrame 과 Series 간의 산술연산은 Series 의 색인을 DataFrame 의 칼럼에 맞추고 아래 로우로 전파한다.
'''
frame - series
'''
    만약 색인 값을 DataFrame 의 칼럼이나 Series 의 색인에서 찾을 수 없다면 그 객체는 형식을 맞추기 위해 재색인된다.
'''
series2 = Series(range(3), index=['b', 'e', 'f'])
frame + series2
'''
    만약 각 로우에 대해 연산을 수행하고 싶다면 산술연산 메서드를 사용하면 된다.
'''
series3 = frame['d']
frame
series3
frame.sub(series3, axis=0)
'''
    이 예에서 인자로 넘기는 axis 값은 연산을 적용할 축 번호이며, 여기서 axis=0 은 DataFrame 의 로우를 따라 연산을 수행하
    라는 의미다.
'''

#todo 5.2.5 함수 적용과 매핑
'''
    pandas 객체에도 NumPy 의 유니버설 함수(배열의 각 원소에 적용되는 메서드)를 적용할 수 있다.
'''
frame = DataFrame(np.random.randn(4, 3), columns=list('bde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])
frame
np.abs(frame)
'''
    자주 사용되는 또 다른 연산은 각 로우나 칼럼의 1차원 배열에 함수를 적용하는 것이다.
    DataFrame 의 apply 메서드를 통해 수행할 수 있다.
'''
f = lambda x: x.max() - x.min()
frame.apply(f)
frame.apply(f, axis=1)
'''
    배열의 합계나 평균같은 일반적인 통계는 DataFrame 의 메서드로 있으므로 apply 메서드를 사용해야만 하는 것은 아니다.
    apply 메서드에 전달된 함수는 스칼라 값을 반환할 필요가 없으며, Series 또는 여러 값을 반환해도 된다.
'''
def f(x):
    return Series([x.min(), x.max()], index=['min', 'max'])

frame.apply(f)
'''
    배열의 각 원소에 적용되는 파이썬의 함수를 사용할 수도 있다.
    frame 객체에서 실수 값을 문자열 포맷으로 변환하고 싶다면 applymap 을 이용해서 다음과 같이 해도 된다.
'''
format = lambda x: '%.2f' % x
frame.applymap(format)
'''
    이 메서드의 이름이 applymap 인 이유는 Series 가 각 원소에 적용할 함수를 지정하기 위한 map 메서드를 가지고 있기 때문
    이다.
'''
frame['e'].map(format)

#todo 5.2.6 정렬과 순위
'''
    어떤 기준에 근거해서 데이터를 정렬하는 것 역시 중요한 명령이다.
    로우나 칼럼의 색인을 알파벳 순으로 정렬하려면 새로운 객체를 반환하는 sort_index 메서드를 사용하면 된다.
'''
obj = Series(range(4), index=['d', 'a', 'b', 'c'])
obj.sort_index()
'''
    DataFrame 은 로우나 칼럼 중 하나의 축을 기준으로 정렬할 수 있다.
'''
frame = DataFrame(np.arange(8).reshape((2, 4)), index=['three', 'one'], columns=['d', 'a', 'b', 'c'])
frame.sort_index()  # 로우 축을 기준으로 정렬
frame.sort_index(axis=1)  # 칼럼 축을 기준으로 정렬
'''
    데이터는 기본적으로 오름차순으로 정렬되지만 내림차순으로 정렬할 수도 있다.
'''
frame.sort_index(axis=1, ascending=False)
'''
    Series 객체를 값에 따라 정렬하고 싶다면 sort_values 메서드를 사용하자.
'''
obj = Series([4, 7, -3, 2])
obj.sort_values()
'''
    정렬할 때 비어있는 값은 기본적으로 Series 객체에서 가장 마지막에 위치한다.
'''
obj = Series([4, np.nan, 7, np.nan, -3, 2])
obj.sort_values()
'''
    DataFrame 에서는 하나 이상의 칼럼에 있는 값으로 정렬이 필요할 수 있다.
    이럴 때는 by 옵션에 필요한 칼럼의 이름을 넘기면 된다.
'''
frame = DataFrame({'b': [4, 7,-3, 2], 'a': [0, 1, 0, 1]})
frame
frame.sort_values(by='b')
'''
    여러 개의 칼럼을 정렬하려면 칼럼의 이름이 담긴 리스트를 전달하면 된다.
'''
frame.sort_values(by=['a', 'b'])
'''
    순위는 정렬과 거의 흡사하며, 1부터 배열의 유효한 데이터 개수까지의 순위를 매긴다.
    또한 순위는 numpy.argsort 에서 반환하는 간접 정렬 색인과 유사한데, 동률인 순위를 처리하는 방식이 다르다.
    기본적으로 Series 와 DataFrame 의 rank 메서드는 동점인 항목에 대해서는 평균 순위를 매긴다.    
'''
obj = Series([7, -5, 7, 4, 2, 0 ,4])
obj.rank()
'''
    데이터 상에서 나타나는 순서에 따라 순위를 매길 수도 있다.
'''
obj.rank(method='first')
'''
    내림차순으로 순위를 매길 수도 있다.
'''
obj.rank(ascending=False, method='max')
'''
    DataFrame 에서는 로우나 칼럼에 대해 순위를 정할 수 있다.
'''
frame = DataFrame({'b': [4.3, 7, -3, 2], 'a': [0, 1, 0, 1], 'c': [-2, 5, 8, -2.5]})
frame
frame.rank(axis=1)