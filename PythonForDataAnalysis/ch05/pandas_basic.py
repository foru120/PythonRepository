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
