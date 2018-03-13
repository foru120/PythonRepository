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