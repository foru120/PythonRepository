#todo 4.1 Numpy ndarray: 다차원 배열 객체
'''
    Numpy 의 핵심 기능 중 하나는 N 차원의 배열 객체 또는 ndarray 로, 파이썬에서 사용할 수 있는 대규모 데이터 집합을 담을
    수 있는 빠르고 유연한 자료 구조다. 배열은 스칼라 원소 간의 연산에 사용하는 문법과 비슷한 방식을 사용해서 전체 데이터
    블록에 수학적인 연산을 수행할 수 있도록 해준다.
'''
import numpy as np
data = np.random.randn(2, 3)  # np.random.randn: 2행 3열 배열에 해당하는 random 값을 생성
data

data * 10
data + data
'''
    ndarray 는 같은 종류의 데이터를 담을 수 있는 포괄적인 다차원 배열이며, ndarray 의 모든 원소는 같은 자료형이어야만 한다.
    모든 배열은 각 차원의 크기를 알려주는 shape 라는 튜플과 배열에 저장된 자료형을 알려주는 dtype 이라는 객체를 가지고 있다.
'''
data.shape  # 배열의 행, 열 형태를 반환
data.dtype  # 배열 데이터의 타입을 반환

#todo 4.1.1 ndarray 생성
'''
    배열을 생성하는 가장 쉬운 방법은 array 함수를 이용하는 것이다. 순차적인 객체(다른 배열도 포함하여)를 받아 넘겨받은
    데이터가 들어있는 새로운 Numpy 배열을 생성한다.
'''
data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)
arr1
'''
    같은 길이의 리스트가 담겨있는 순차 데이터는 다차원 배열로 변환이 가능하다.
    다른 길이의 리스트가 담겨있는 순차 데이터는 리스트 자체를 배열의 한 요소로 변환한다.
'''
data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)
arr2.ndim  # ndim: 차원 수를 반환
arr2.shape
'''
    명시적으로 타입을 지정하지 않으면 np.array 는 생성될 때 적절한 자료형을 추정한다.
'''
arr1.dtype
arr2.dtype
'''
    또한 np.array 는 새로운 배열을 생성하기 위한 여러 함수를 가지고 있는데, 예를 들면 zeros 와 ones 는 주어진 길이나 모양에
    각각 0과 1이 들어있는 배열을 생성한다. empty 함수는 초기화되지 않은 배열을 생성하는데, 이런 메소드를 사용해서 다차원
    배열을 생성하려면 원하는 형태의 튜플을 넘기면 된다.
'''
np.zeros(10)
np.zeros((3, 6))
np.empty((2, 3, 2))  # np.empty 는 0 으로 초기화된 배열을 반환하지 않는다. 대부분의 경우 empty 는 초기화되지 않은 값으로 반환한다.
'''
    arange 는 파이썬의 range 함수의 배열 버전이다.
'''
np.arange(15)
np.eye(4)  # N x N 크기의 단위 행렬을 생성한다.

#todo 4.1.2 ndarray 의 자료형
'''
    자료형, dtype 은 ndarray 가 특정 데이터를 메모리에서 해석하기 위해 필요한 정보를 담고 있는 특수한 객체다.
'''
arr1 = np.array([1, 2, 3], dtype=np.float64)
arr2 = np.array([1, 2, 3], dtype=np.int32)
arr1.dtype
arr2.dtype
'''
    ndarray 의 astype 메서드를 사용해서 배열의 dtype 을 다른 형으로 명시적 변경이 가능하다.
'''
arr = np.array([1, 2, 3, 4, 5])
arr.dtype
float_arr = arr.astype(np.float64)  # astype 은 기존의 배열 객체를 copy 한다.
float_arr.dtype
'''
    부동소수점 숫자를 정수형으로 변환하면 소수점 아랫자리는 버려진다.
'''
arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])
arr
arr.astype(np.int32)
'''
    숫자 형식의 문자열을 담고 있는 배열이 있다면 astype 을 사용하여 숫자로 변환할 수 있다.
'''
numeric_strings = np.array(['1.25', '-9.6', '42'], dtype=np.string_)
numeric_strings.astype(float)
'''
    만일 문자열처럼 float64 형으로 변환되지 못하는 경우, 형 변환이 실패하면 TypeError 예외가 발생한다.
'''
int_array = np.arange(10)
calibers = np.array([.22, .270, .357, .380, .44, .50], dtype=np.float64)
int_array.astype(calibers.dtype)
'''
    dtype 으로 사용할 수 있는 축약 코드도 있다. (u4 는 uint32 와 동일하다.)
'''
empty_uint32 = np.empty(8, dtype='u4')
empty_uint32
'''
    astype 을 호출하면 새로운 dtype 이 이전 dtype 과 같아도 항상 새로운 배열을 생성한다.
'''

#todo 4.1.3 배열과 스칼라 간의 연산
'''
    배열은 for 반복문을 작성하지 않고 데이터를 일괄처리할 수 있기 때문에 중요하다. 이를 벡터화라고 하는데, 같은 크기의
    배열 간 산술연산은 배열의 각 요소 단위로 적용된다.
'''
arr = np.array([[1., 2., 3.], [4., 5., 6.]])
arr
arr * arr
arr - arr
'''
    스칼라 값에 대한 산술연산은 각 요소로 전달된다.
'''
1 / arr
arr ** 0.5

#todo 4.1.4 색인과 슬라이싱 기초
arr = np.arange(10)
arr
arr[5]
arr[5:8]
arr[5:8] = 12
arr
'''
    앞에서 봤듯이 arr[5:8] = 12 처럼 배열 조각에 스칼라 값을 대입하면 12 가 선택 영역 전체로 전파된다.
    리스트와의 중요한 차이점은 배열 조각은 원본 배열의 뷰라는 점이다. 즉, 데이터는 복사되지 않고 뷰에 대한 변경은 그대로
    원본 배열에 반영된다는 것이다.
'''
arr_slice = arr[5:8]
arr_slice[1] = 12345
arr
arr_slice[:] = 64
arr
'''
    만약에 뷰 대신 ndarray 슬라이스의 복사본을 얻고 싶다면 arr[5:8].copy() 를 사용해서 명시적으로 배열을 복사하면 된다.
'''
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d[2]
arr2d[0][2]
arr2d[0, 2]
'''
    다차원 배열에서 마지막 색인을 생략하면 반환되는 객체는 상위 차원의 데이터를 포함하고 있는 한 차원 낮은 ndarray 가 된다.
'''
arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d
arr3d[0]
'''
    arr3d[0] 에는 스칼라 값과 배열 모두 대입할 수 있다.
'''
old_values = arr3d[0].copy()
arr3d[0] = 42
arr3d
arr3d[0] = old_values
arr3d
arr3d[1, 0]

#todo 슬라이스 색인
'''
    파이썬의 리스트 같은 1차원 객체처럼 ndarray 는 익숙한 문법으로 슬라이싱 할 수 있다.
'''
arr[1:6]
'''
    다차원 객체는 숫자를 조합해서 하나 이상의 축을 기준으로 슬라이싱을 할 수 있다.
'''
arr2d
arr2d[:2]
arr2d[:2, 1:]
'''
    정수 색인과 슬라이스를 함께 사용하면 한 차원 낮은 슬라이스를 얻을 수 있다.
'''
arr2d[1, :2]
arr2d[2, :1]
'''
    그냥 콜론만 쓰면 전체 축을 선택한다는 의미이므로 이렇게 하면 원래 차원의 슬라이스를 얻을 수 있다.
'''
arr2d[:, :1]
'''
    슬라이싱 구문에 값을 대입하면 선택 영역 전체에 값이 할당된다.
'''
arr2d[:2, 1:] = 0
arr2d

#todo 4.1.5 불리언 색인
'''
    중복된 이름이 포함된 배열이 있다고 하자. 그리고 numpy.random 모듈에 있는 randn 함수를 사용해서 임의의 표준정규분포
    데이터를 생성하자.
'''
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
names
data
'''
    각각의 이름은 data 배열의 각 로우에 대응한다고 가정하자. 만약에 전체 로우에서 'Bob' 과 같은 이름을 선택하려면 산술연산과
    마찬가지로 배열에 대한 비교연산(==같은)도 벡터화된다.
'''
names == 'Bob'
'''
    이 불리언 배열은 배열의 색인으로 사용할 수 있다.
'''
data[names == 'Bob']
'''
    이 불리언 배열은 반드시 색인하려는 축의 길이와 동일한 길이를 가져야 한다. 불리언 배열 색인도 슬라이스 또는 숫자 색인과
    함께 혼용할 수 있다.
'''
data[names == 'Bob', 2:]
data[names == 'Bob', 3]
'''
    'Bob' 이 아닌 요소를 선택하려면 != 연산자를 사용하거나 -(not supported), ~ 를 사용해서 조건절을 부정하면 된다.
'''
names != 'Bob'
data[~(names == 'Bob')]
'''
    세 가지 이름 중에서 두 가지 이름을 선택하려면 &(and)와 |(or) 같은 논리연산자를 사용한 여러 개의 불리언 조건을 조합하여
    사용하면 된다.
'''
mask = (names == 'Bob') | (names == 'Will')
mask
'''
    불리언 배열에 값을 대입하는 것은 상식선에서 이루어지며, data 에 저장된 모든 음수를 0 으로 대입하려면 아래와 같이 하면 된다.
'''
data[data < 0] = 0
data
'''
    1차원 불리언 배열을 사용해서 전체 로우나 칼럼을 선택하는 것은 쉽게 할 수 있다.
'''
data[names != 'Joe'] = 7
data

#todo 4.1.6 팬시 색인
'''
    팬시 색인은 정수 배열을 사용한 색인을 설명하기 위해 NumPy 에서 차용한 단어다. 8 x 4 크기의 배열이 있다고 하자.
'''
arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i
arr
'''
    특정한 순서로 로우를 선택하고 싶다면 그냥 원하는 순서가 명시된 정수가 담긴 ndarray 나 리스트를 넘기면 된다.
'''
arr[[4, 3, 0, 6]]
'''
    색인으로 음수를 사용하면 끝에서부터 로우를 선택한다.
'''
arr[[-3, -5, -7]]
'''
    다차원 색인 배열을 넘기는 것은 조금 다르게 동작하며, 각각의 색인 튜플에 대응하는 1차원 배열이 선택된다.
'''
arr = np.arange(32).reshape((8, 4))
arr
arr[[1, 5, 7, 2], [0, 3, 1, 2]]
'''
    행렬의 행과 열에 대응하는 사각형 모양의 값이 선택되기를 기대했는데 사실 그렇게 하려면 다음처럼 선택해야 한다.
'''
arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]]
'''
    np.ix_ 함수를 사용하면 같은 결과를 얻을 수 있는데, 1차원 정수 배열 2개를 사각형 영역에서 사용할 색인으로 변환해준다.
    팬시 색인은 슬라이싱과는 달리 선택된 데이터를 새로운 배열로 복사한다.
'''
arr[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])]

#todo 4.1.7 배열 전치와 축 바꾸기
'''
    배열 전치는 데이터를 복사하지 않고 데이터 모양이 바뀐 뷰를 반환하는 특별한 기능이다.
    ndarray 는 transpose 메서드와 T 라는 이름의 특수한 속성을 가지고 있다.
'''
arr = np.arange(15).reshape((3, 5))
arr
arr.T
'''
    행렬의 내적 XTX 는 np.dot 을 이용해서 구할 수 있다.
'''
arr = np.random.randn(6, 3)
np.dot(arr.T, arr)
'''
    다차원 배열의 경우 transpose 메서드는 튜플로 축 번호를 받아서 치환한다.
'''
arr = np.arange(16).reshape((2, 2, 4))
arr
arr.transpose((1, 0, 2))
'''
    .T 속성을 이용하는 간단한 전치는 축을 뒤바꾸는 특별한 경우다. ndarray 에는 swapaxes 메서드가 있는데 2 개의 축 번호를
    받아서 배열을 뒤바꾼다.
'''
arr
arr.swapaxes(1, 2)

#todo 4.2 유니버설 함수
'''
    ufunc 라고 불리는 유니버설 함수는 ndarray 안에 있는 데이터 원소별로 연산을 수행하는 함수다.
    유니버설 함수는 하나 이상의 스칼라 값을 받아서 하나 이상의 스칼라 결과 값을 반환하는 간단한 함수를 고속으로 수행할 수 
    있는 벡터화된 래퍼 함수라고 생각하면 된다.
'''
arr = np.arange(10)
np.sqrt(arr)
np.exp(arr)
'''
    add 나 maximum 처럼 2 개의 인자를 취해서 단일 배열을 반환하는 함수를 이항 유니버설 함수라고 한다.
'''
x = np.random.randn(8)
y = np.random.randn(8)
x
y
np.maximum(x, y)
'''
    배열 여러 개를 반환하는 유니버설 함수도 있다. modf 는 파이썬 내장 함수인 divmod 의 벡터화 버전이며, modf 는 분수를
    받아 몫과 나머지를 함께 반환한다.
'''
arr = np.random.randn(7) * 5
np.modf(arr)