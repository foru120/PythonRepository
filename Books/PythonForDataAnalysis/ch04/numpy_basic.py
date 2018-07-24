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

#todo 4.3 배열을 사용한 데이터 처리
'''
    Numpy 배열을 사용하면 반복문을 작성하지 않고 간결한 배열연산을 통해 많은 종류의 데이터 처리 작업을 할 수 있다.
    배열연산을 사용해서 반복문을 명시적으로 제거하는 기법을 흔히 벡터화라고 부르는데, 일반적으로 벡터화된 배열에 대한
    산술연산은 순수 파이썬 연산에 비해 2~3배에서 많게는 수십, 수백 배까지 빠르다.
'''
'''
    np.meshgrid 함수는 2개의 1차원 배열을 받아 가능한 한 모든 (x, y) 짝을 만들 수 있는 2차원 배열 2개를 반환한다.
'''
import numpy as np
points = np.arange(-5, 5, 0.01)
xs, ys = np.meshgrid(points, points)
xs
ys
'''
    이제 그리드 상의 두 포인트를 가지고 간단하게 계산을 적용할 수 있다.
'''
import matplotlib.pyplot as plt
z = np.sqrt(xs ** 2 + ys ** 2)
z
plt.imshow(z, cmap=plt.cm.gray); plt.colorbar()
plt.title('Image plot of $\sqrt{x^2 + y^2}$ for a grid of values')
plt.show()

#todo 4.3.1 배열연산으로 조건절 표현하기
'''
    numpy.where 함수는 'x if 조건 else y' 같은 삼항식의 벡터화된 버전이다. 다음과 같이 불리언 배열 하나와 값이 들어있는
    2개의 배열이 있다고 하자.
'''
xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])
'''
    cond 의 값이 True 일 때, xarr 의 값이나 yarr 의 값을 취하고 싶다면 리스트 내포를 이용해서 다음처럼 작성할 수 있다.
'''
result = [(x if c else y) for x, y, c in zip(xarr, yarr, cond)]
result
'''
    이 방법에는 몇 가지 문제, 즉 순수 파이썬으로 수행되기 때문에 큰 배열을 빠르게 처리하지 못한다는 것과 다차원 배열에서는
    사용할 수 없다는 문제가 있다. 하지만 np.where 를 사용하면 아주 간결하게 작성할 수 있다.
'''
result = np.where(cond, xarr, yarr)
result
'''
    np.where 의 두 번째와 세 번째 인자는 배열이 아니라도 괜찮다. 둘 중 하나 혹은 둘 다 스칼라 값이라도 동작한다.
    데이터 분석에서 일반적인 where 의 사용은 다른 배열에 기반한 새로운 배열을 생성한다.
'''
arr = np.random.randn(4, 4)
arr
np.where(arr > 0, 2, -2)
np.where(arr > 0, 2, arr)
'''
    where 에 넘긴 배열은 같은 크기의 배열이거나 스칼라 값일 수 있다.
'''
cond1 = np.array([True, False, False, True, True])
cond2 = np.array([False, False, True, True, False])
np.where(cond1 & cond2, 0, np.where(cond1, 1, np.where(cond2, 2, 3)))
'''
    불리언 값은 0이거나 1인 값만을 취하므로 가독성이 떨어지긴 해도 다음 코드처럼 산술연산만으로 표현하는 것도 가능하다.
'''
result = 1 * (cond1 & ~cond2) + 2 * (cond2 & ~cond1) + 3 * (~cond1 & ~cond2)
result

#todo 4.3.2 수학 메서드와 통계 메서드
import numpy as np
arr = np.random.randn(5, 4)
arr.mean()
np.mean(arr)
arr.sum()
'''
    mean 이나 sum 같은 함수는 선택적으로 axis 인자를 받아 해당 axis 에 대한 통계를 계산하고 한 차수 낮은 배열을 반환한다.
'''
arr.mean(axis=1)
arr.sum(0)
'''
    cumsum 과 cumprod 메서드는 중간 계산 값을 담고 있는 배열을 반환한다.
    지정된 축에 따라 누적된 결과 값을 보여준다.
'''
arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
arr.cumsum(0)
arr.cumprod(1)

#todo 4.3.3 불리언 배열을 위한 메서드
'''
    앞의 메서드에서 불리언 값은 1(True), 0(False) 으로 취급된다. 따라서 불리언 배열에 대한 sum 메서드를 실행하면 True 인
    원소의 개수를 반환한다.
'''
arr = np.random.randn(100)
(arr > 0).sum()
'''
    any, all 메서드는 불리언 배열에 사용할 때 특히 유용하다. any 메서드는 하나 이상의 True 값이 있는지 검사하고, all 
    메서드는 모든 원소가 True 인지 검사한다.
'''
bools = np.array([False, False, True, False])
bools.any()
bools.all()
'''
    any(), all() 메서드는 불리언 배열이 아니어도 동작하며, 0 이 아닌 원소는 모두 True 로 간주한다.
'''

#todo 4.3.4 정렬
'''
    파이썬의 내장 리스트형처럼 NumPy 배열 역시 sort 메서드를 이용해서 정렬할 수 있다.
'''
arr = np.random.randn(8)
arr
arr.sort()
arr
'''
    다차원 배열의 정렬은 sort 메서드에 넘긴 축의 값에 따라 1차원 부분을 정렬한다.
'''
arr = np.random.randn(5, 3)
arr
arr.sort(1)
arr
'''
    np.sort 메서드는 배열을 직접 변경하지 않고 정렬된 결과를 가지고 복사본을 반환한다.
    배열의 분위수를 구하는 쉽고 빠른 방법은 우선 배열을 정렬한 후에 특정 분위의 값을 선택하는 것이다.
'''
large_arr = np.random.randn(1000)
large_arr.sort()
large_arr[int(0.05 * len(large_arr))]  # 5% quantile

#todo 4.3.5 집합 함수
'''
    NumPy 는 1차원 ndarray 를 위한 몇 가지 기본 집합연산을 제공한다. 아마도 가장 자주 사용되는 함수는 배열 내에서 중복된
    원소를 제거하고 남은 원소를 정렬된 형태로 반환하는 np.unique 일 것이다.
'''
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names)
ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
np.unique(ints)
'''
    np.unique 를 순수 파이썬으로만 구현하면 다음과 같다.
'''
sorted(set(ints))
'''
    np.in1d 함수는 2 개의 배열을 인자로 받아 첫 번째 배열의 각 원소가 두 번째 배열의 원소를 포함하는지를 나타내는 불리언
    배열을 반환한다.
'''
values = np.array([6, 0, 0, 3, 2, 5, 6])
np.in1d(values, [2, 3, 6])

#todo 4.4 배열의 파일 입·출력
'''
    NumPy 는 디스크에서 텍스트나 바이너리 형식의 파일로부터 데이터를 불러오거나 저장할 수 있다.
'''

#todo 4.4.1 배열을 바이너리 형식으로 디스크에 저장하기
'''
    np.save 와 np.load 는 배열 데이터를 효과적으로 디스크에 저장하고 불러오는 함수다.
    배열은 기본적으로 압축되지 않은 raw 바이너리 형식의 .npy 파일로 저장된다.
'''
arr = np.arange(10)
np.save('some_array', arr)
'''
    저장되는 파일 경로가 .npy 로 끝나지 않으면 자동적으로 확장자를 추가한다.
'''
np.load('some_array')
'''
    np.savez 함수를 이용하면 여러 개의 배열을 압축된 형식으로 저장할 수 있는데, 저장하려는 배열은 키워드 인자 형태로 
    전달된다.
'''
np.savez('array_archive.npz', a=arr, b=arr)
'''
    npz 파일을 불러올 때는 각각의 배열을 언제라도 불러올 수 있게 사전 형식의 객체에 저장한다.
'''
arch = np.load('array_archive.npz')
arch['b']

#todo 4.4.2 텍스트 파일 불러오기와 저장하기
'''
    np.loadtxt 함수는 구분자를 지정하거나 특정 칼럼에 대한 변환 함수를 지정하거나 로우를 건너뛰는 등의 다양한 기능을 
    제공한다.
'''
arr = np.loadtxt('PythonForDataAnalysis\\ch04\\array_ex.txt', delimiter=',')
arr
'''
    np.savetxt 는 앞에서 살펴본 np.loadtxt 와 반대로 배열을 파일로 저장한다.
    genfromtxt 는 loadtxt 와 유사하지만 구조화된 배열과 누락된 데이터 처리를 위해 설계되었다.
'''

#todo 4.5 선형대수
'''
    행렬의 곱셈, 분할, 행렬식, 정사각 행렬 수학 같은 선형대수는 배열을 다루는 라이브러리에서 중요한 부분이다.
    MATLAB 같은 다른 언어와 달리 2개의 2차원 배열을 * 연산자로 곱하는 건 행렬 곱셈이 아니라 대응하는 각각의 원소의 곱을
    계산하는 것이다.
    행렬 곱셈은 배열 메서드이자 numpy 네임스페이스 안에 있는 함수인 dot 함수를 사용해서 계산한다.
'''
x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
x
x.dot(y)
'''
    2 차원 배열과 곱셈이 가능한 크기의 1 차원 배열 간 행렬 곱셈의 결과는 1차원 배열이다.
'''
np.dot(x, np.ones(3))
'''
    numpy.linalg 는 행렬의 분할과 역행렬, 행렬식 같은 것을 포함하고 있다.
    이는 MATLAB, R 같은 언어에서 사용하는 표준 포트란 라이브러리인 BLAS, LAPACK 또는 Intel MKL 를 사용해서 구현되었다.
'''
from numpy.linalg import inv, qr
X = np.random.randn(5, 5)
mat = X.T.dot(X)
inv(mat)  # inv(inverse of a matrix): 정사각행렬의 역행렬을 계산
mat.dot(inv(mat))
q, r = qr(mat)  # qr(QR Factorization): 행렬 분해의 일종으로 임의의 행렬을 직교행렬과 상삼각행렬의 곱으로 분해하는 방법
r

#todo 4.6 난수 생성
'''
    numpy.random 모듈은 파이썬 내장 random 함수를 보강하여 다양한 종류의 확률분포로부터 효과적으로 표본 값을 생성하는 데
    주로 사용된다.
    예를 들어 normal 을 사용하여 표준정규분포로부터 4x4 크기의 표본을 생성할 수 있다.
'''
samples = np.random.normal(size=(4, 4))
samples
'''
    이에 비해 파이썬 내장 random 모듈은 한 번에 하나의 값만 생성할 수 있다.
    아래 성능 비교에서 알 수 있듯이 numpy.random 은 매우 큰 표본을 생성하는데 파이썬 내장 모듈보다 수십 배 이상 빠르다.
'''
from random import normalvariate
N = 1000000
# %timeit samples = [normalvariate(0, 1) for _ in range(N)]
# %timeit np.random.normal(size=N)

#todo 4.7 계단 오르내리기 예제
'''
    계단의 중간에서 같은 확률로 한 계단 올라가거나 내려간다고 가정하자.
    순수 파이썬으로 내장 random 모듈을 사용하여 계단 오르내리기를 1,000 번 수행하는 코드는 다음처럼 작성할 수 있다.
'''
import random
position = 0
walk = [position]
steps = 1000
for i in range(steps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)
'''
    np.random 모듈을 사용해서 1,000 번 수행한 결과 (1, -1) 를 한 번에 저장하고 누적 합을 계산한다.
'''
nsteps = 1000
draws = np.random.randint(0, 2, size=nsteps)
steps = np.where(draws > 0, 1, -1)
walk = steps.cumsum()
'''
    계단을 오르내린 위치의 최소/최대 값 같은 간단한 통계를 구할 수 있다.
'''
walk.min()
walk.max()
'''
    계단에서 특정 위치에 도달하기까지의 시간 같은 좀 더 복잡한 통계를 구할 수 있는데, 계단의 처음 위치에서 최초로 10 칸
    떨어지기까지 얼마나 걸렸는지 확인해보자.
    np.abs(walk) >= 10 을 통해 처음 위치에서 10 칸 이상 떨어진 시점을 알려주는 불리언 배열을 얻을 수 있다.
    우리는 최초의 10 혹은 -10 인 시점을 구해야 하므로 불리언 배열에서 최대 값의 처음 색인을 반환하는 argmax 를 사용하자.
'''
(np.abs(walk) > 10).argmax()
'''
    여기서 argmax 를 사용하긴 했지만 argmax 는 배열 전체를 모두 확인하기 때문에 효과적인 방법은 아니다.
    또한 이 예제에서는 True 가 최대 값임을 이미 알고 있었다.
'''

#todo 4.7.1 한 번에 계단 오르내리기 시뮬레이션하기
'''
    numpy.random 함수에 크기가 2 인 튜플을 넘기면 2 차원 배열이 생성되고 각 칼럼에서 누적 합을 구해 5,000 회의 시뮬레이션을
    한 번에 처리할 수 있다.
'''
nwalks = 5000
nsteps = 1000
draws = np.random.randint(0, 2, size=(nwalks, nsteps))
steps = np.where(draws > 0, 1, -1)
walks = steps.cumsum(axis=1)
walks
'''
    모든 시뮬레이션에 대해 최대 값과 최소 값을 구해보자.
'''
walks.max(axis=1)
walks.min(axis=1)
'''
    이 데이터에서 누적 합이 30 혹은 -30 이 되는 최소 시점을 계산해보자.
    5,000 회의 시뮬레이션 중 모든 경우가 30 에 도달하지 않아 계산이 약간 까다롭긴 하지만 any 메서드를 사용해서 해결할 수 있다.
'''
hits30 = (np.abs(walks) >= 30).any(1)
hits30
hits30.sum()
'''
    이 불리언 배열을 사용해 walks 에서 칼럼을 선택하고 절대 값이 30 이 넘는 경우에 대해 축 1 의 argmax 값을 구하면 처음
    위치에서 30 칸 이상 멀어지는 최소 횟수를 구할 수 있다.
'''
crossing_times = (np.abs(walks) >= 30).argmax(1)
crossing_times.mean()
'''
    다른 분포를 사용해서도 여러 가지 시도를 해보자.
    normal 함수에 표준편차와 평균 값을 넣어 정규분포에서 표본을 추출하는 것처럼 그냥 다른 난수 발생 함수를 사용하기만 하면 된다.
'''
steps = np.random.normal(loc=0, scale=0.25, size=(nwalks, nsteps))