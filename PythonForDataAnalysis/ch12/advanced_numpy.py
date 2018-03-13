#todo 12.1 ndarray 객체 내부 알아보기
'''
    ndarray 는 내부적으로 다음을 포함하고 있다.
     - 데이터에 대한 포인터. 시스템 메모리를 가리킨다.
     - 자료형 또는 dtype.
     - 배열의 형태를 나타내는 튜플. 예를 들어 10x5 짜리 배열은 (10, 5) 와 같은 형태를 취한다.
'''
import numpy as np
np.ones((10, 5)).shape
'''
    하나의 원소에서 다음 원소까지의 너비를 표현한 정수를 담고 있는 스트라이드 튜플.
    예를 들어 일반적인 3x4x5 크기의 float64(8바이트) 배열의 스트라이드 값은 (160, 40, 8) 이다.
'''
np.ones((3, 4, 5), dtype=np.float64).strides

#todo 12.1.1 NumPy dtype 구조
'''
    가끔 배열의 원소가 정수나 실수, 문자열, 파이썬 객체인지 검사해야 하는 코드를 작성할 일이 생긴다.
    왜냐하면 실수는 여러 가지 자료형(float16 부터 float128 까지)으로 표현되므로 자료형의 목록에서 dtype 을 검사하는 일은
    매우 다양하기 때문이다.
    다행히도 dtype 은 np.integer, np.floating 같은 상위 클래스를 가지므로 np.issubdtype 함수와 함께 사용할 수 있다.
'''
ints = np.ones(10, dtype=np.uint16)
floats = np.ones(10, dtype=np.float32)
np.issubdtype(ints.dtype, np.integer)
np.issubdtype(floats.dtype, np.floating)
'''
    각 자료형의 mro 메서드를 이용해 특정 dtype 의 모든 부모 클래스 목록을 확인할 수 있다.
'''
np.float64.mro()

#todo 12.2 고급 배열 조작 기법
'''
    배열을 세련된 방법으로 색인하고 나누고 불리언으로 값을 지정하는 방법은 다양하다.
    데이터 분석 애플리케이션에서 까다로운 작업의 대부분은 pandas 의 상위 레벨 함수에서 처리하지만 라이브러리에 존재하지
    않는 데이터 알고리즘을 직접 작성해야 하는 경우도 있다.
'''

#todo 12.2.1 배열 재형성하기
'''
    NumPy 배열에 대해 지금까지 배운 것을 이용해 배열의 데이터를 복사하지 않고 다른 모양으로 변환할 수 있다는 점은 다소
    놀라운 점이다.
    배열의 모양을 변환하려면 배열의 인스턴스 메서드인 reshape 메서드에 새로운 모양을 나타내는 튜플을 넘기면 된다.
    예를 들어 1 차원 배열을 행렬로 바꾼다고 가정해보자.
'''
arr = np.arange(8)
arr
arr.reshape((4, 2))
'''
    다차원 배열 또한 재형성이 가능하다.
'''
arr.reshape((4, 2)).reshape((2, 4))
'''
    reshape 에 넘기는 값 중 하나는 -1 이 될 수도 있는데, 이 경우에는 원본 데이터를 참조해서 적절한 값을 추론하게 된다.
'''
arr = np.arange(15)
arr.reshape((5, -1))
'''
    배열의 shape 속성은 튜플이기 때문에 reshape 메서드에 이를 직접 넘기는 것도 가능하다.
'''
other_arr = np.ones((3, 5))
other_arr.shape
arr.reshape(other_arr.shape)
'''
    다차원 배열을 낮은 차원으로 변환하는 것을 평탄화라고 한다.
'''
arr = np.arange(15).reshape((5, 3))
arr
arr.ravel()
'''
    ravel 메서드는 필요하지 않다면 원본 데이터의 복사본을 생성하지 않는다.
    flatten 메서드는 ravel 메서드와 유사하게 동작하지만 항상 데이터의 복사본을 반환한다.
'''
arr.flatten()

#todo 12.2.2 C 와 포트란 순서
'''
    R 과 MATLAB 같은 다른 과학계산 환경과는 달리 NumPy 는 메모리 상의 데이터 배치에 대한 유연하고 다양한 제어 기능을
    제공한다.
    기본적으로 NumPy 배열은 로우 우선순위로 생성된다.
    이 말은 만약 2 차원 배열이 있다면 배열의 각 로우에 해당하는 데이터들은 공간적으로 인접한 메모리에 적재된다는 뜻이다.
    로우 우선순위가 아니면 칼럼 우선순위를 가지게 되는데, 각 칼럼에 담긴 데이터들이 인접한 메모리에 적재된다.
    로우, 칼럼 우선순위는 각각 C 순서, 포트란 순서로 알려져 있다.
    고전 프로그래밍 언어인 포트란 77 의 경우 배열은 칼럼 우선순위로 저장된다.
    reshape 나 ravel 같은 함수는 배열에서 데이터 순서를 나타내는 인자를 받는다.
    이 값은 대부분의 경우 'C' 아니면 'F' 인데 아주 드물게 'A' 나 'K' 를 쓰기도 한다.
'''
arr = np.arange(12).reshape((3, 4))
arr
arr.ravel()
arr.ravel('F')
'''
    C 와 포트란 순서의 핵심적인 차이는 어느 차원부터 처리하느냐이다.
    - C / 로우 우선순위: 상위 차원을 먼저 탐색한다.
    - 포트란 / 칼럼 우선순위: 상위 차원을 나중에 탐색한다.
'''

#todo 12.2.3 배열 이어붙이고 나누기
'''
    numpy.concatenate 는 배열의 목록(튜플, 리스트 등)을 받아서 주어진 axis 에 따라 하나의 배열로 합쳐준다.
'''
arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[7, 8, 9], [10, 11, 12]])
np.concatenate([arr1, arr2], axis=0)
np.concatenate([arr1, arr2], axis=1)

concat1 = np.concatenate([arr1, arr2], axis=0)
concat1.reshape((6, 2), order='C')  # C Order
concat1.reshape((6, 2), order='F')  # Fortran Order
'''
    vstack 과 hstack 함수를 이용하면 일반적인 이어붙이기 작업을 쉽게 처리할 수 있다.
    이 연산은 vstack 과 hstack 메서드를 사용해서 다음처럼 표현할 수 있다.
'''
np.vstack((arr1, arr2))  # (2, 3) + (2, 3) => (4, 3)
np.hstack((arr1, arr2))  # (2, 3) + (2, 3) => (2, 6)

arr1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
arr2 = np.array([[[11, 12], [13, 14]], [[15, 16], [17, 18]]])
np.vstack((arr1, arr2))  # (2, 2, 2) + (2, 2, 2) => (4, 2, 2)
np.hstack((arr1, arr2))  # (2, 2, 2) + (2, 2, 2) => (2, 4, 2), np.concatenate([arr1, arr2], axis=1) 과 동일
'''
    다른 한편으로는 split 메서드를 사용해서 하나의 배열을 축을 따라 여러 개의 배열로 나눌 수 있다.
'''
from numpy.random import randn
arr = randn(5, 2)
arr
first, second, third = np.split(arr, [1, 3])
first
second
third

#todo 배열 쌓기 도우미: r_ 와 c_
'''
    Numpy 의 네임스페이스에는 r_ 와 c_ 라는 두 가지 특수한 객체가 있는데, 이는 배열 쌓기를 좀 더 편리하게 해준다.
'''
arr = np.arange(6)
arr1 = arr.reshape((3, 2))
arr2 = randn(3, 2)
np.r_[arr1, arr2]  # vstack, concatenate(axis=0) 메서드와 동일
np.c_[arr1, arr2]  # hstackm concatenate(axis=1) 메서드와 동일
'''
    뿐만 아니라 슬라이스를 배열로 변환해준다.
'''
np.c_[1:6, -10:-5]

#todo 12.2.4 원소 반복시키기: repeat 과 tile
'''
    큰 배열을 만들기 위해 배열을 반복하거나 복제하는 함수로 repeat 과 tile 이 있다.
    repeat 은 한 배열의 각 원소를 원하는 만큼 복제해서 큰 배열을 생성한다.
'''
arr = np.arange(3)
arr.repeat(3)
'''
    기본적으로 정수를 넘기면 각 배열은 그 수만큼 반복된다.
    만약 정수의 배열을 넘긴다면 각 원소는 배열에 담긴 정수만큼 다르게 반복될 것이다.
'''
arr.repeat([2, 3, 4])
'''
    다차원 배열의 경우에는 특정 축을 따라 각 원소를 반복시킨다.
'''
arr = randn(2, 2)
arr
arr.repeat(2, axis=0)
'''
    다차원 배열에서 만약 axis 인자를 넘기지 않으면 배열은 평탄화되므로 주의해야 한다.
    repeat 메서드에 정수의 배열을 넘기면 축을 따라 배열에서 지정한 횟수만큼 원소가 반복된다.
'''
arr.repeat([2, 3], axis=0)
arr.repeat([2, 3], axis=1)
'''
    tile 메서드는 축을 따라 배열을 복사해서 쌓는 함수다.
    벽에 타일을 이어붙이듯이 같은 내용의 배열을 이어붙인다고 생각하면 된다.
'''
arr
np.tile(arr, 2)
'''
    tile 메서드의 두 번째 인자는 타일의 개수로, 스칼라 값이며 칼럼 대 칼럼이 아니라 로우 대 로우로 이어붙이게 된다.
    tile 메서드의 두 번째 인자는 타일을 이어붙인 모양을 나타내는 튜플이 될 수 있다.
'''
arr
np.tile(arr, (2, 1))
np.tile(arr, (3, 2))  # axis = 0 방향으로 3번, axis = 1 방향으로 2번 복사한다.

#todo 12.2.5 팬시 색인: take 와 put
arr = np.arange(10) * 100
inds = [7, 1, 2, 6]
arr[inds]
'''
    ndarray 에는 단일 축에 대한 값을 선택할 때만 사용할 수 있는 유용한 메서드가 존재한다.
'''
arr.take(inds)  # arr[inds] 와 같은 동작 수행
arr.put(inds, 42)  # arr[inds] = 42 와 같은 동작 수행
arr
arr.put(inds, [40, 41, 42, 43])
arr
'''
    다른 축에 take 메서드를 적용하려면 axis 인자를 넘기면 된다.
'''
inds = [2, 0, 2, 1]
arr = randn(2, 4)
arr
arr.take(inds, axis=1)
'''
    put 메서드는 axis 인자를 받지 않고 평탄화된 배열(1차원, C 순서)에 대한 색인을 받으므로(변경될 가능성이 있다) 다른 축에
    대한 색인 배열을 사용해서 배열의 원소에 값을 넣으려면 팬시 색인을 이용하는 것이 좋다.
    이 책을 집필하는 현재, take 와 put 함수는 일반적으로 동일한 팬시 색인을 사용하는 것보다 뚜렷한 성능상의 이점을 갖고 있다.
    나는 이 차이가 NumPy 의 버그에서 기인한 것이며 언젠가는 수정될 것이라 생각하지만 정수 배열을 사용해서 큰 배열의 일부를
    선택해야 할 경우에 대비해 알아두는 것도 나쁘지 않을 것이다. 
'''
arr = randn(1000, 50)
inds = np.random.permutation(1000)[:500]
%timeit arr[inds]  # 팬시 색인
# 15.8 µs ± 84.1 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
%timeit arr.take(inds, axis=0)
# 13.6 µs ± 321 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

#todo 12.3 브로드캐스팅
'''
    브로드캐스팅은 다른 모양의 배열 간 산술연산을 어떻게 수행해야 하는지를 설명한다.
    이는 매우 강력한 기능이지만 NumPy 의 오랜 사용자들도 흔히 잘못 이해하고 있는 기능이기도 하다.
    브로드캐스팅의 가장 단순한 예제는 배열과 스칼라 값을 결합할 때 발생한다.
'''
arr = np.arange(5)
arr
arr * 4
'''
    여기서 스칼라 값 4 는 곱셈연산 과정에서 배열의 모든 원소로 전파(broadcast) 되었다.
    예를 들어 배열의 각 칼럼에서 칼럼의 평균 값을 뺀다면 다음처럼 간단히 처리할 수 있다.
'''
arr = randn(4, 3)
arr.mean(0)
demeaned = arr - arr.mean(0)
demeaned
demeaned.mean(0)
'''
    브로드캐스팅 규칙
    - 만일 이어지는 각 차원(시작부터 끝까지)에 대해 축의 길이가 일치하거나 둘 중 하나의 길이가 1이라면 두 배열은 브로드캐스팅
      호환이다. 브로드캐스팅은 누락된, 혹은 길이가 1인 차원에 대해 수행된다.
'''
'''
    아무리 NumPy 에 익숙한 사용자일지라도 종종 멈춰서 그림을 그리고 브로드캐스팅 규칙에 대해 생각해봐야 한다.
    이전 예제에서 칼럼이 아니라 각 로우에서 평균 값을 뺀다고 가정해보자.
    arr.mean(0) 은 길이가 3이고 arr 의 이어지는 크기 역시 3 이므로 0 번 축에 대해 브로드캐스팅이 가능하다.
    브로드캐스팅 규칙에 따르면 1번 축에 대해 뺄셈을 하려면(각 로우에서 로우 평균 값을 빼려면) 작은 크기의 배열은 (4, 1)의
    크기가 되어야 한다.
'''
arr
row_means = arr.mean(1)
row_means.reshape((4, 1))
demeaned = arr - row_means.reshape((4, 1))
demeaned.mean(1)

#todo 12.3.1 다른 축에 대해 브로드캐스팅하기
'''
    낮은 차원의 배열로 0번 축이 아닌 다른 축에 대해 산술연산을 수행하는 일은 흔히 있을 수 있는 일이다.
    브로드캐스팅 규칙을 따르자면 전파되는 차원은 작은 배열에서는 반드시 1이어야 한다.
    로우에서 평균 값을 빼는 앞의 예제에서 로우 평균 값은 (4, )가 아니라 (4, 1)로 재형성한다는 의미다.
'''
arr - arr.mean(1).reshape((4 ,1))
'''
    3 차원의 경우 세 가지 차원 중 어느 하나에 대한 브로드캐스팅은 호환되는 모양으로 데이터를 재형성하면 된다.
    따라서 아주 일반적인 문제는 브로드캐스팅 전용 목적으로 길이가 1 인 새로운 축을 추가해야 한다는 것이다.
    reshape 를 사용하는 것도 한 방법이지만 축을 하나 새로 추가하는 것은 새로운 모양을 나타낼 튜플을 하나 생성해야 하는데,
    이는 꽤 지루한 방법이므로 NumPy 배열은 색인을 통해 새로운 축을 추가하는 특수한 문법을 제공한다.
    np.newaxis 라는 이 특수한 속성을 배열의 전체 슬라이스와 함께 사용해 새로운 축을 추가할 수 있다.
'''
arr = np.zeros((4, 4))
arr_3d = arr[:, np.newaxis, :]
arr_3d.shape
arr_1d = np.random.normal(size=3)
arr_1d[:, np.newaxis]
arr_1d[np.newaxis, :]
'''
    이렇게 해서 만약 3차원 배열에서 2번 축에 대해 평균 값을 빼고 싶다면 다음과 같이 작성하기만 하면 된다.
'''
arr = randn(3, 4 ,5)
demeaned = arr - arr.mean(axis=2)[:, :, np.newaxis]
demeaned.mean(2)
'''
    성능을 떨어뜨리지 않으면서 한 축에 대해 평균 값을 빼는 과정을 일반화 하는 함수
'''
def demean_axis(arr, axis=0):
    means = arr.mean(axis)
    indexer = [slice(None)] * arr.ndim  # [slice(None, None, None), slice(None, None, None), slice(None, None, None)]
    indexer[axis] = np.newaxis  # [slice(None, None, None), slice(None, None, None), None]
    return arr - means[indexer]

demean_axis(arr, 2)

#todo 12.3.2 브로드캐스팅 이용해 배열에 값 대입하기
'''
    배열의 색인을 통해 값을 대입할 때도 산술연산에서의 브로드캐스팅 규칙이 적용된다.
    간단하게는 다음과 같이 할 수 있을 것이다.
'''
arr = np.zeros((4, 3))
arr[:] = 5
arr
'''
    하지만 만약 값이 담긴 1차원 배열이 있고, 그 배열의 칼럼에 값을 대입하고 싶다면 배열의 모양이 호환되는 한 그렇게 하는
    것도 가능하다.
'''
col = np.array([1.28, -0.42, 0.44, 1.6])
arr[:] = col[:, np.newaxis]
arr
arr[:2] = [[-1.37], [0.509]]
arr

#todo 12.4 고급 ufunc 사용법
'''
    많은 NumPy 사용자들은 유니버설 함수로 제공되는 빠른 원소별 연산만 주로 사용하는 경향이 있는데 이런 경우 반복문을 작성하지
    않고 좀 더 간결한 코드를 작성할 수 있는 다양한 부가적인 기능이 있다.
'''

#todo 12.4.1 ufunc 인스턴스 메서드
'''
    reduce 는 하나의 배열을 받아 순차적인 이항 연산을 통해 축에 따라 그 값을 집계해준다.
    예를 들어 배열의 모든 원소를 더하는 방법으로 np.add.reduce 를 사용할 수 있다.
'''
arr = np.arange(10)
np.add.reduce(arr)  # np.add.reduce == np.sum
arr.sum()
'''
    시작 값(add 에서는 0)은 ufunc 에 의존적이다.
    만약 axis 인자가 넘어오면 reduce 는 그 축을 따라 수행된다.
    이를 통해 축약된 방법으로 어떤 질문에 대한 답을 구할 수 있을 것이다.
    약간 복잡한 예제이긴 하지만 np.logical_and 를 사용해서 배열의 각 로우에 있는 값이 정렬된 상태인지 검사하는 것을
    생각해볼 수 있다.
'''
arr = randn(5, 5)
arr[::2].sort(1)
arr[:, :-1] < arr[:, 1:]
np.logical_and.reduce(arr[:, :-1] < arr[:, 1:], axis=1)  # np.logical_and.reduce == np.all
np.all(arr[:, :-1] < arr[:, 1:], axis=1)
'''
    accumulate 는 cumsum 메서드나 sum 메서드와 마찬가지로 reduce 메서드와 관련이 있다.
    accumulate 메서드는 누계를 담고 있는 같은 크기의 배열을 생성한다.
'''
arr = np.arange(15).reshape((3, 5))
np.add.accumulate(arr, axis=1)  # np.add.accumulate == np.cumsum
np.cumsum(arr, axis=1)
'''
    ▶ 벡터의 내적
     - 두 벡터의 각 성분을 곱한 후 더해서(스칼라곱) 나오는 스칼라 값
     - 두 벡터 사이의 각을 구하기 위해 사용 가능
    ▶ 벡터의 외적
     - 내적과 다른 방식의 두 벡터의 곱(벡터곱)으로 두 벡터와 직교하는 또 다른 벡터
     - 회전할 물체의 회전축을 찾는데 사용 가능
    - outer 메서드는 두 배열 간의 벡터 곱, 외적을 계산한다. (평행사변형의 넓이와 같다. A x B = |A|*|B|*sinθ*n
'''
arr = np.arange(3).repeat([1, 2, 2])
arr
np.multiply.outer(arr, np.arange(5))
'''
    outer 메서드 결과의 차원은 입력된 차원의 합이 된다.
'''
result = np.subtract.outer(randn(3, 4), randn(5))
result.shape
'''
    마지막으로 reduceat 메서드는 로컬 reduce 를 수행하는데, 본질적으로 배열의 groupby 연산으로 배열의 슬라이스를 모두 함께
    집계한 것이다.
    pandas 의 GroupBy 기능보다 유연하지는 못하지만 적절한 상황에서 사용한다면 매우 빠르고 강력한 메서드다.
    reduceat 메서드는 값을 어떻게 나누고 집계할지 나타내는 경계 목록을 인자로 받는다.
'''
arr = np.arange(10)
np.add.reduceat(arr, [0, 5 ,8])  # arr[0:5], arr[5:8], arr[8:] 에 대한 수행 결과.(합)
arr = np.multiply.outer(np.arange(4), np.arange(5))
arr
np.add.reduceat(arr, [0, 2, 4], axis=1)
np.add.reduceat(arr, [0, 2, 3], axis=0)

#todo 12.4.2 사용자 ufunc
'''
    ufunc 와 유사한 사용자 함수를 만들 수 있는 몇 가지 기능이 있다.
    numpy.frompyfunc 는 입력과 출력에 대한 표준과 함께 파이썬 함수를 인자로 취한다.
    예를 들어 원소별로 합을 구하는 함수는 다음과 같이 작성할 수 있다.
'''
def add_elements(x, y):
    return x + y

add_them = np.frompyfunc(add_elements, 2, 1)
add_them(np.arange(8), np.arange(8))
'''
    frompyfunc 를 이용해서 생성한 함수는 항상 파이썬 객체가 담긴 배열을 반환하는데, 이는 그다지 유용하지 않다.
    다행스럽게도 대안이 있는데, numpy.vectorize 을 사용하면 자료형을 추론한다는 이점이 있다.
    하지만 이 함수는 기능이 조금 미흡하다.
'''
add_them = np.vectorize(add_elements, otypes=[np.float64])
add_them(np.arange(8), np.arange(8))
'''
    이 두 함수는 ufunc 스타일의 함수 만드는 방법을 제공하지만 각 원소를 계산하기 위해 파이썬 함수를 호출하게 되므로
    NumPy 의 C 기반 ufunc 반복문보다 많이 느리다.
'''
arr = randn(10000)

%timeit add_them(arr, arr)
%timeit np.add(arr, arr)

#todo 12.5 구조화된 배열과 레코드 배열
'''
    지금쯤은 ndarray 가 단일 데이터 저장소라는 사실을 눈치챘을 것이다.
    이 말은 각 원소는 dtype 에 의해 결정된, 같은 크기의 메모리를 차지하고 있다는 뜻이다.
    표면적으로는 다중 데이터나 표 형식의 데이터를 표현할 수 없는 것처럼 보인다.
    구조화된 배열은 배열의 각 원소가 C 의 구조체 혹은 다양한 이름의 필드를 갖는 SQL 테이블의 한 로우라고 생각할 수 있는 
    ndarray 이다.
'''
dtype = [('x', np.float64), ('y', np.int32)]
sarr = np.array([(1.5, 6), (np.pi, -2)], dtype=dtype)
sarr
'''
    구조화된 dtype 을 지정하는 방법은 여러 가지다.
    일반적인 방법 중 하나는 튜플(field_name, field_data_type)을 이용하는 방법이다.
    이제 sarr 의 각 원소는 사전처럼 접근할 수 있는 튜플 같은 객체다.
'''
sarr[0]
sarr[0]['y']
'''
    필드 이름은 dtype.names 속성에 저장된다.
    구조화된 배열의 필드에 접근하면 데이터의 뷰가 반환되며 따라서 아무것도 복사되지 않는다.
'''
sarr['x']

#todo 12.5.1 중첩된 dtype 과 다차원 필드
'''
    구조화된 dtype 을 지정할 때 추가적으로 그 모양(정수나 튜플)을 전달할 수 있다.
'''
dtype = [('x', np.int64, 3), ('y', np.int32)]
arr = np.zeros(4, dtype=dtype)
arr
'''
    이 경우 x 필드는 각 원소에 대해 길이가 3 인 배열을 참조하게 된다.
'''
arr[0]['x']
'''
    편리하게도 arr['x']로 접근하면 이전 예제에서처럼 1차원 배열 대신 2차원 배열이 반환된다.
'''
arr['x']
'''
    이를 통해 좀 더 복잡한 중첩 구조를 하나의 배열 안에서 단일 메모리로 표현할 수 있게 된다.
    dtype 을 무한히 복잡하게 만들 수 있으며, 중첩된 dtype 도 만들 수 있다.
    다음의 간단한 예제를 살펴보자.
'''
dtype = [('x', [('a', 'f8'), ('b', 'f4')]), ('y', np.int32)]
data = np.array([((1, 2), 5), ((3, 4), 6)], dtype=dtype)
data['x']
data['y']
data['x']['a']
'''
    위에서 확인할 수 있듯이 다양한 형태의 필드와 중첩된 레코드는 특정 상황에서는 매우 강력한 기능을 발휘할 수 있다.
    이와 대조적으로 pandas 의 DataFrame 의 계층적 색인이 이와 유사하긴 하지만 이런 기능을 직접 지원하지는 않는다.
'''

#todo 12.5.2 구조화된 배열을 사용해야 하는 이유
'''
    pandas 의 DataFrame 과 비교해보면 NumPy 의 구조화된 배열은 상대적으로 저수준의 도구이며, 메모리 블록을 복잡하게 중첩된
    칼럼이 있는 표 형식처럼 해석할 수 있는 방법을 제공한다.
    배열의 각 원소는 메모리 상에서 고정된 크기의 바이트로 표현되기 때문에 구조화된 배열은 데이터를 디스크에서 읽거나 쓰고
    네트워크를 통해 전송할 때 매우 빠르고 효과적인 방법을 제공한다.
    구조화된 배열의 또 다른 사용 방법으로 데이터 파일을 고정된 크기의 레코드 바이트 스트림으로 기록하는 것은 C 나 C++ 코드에서
    데이터를 직렬화하는 일반적인 방법이다.
    파일의 포맷을 알고 있다면(즉, 각 레코드의 크기와 순서, 바이트 크기 그리고 각 원소의 자료형을 알고 있다면) np.fromfile
    을 사용해서 데이터를 메모리로 읽어 들일 수 있다.
    이와 같은 특수한 사용법은 이 책에서 다르는 내용의 범주를 벗어나지만 그런 방법이 가능하다는 것을 알아두는 것도 가치가
    있는 일이다.
'''

#todo 12.5.3 구조화된 배열 다루기: numpy.lib.recfunctions
'''
    구조화된 배열은 DataFrame 만큼 많은 함수를 가지고 있지 않지만 NumPy 모듈인 numpy.librecfunctions 에 필드를 추가, 삭제하거나
    기본적인 조인과 유사한 연산을 수행할 수 있는 몇 가지 유용한 도구가 있다.
    이 도구에 대해 기억해야 할 것은 dtype 에 어떤 변경(칼럼을 추가, 삭제하는 등)을 가하기 위해서는 일반적으로 새로운 배열을
    생성해야 한다는 것이다.
    numpy.lib.recfunctions 에 대해서는 관심이 많은 독자들이 직접 찾아보도록 이 책에서는 사용하지 않는다.
'''

#todo 12.6 정렬에 관하여
'''
    파이썬의 내장 리스트와 마찬가지로 ndarray 의 sort 인스턴스 메서드는 새로운 배열을 생성하지 않고 직접 해당 배열의 내용을
    정렬한다.
'''
arr = randn(6)
arr.sort()
arr
'''
    배열을 그대로 정렬할 때는 그 배열이 다른 ndarray 의 뷰일 경우 원본 배열의 값이 변경된다는 점을 꼭 기억하자.
'''
arr = randn(3, 5)
arr
arr[:, 0].sort()
arr
'''
    다른 한편으로는 numpy.sort 를 사용해 정렬된 배열의 복사본을 생성할 수 있다.
    그 외에는 ndarray.sort 와 같은 인자를 받는다.
'''
arr = randn(5)
arr
np.sort(arr)
arr
'''
    여기서 소개한 모든 정렬 메서드는 전달된 축에 독립적으로 정렬을 수행하기 위해 axis 인자를 받는다.
'''
arr = randn(3, 5)
arr
arr.sort(axis=1)
arr
'''
    어떤 정렬 메서드도 내림차순 정렬을 위한 옵션이 없다는 걸 알 수 있다.
    배열의 슬라이스는 복사본을 만들거나 어떠한 연산도 수행하지 않고 그저 뷰를 생성하기 때문에 이는 사실 큰 문제가 되지 않는다.
    많은 파이썬 사용자들은 values[::-1]을 통해 순서가 뒤집어진 리스트를 얻어내는 트릭에 익숙하다.
    ndarray 에서도 마찬가지로 사용할 수 있다.
'''
arr[:, ::-1]

#todo 12.6.1 간접 정렬: argsort 와 lexsort
'''
    데이터 분석에서 하나 이상의 키를 기준으로 데이터를 정렬하는 일은 아주 흔한 일이다.
    예를 들어 학생 데이터는 성(first name)으로 정렬한 후에 다시 이름으로 정렬할 필요가 있다.
    이는 간접 정렬의 한 예로, pandas 와 관련한 내용에서 이미 다양한 고수준의 예제를 많이 다뤘다.
    주어진 단일 키 혹은 여러 개의 키(배열이나 여러 개의 값)를 가지고 데이터를 정렬하려면 어떤 순서로 나열해야 하는지
    알려주는 정수 색인이 담긴 배열을 얻고 싶을 때가 있다.
    이를 위한 두 가지 메서드가 있는데, 바로 argsort 와 numpy.lexsort 다.
'''
values = np.array([5, 0, 1, 3, 2])
indexer = values.argsort()
indexer
values[indexer]
'''
    다음은 좀 더 복잡한 예제로, 2차원 배열을 첫 번째 로우 순서대로 정렬하는 코드다.
'''
arr = randn(3, 5)
arr[0] = values
arr
arr[:, arr[0].argsort()]
'''
    lexsort 는 argsort 와 유사하지만 다중 키 배열에 대해 간접 사전 순으로 정렬을 한다.
    성과 이름으로 구분되는 아래 데이터를 정렬한다고 가정하자.
'''
first_name = np.array(['Bob', 'Jane', 'Steve', 'Bill', 'Barbara'])
last_name = np.array(['Jones', 'Arnold', 'Arnold', 'Jones', 'Walters'])
sorter = np.lexsort((first_name, last_name))
list(zip(last_name[sorter], first_name[sorter]))
'''
    lexsort 를 처음 접하는 경우라면 약간 혼란스러울 수도 있을 것이다.
    왜냐하면 나중에 넘겨준 배열로 먼저 데이터를 정렬하는 데 사용되기 때문이다.
    위 데이터에서 확인할 수 있듯이 last_name 이 first_name 보다 먼저 정렬된다.
'''

#todo 12.6.2 다른 정렬 알고리즘
'''
    견고한 정렬 알고리즘은 동일한 원소의 상대적인 위치를 그대로 둔다.
    이는 상대적인 순서가 의미를 가지는 간접 정렬의 경우 특히 중요한 기능이다.
'''
values = np.array(['2:first', '2:second', '1:first', '1:second', '1:third'])
key = np.array([2, 2, 1, 1, 1])
indexer = key.argsort(kind='mergesort')
indexer
values.take(indexer)

#todo 12.6.3 numpy.searchsorted: 정렬된 배열에서 원소 찾기
'''
    searchsorted 는 정렬된 배열상에서 이진 탐색을 수행해 새로운 값을 삽입할 때 정렬된 상태를 계속 유지하기 위해 위치를
    반환하는 메서드다.
'''
arr = np.array([0, 1, 7, 12, 15])
arr.searchsorted(9)
arr
'''
    값이 담긴 배열을 넘기면 해당 원소별로 알맞은 위치를 담고 있는 배열을 반환한다.
'''
arr.searchsorted([0, 8, 11, 16])
'''
    searchsorted 메서드가 0번째 원소에 대해 0을 반환한 것을 확인할 수 있다.
'''
arr = np.array([0, 0, 0, 1, 1, 1, 1])
arr.searchsorted([0, 1])
arr.searchsorted([0, 1], side='right')
'''
    searchsorted 의 다른 활용법으로 0 부터 10,000 까지의 값을 특정 구간별로 나눈 배열을 살펴보자.
'''
data = np.floor(np.random.uniform(0, 10000, size=50))
bins = np.array([0, 100, 1000, 5000, 10000])
data
'''
    그리고 각 데이터가 어떤 구간에 속해야 하는지 알아보기 위해 searchsorted 메서드를 사용하자.
'''
labels = bins.searchsorted(data)
labels
'''
    이를 pandas 의 groupby 와 조합하면 쉽게 해당 구간의 데이터를 구할 수 있다.
'''
import pandas
from pandas import Series
Series(data).groupby(labels).mean()
'''
    사실 NumPy 에는 위 과정을 계산해주는 digitize 함수가 있다.
'''
np.digitize(data, bins)

#todo 12.7 NumPy matrix 클래스
'''
    MATLAB 이나 Julia, GAUSS 같은 다른 행렬 연산과 선형대수를 위한 언어를 비교하면 NumPy 의 선형대수 문법은 좀 장황한 편이다.
    이렇게 말할 수 있는 한 가지 이유는 NumPy 에서 행렬 곱은 numpy.dot 을 이용해야 하기 때문이다.
    또한 NumPy 의 색인 구문에 차이가 있기 때문에 파이썬으로 코드를 포팅하는 작업은 때로는 덜 직관적일 수 있다.
    2차원 배열에서 하나의 로우를 선택하거나 (X[1, :[) 칼럼을 선택하거나 (X[:, 1]) 되면 MATLAB 처럼 2차원 배열이 아니라 1차원
    배열을 반환한다.
'''
X = np.random.randn(4, 4)
X[:, 0]  # 1차원
y = X[:, :1]  # 슬라이싱에 의한 2차원
X
y
'''
    이 경우 행렬 곱 y역행렬Xy 는 아래처럼 구할 수 있다.
'''
np.dot(y.T, np.dot(X, y))
'''
    NumPy 는 많은 행렬연산을 위한 코드 작성을 돕기 위해 MATLAB 과 유사하게 단일 로우와 칼럼을 2차원으로 반환하고 * 연산자를
    이용해서 행렬 곱셈을 수행하도록 색인 방법을 변경한 matrix 클래스를 제공한다.
    위에서 살펴본 연산은 numpy.matrix 를 사용해서 아래처럼 작성할 수 있다.
'''
Xm = np.matrix(X)
ym = Xm[:, 0]
Xm
ym
ym.T * Xm * ym
'''
    matrix 클래스는 역행렬을 반환하는 특수한 속성, I 를 가지고 있다.
'''
Xm.I * X
'''
    책의 저자는 일반적인 목적의 ndarray 를 numpy.matrix 로 대체해 사용하는 걸 추천하지 않는데, 그 이유는 좀처럼 그렇게
    사용하지 않기 때문이다.
    선형대수를 위한 많은 개별 함수에서는 전달받은 함수의 인자를 matrix 로 변환하고 데이터를 복사하지 않는 np.asarray 를
    사용해서 반환하기 전에 평범한 배열로 다시 변환하는 것이 좀 더 유용할 것이다.
'''

#todo 12.8 고급 배열 입출력
'''
    4장에서 np.save 와 np.load 를 사용해서 배열을 이진 형식으로 디스크에 저장하는 방법을 소개했는데, 여기에서는 이를 좀 더
    우아하게 사용할 수 있는 몇 가지 부가적인 옵션을 소개한다.
    특히 메모리 맵은 RAM 에 적재할 수 없는 데이터를 다룰 때 추가적인 이점을 얻을 수 있다.
'''

#todo 12.8.1 메모리 맵 파일
'''
    메모리 맵 파일은 디스크에 저장된 아주 큰 이진 데이터를 메모리에 적재된 배열처럼 취급할 수 있다.
    NumPy 에는 ndarray 와 유사한 memmap 객체가 있는데, 배열 전체를 메모리에 적재하지 않고 큰 파일의 작은 부분을 읽고 쓸 수
    있도록 해준다.
    게다가 memmap 객체는 메모리에 적재된 배열에서 제공하는 것과 동일한 메서드를 제공하기 때문에 ndarray 를 사용해야 하는
    많은 알고리즘에서 ndarray 의 대체자로 사용할 수 있다.
    새로운 memmap 객체를 생성하려면 np.memmap 함수에 파일 경로와 dtype, 모양 그리고 파일의 모드를 전달해야 한다.
'''
mmap = np.memmap('mymmap', dtype='float64', mode='w+', shape=(10000, 10000))
mmap
'''
    memmap 객체의 슬라이스는 디스크에 있는 데이터에 대한 뷰를 반환한다.
'''
section = mmap[:5]
section
'''
    여기에 데이터를 대입하면 파이썬의 파일 객체처럼 메모리에 잠시 보관되어 있다가 flush 를 호출하면 디스크에 기록하게 된다.
'''
section[:] = np.random.randn(5, 10000)
mmap.flush()
mmap
del mmap
'''
    메모리 맵은 스코프를 벗어나서 메모리가 회수되면 디스크에 변경 사항이 기록된다.
    기존의 메모리 맵 파일을 열 때 메타데이터 없이 디스크에 저장된 이진 데이터 파일처럼 dtype 과 모양을 지정할 수 있다.
'''
mmap = np.memmap('mymmap', dtype='float64', shape=(10000, 100000))  # mode='r+' 가 default
mmap
'''
    메모리 맵은 디스크 상의 ndarray 이므로 위에서 설명한 것처럼 구조화된 dtype 을 사용하는데도 아무런 문제가 없다.
'''

#todo 12.8.2 HDF5 및 기타 배열 저장 옵션
'''
    PyTables 와 h5py 는 효율적이며, HDF5 형식으로 압축할 수 있도록 배열 데이터를 저장할 수 있게 하는 NumPy 친화적인 인터페이스의
    파이썬 프로젝트다.
    또한 수백 기가 혹은 수 테라바이트의 데이터를 HDF5 형식으로 안전하게 저장할 수 있다.
    이 라이브러리의 사용 방법은 아쉽게도 이 책의 범위를 벗어난다.
    PyTables 는 구조화된 배열을 진보된 질의 기능 및 질의 속도를 높일 수 있도록 칼럼 색인을 추가하는 기능과 함께 사용할 수 있도록 한다.
    이는 관계형 데이터베이스에서 제공하는 테이블 색인 기능과 매우 유사하다.
'''

#todo 12.9 성능 팁
'''
    NumPy 를 활용하는 코드에서 좋은 성능을 이끌어내는 방법은 상당히 직관적인데, 순수 파이썬 반복문은 상대적으로 매우 느리므로
    일반적으로 배열연산으로 대체한다.
     - 파이썬 반복문과 조건문을 배열연산과 불리언 배열연산으로 변환한다.
     - 가능한 한 브로드캐스팅을 사용한다.
     - 배열의 뷰(슬라이스)를 사용해 데이터를 복사하는 것을 피한다.
     - ufunc 메서드를 활용한다.
    NumPy 만으로 원하는 성능을 이끌어내지 못한다면 코드를 C 나 포트란으로 작성하거나 아니면 Cython 을 사용해 성능을 향상시킬 수 있다.
    나는 개인적으로 Cython 을 자주 사용하는데 최소한의 개발 노력으로 쉽게 C 수준의 성능을 이끌어낼 수 있기 때문이다.
'''

#todo 12.9.1 인접 메모리의 중요성
'''
    이 주제에 대한 전체 내용은 이 책의 범위를 벗어나는데, 어떤 애플리케이션에서는 배열이 메모리 상에 배치된 모양에 따라
    연산 속도에 많은 영향을 끼친다.
    이는 부분적으로 CPU 의 캐시 구조에 의한 성능 차이에 기반하며 연속된 메모리에 접근하는 연산(예를 들어 C 순서로 저장된
    배열에서 로우를 합산하는)의 경우, 메모리 서브시스템이 적절한 메모리 블록을 매우 빠른 CPU 의 L1 이나 L2 에 저장하게 되므로
    가장 빠르다.
    또한 NumPy 의 C 코드 기반 내부의 어떤 코드는 연속된 메모리일 경우에 최적화되어 인접하지 않은 메모리를 읽게 되는 문제를
    피할 수 있다.
    배열이 메모리 상에 연속적으로 존재한다는 의미는 배열의 원소가 실제 배열 상에 나타나는 모습대로 메모리에 저장되었다는 의미다.
    기본적으로 NumPy 배열은 메모리에 C 순서 혹은 그냥 단순히 연속적으로 생성된다.
    C 순서로 저장된 배열의 전치 배열 같은 칼럼 우선순위 배열은 포트란 순서 배열이라고 할 수 있다.
    이 속성은 ndarray 의 flags 속성을 통해 명시적으로 확인할 수 있다.
'''
arr_c = np.ones((1000, 1000), order='C')
arr_f = np.ones((1000, 1000), order='F')
arr_c.flags
arr_f.flags
arr_f.flags.f_contiguous
'''
    예제에서 이 배열의 로우 합은 메모리에 로우가 연속적으로 존재하므로 이론적으로 arr_c 가 arr_f 보다 빠르게 계산된다.
    %timeit 을 사용해서 성능차를 확인해볼 수 있다.
'''
%timeit np.sum(arr_c, axis=1)
%timeit np.sum(arr_f, axis=1)
'''
    이는 NumPy 에서 성능을 더 이끌어내야 할 때 더 많은 노력을 기울이게 되는 부분이다.
    원하는 메모리 순서로 저장되지 않은 배열이 있다면 그 배열을 'C' 나 'F' 순서로 복사해서 사용할 수 있다.
'''
arr_f.copy(order='C').flags
'''
    한 배열에 대한 뷰를 생성할 때 그 결과가 항상 연속된 메모리에 할당되지 않을 수도 있다는 점을 기억하자.
'''
arr_c[:50].flags.contiguous
arr_c[:, :50].flags

#todo 12.9.2 기타 성능 옵션: Cython, f2py, C
'''
    최근 몇 년간 Cython 프로젝트는 많은 과학계산 파이썬 개발자가 C 나 C++ 라이브러리와 함께 작동할 필요가 있지만 순수 C 코드를
    작성할 필요 없이 빠른 코드를 구현하기 위한 도구로 선택되었다.
    Cython 은 정적 자료형과 C 로 작성된 코드를 파이썬 스타일의 코드에 끼워 넣을 수 있는 기능을 가진 파이썬이라고 생각하면 된다.
    예를 들어 1차원 배열에서 각 원소의 합을 구하는 간단한 Cython 함수는 아래와 같다.
'''
# from numpy cimport ndarray, float64_t
#
# def sum_elements(ndarray[float64_t] arr):
#     cdef Py_ssize_t i, n = len(arr)
#     cdef float64_t result = 0
#
#     for i in range(n):
#         result += arr[i]
#     return result
'''
    Cython 은 이 코드를 C 로 변환한 후에 생성된 C 코드를 컴파일해서 파이썬 확장을 생성한다.
    Cython 은 순수 파이썬 코드를 작성하는 시간에 아주 약간의 시간을 더 할애해서 NumPy 와 유기적으로 작동하면서도 더 나은
    성능을 얻을 수 있는 매력적인 옵션이다.
    일반적인 작업 흐름은 파이썬에서 동작하는 알고리즘으로 작성하고 그 알고리즘을 자료형 선언과 다른 유용한 수정을 통해 
    Cython 코드로 변환하는 것이다.
    NumPy 를 활용하는 고성능 코드를 작성하기 위한 다른 옵션으로는 포트란 77과 포트란 90 코드를 위한 래퍼를 생성해주는 f2py
    와 순수 C 확장을 작성하는 방법이 있다.
'''