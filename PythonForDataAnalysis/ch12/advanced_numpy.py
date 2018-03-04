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