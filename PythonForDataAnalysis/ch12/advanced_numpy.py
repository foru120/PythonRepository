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