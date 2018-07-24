#todo 텐서플로우 프로그램 구조 이해
'''
    1. 만들고자하는 신경망의 청사진을 만든다.
    2. 계산 그래프의 정의와 실행을 위한 두 개의 부분으로 나누어서 만든다.
    3. 그래프 정의와 실행 그래프의 분리는 여러 플랫폼 및 병렬 실행 기능을 제공한다.

    Computational Graph:
     - 계산 그래프 는 노드와 엣지들의 네트워크이다.
     - tensor object(constants, variables, placeholders) 와 operation object 가 정의된다.
     - 각 노드는 0 개 이상의 입력을 가질 수 있지만 출력은 항상 하나이다.
     - 네트워크 내의 노드들은 Objects 로 표현하고 엣지들은 연산들 사이에 흐르는 텐서로 표현한다.
     - 계산 그래프는 신경망의 청사진을 정의하지만 그 안에있는 텐서는 아직 값이 없다.
     - 계산 그래프를 만들기 위해 수행에 필요한 constants, variables, operations 를 정의한다.

    Execution of the graph:
     - 실행 그래프는 세션 객체 사용으로 수행된다.
     - 세션 객체는 텐서와 연산 객체가 평가되는 환경을 캡슐화한다.
     - 세션 객체는 실제 계산과 한 레이어에서 다른 레이어로의 정보 전송이 이루어지는 장소이다.
     - 다른 텐서 객체들의 값들은 세션 객체 내에서 초기화되고, 접근되고, 저장된다.
     - 세션 객체 전까지 텐서 객체는 단지 추상적인 정의였지만 여기서 생성된다.
'''

#todo How to do it...
import tensorflow as tf
'''computation graph 정의'''
v_1 = tf.constant([1, 2, 3, 4])
v_2 = tf.constant([2, 1, 5, 3])
v_add = tf.add(v_1, v_2)
'''execute graph'''
with tf.Session() as sess:
    print(sess.run(v_add))
# ※ 각 session 은 명시적으로 close() 메서드를 통해 닫아야하지만 with 절을 사용하면 종료되는 시점에 암묵적으로 세션을 닫아준다.

#todo How it works...
'''
    1. 변수들과 연산을 추가
    2. 순서대로 신경망을 계층별로 전달
    3. with tf.device() 절을 사용해서 계산 그래프의 다른 객체들이 CPU/GPU 를 사용하도록 설정하는 것이 가능하다.
    4. 위의 계산 그래프는 v_1 과 v_2 로 표현되는 두 개의 벡터와 하나의 Add 연산, 즉 3 개의 노드들로 구성 되어 있다.
    5. 이 그래프를 실체화하기 위해 tf.Session() 을 사용해서 세션 객체를 정의한다.
    6. 세션 클래스에 정의된 run 메서드를 사용해 실행한다. (fetches 내의 텐서들을 평가한다.)
    7. run 메서드는 v_add 로 이어지는 그래프 내의 모든 텐셔와 연산들을 실행한다.
    8. fetches 는 하나 이상의 텐서/연산이 될 수 있다.     
'''

#todo There's more...
'''
    간단한 벡터 추가나 작은 메시지를 출력할 때 왜 많은 코드를 써야하는 지 궁금할 것이다.
    한 라인으로 이 작업을 매우 편리하게 할 수 있다.
'''
print(tf.Session().run(tf.add(tf.constant([1,2,3,4]), tf.constant([2,1,5,3]))))
'''
    이런 형태의 코드는 계산 그래프에 영향을 미치지 않지만, for loop 안에서 반복적으로 수행되어질 때 메모리를 많이 사용할 수 있다.
    모든 텐서 및 연산 객체를 명시적으로 정의하는 습관을 가지면 코드를 더 쉽게 읽을 수 있을뿐만 아니라 계산 그래프를 더 깨끗하게 
    시각화하는데 도움이 된다.
    텐서보드를 사용한 그래프 시각화는 텐서플로우의 가장 유용한 능력중 하나이다.
    우리가 만든 계산 그래프는 Graph 객체의 도움을 가지고 보여질 수 있다.
    tf.Session.InteractiveSession 대신에 tf.InteractiveSession 을 사용해서 명시적인 세션 호출 없이 eval() 을 사용해 직접 텐서 객체
    를 호출할 수 있다.
'''
sess = tf.InteractiveSession()
v_1 = tf.constant([1,2,3,4])
v_2 = tf.constant([2,1,5,3])
v_add = tf.add(v_1, v_2)
print(v_add.eval())
sess.close()

#todo Working with constants, variables, and placeholders
'''
    가장 간단한 형태인 텐서플로우는 텐서들을 가지고 다른 수학 연산들을 정의하고 수행하기 위해 라이브러리들을 제공한다.
    텐서는 기본적으로 n-차원의 matrix 이다.
    scalar, vector 및 matrices 는 특수 유형의 텐서이다.
    텐서플로우는 3가지 형태의 텐서들을 제공한다.(Constants, Variables, Placeholders)
    Constants
      - 값이 변할 수 없는 텐서.
    Variables
      - 세션 내부에서 값이 변하는게 요구될 때 사용한다.
      - neural network 에서 훈련 세션 동안 가중치들이 변경되는 것이 필요하다.
      - variables 는 사용하기 전에 명시적으로 초기화 되어야 한다.
      - constants 는 계산 그래프 정의에 저장되어 있어 그래프가 로드 될 때마다 로드되지만, variables 는 별도의 파라미터 서버에 존재한다.
    Placeholders
     - placeholders 는 텐서플로우 그래프 내부로 값들을 주입하기 위해 사용된다.
     - data 를 주입하기 위해 fead_dict 를 가지고 사용한다.
     - 일반적으로 neural network 훈련동안 새로운 훈련 예제들을 주입하기 위해 사용한다.
     - 세션 내의 그래프를 수행하는 동안 placeholder 에 값을 할당한다.
     - 데이터의 요구 없이 계산 그래프와 연산들을 만들 수 있게 한다.
     - placeholder 에 데이터가 없으므로 초기화 할 필요가 없다.
'''

#todo How to do it...
# 1. scalar constant 선언
t_1 = tf.constant(value=4)
# 2. shape [1, 3] 의 constant vector 선언
t_2 = tf.constant(value=[4, 3, 2])
# 3. 모든 요소들이 0을 가진 텐서를 만들기 위해 tf.zeros() 사용 (dtype(int32, float32 등) 을 가지고 [M, N] 형태의 zero matrix 생성)
zero_t = tf.zeros(shape=[2, 3], dtype=tf.int32)
# 4. 기존에 존재하는 Numpy 배열과 텐서 상수를 가지고 같은 형태의 텐서 상수를 생성
tf.zeros_like(tensor=t_2)
tf.ones_like(tensor=t_2)
# 5. 모든 요소가 1로 설정된 텐서 생성
ones_t = tf.ones(shape=[2, 3], dtype=tf.int32)

# 1. 전체 num 값 내에서 start 부터 end 까지 균일하게 간격을 둔 벡터 시퀀스를 생성
#    start: 시작 값(float), stop: 종료 값(float), num: 개수
range_t = tf.linspace(start=2., stop=5., num=10)  # (stop-start)/(num-1)
# 2. start 부터 limit 까지 delta 간격으로 숫자의 시퀀스를 생성
range_tt = tf.range(start=2, limit=5, delta=2)

# 1. shape [M,N] 의 normal distribution 으로 random value 생성
normal_random = tf.random_normal(shape=[2, 3], mean=2.0, stddev=4, seed=12)
# 2. shape [M,N] 의 truncated normal distribution 으로 random value 생성
t_normal_random = tf.truncated_normal(shape=[2, 3], mean=2.0, stddev=2, seed=12)
# 3. shape [M,N] 의 gamma distribution 으로 random value 생성
u_random = tf.random_uniform(shape=[2, 3], maxval=4, seed=12)
# 4. 특정 크기로 주어진 텐서를 랜덤하게 crop
crop_value = tf.random_crop(u_random, size=[2, 1], seed=12)
# 5. random 순서로 훈련 데이터를 표현 (첫 번째 차원을 따라서 랜덤하게 섞는다)
tf.random_shuffle(u_random)
# 6. 랜덤하게 생성된 텐서들을 초기 seed 값에 의해 영향을 받는다.
#    여러번의 수행 또는 세션에서 같은 랜덤 값을 얻기 위해서는 상수 값으로 설정되어야 한다.
#    사용중인 랜덤 텐서가 많은 경우 tf.set_random_seed() 를 사용하여 임의로 생성된 모든 텐서의 seed 를 설정할 수 있다.

# 1. variable 의 정의는 초기화 해야하는 상수/랜덤 값들을 포함한다.
#    variables 는 neural network 내에서 가중치와 편향으로 표현된다.
rand_t = tf.random_uniform(shape=[50, 50], minval=0, maxval=10, seed=0)
t_a = tf.Variable(initial_value=rand_t)
t_b = tf.Variable(initial_value=rand_t)
# 2. weights 와 bias 변수를 각각 normal distribution 과 zeros 로 초기화 하고, 계산 그래프 내에서 정의된 변수의 이름을 설정
weights = tf.Variable(initial_value=tf.random_normal(shape=[100, 100], stddev=2), name='weights')
bias = tf.Variable(initial_value=tf.zeros(shape=100), name='biases')
# 3. 앞의 예제들의 variables 초기화 소스는 어떤 상수이다.
#    초기화 된 다른 variable 을 지정할 수 있다.
weights2 = tf.Variable(weights.initialized_value(), name='w2')
# 4. variable 의 정의는 초기화 되는 방법을 지정하지만 모든 선언된 variable 을 명시적으로 초기화해야 한다.
#    초기화 연산 객체 정의에 의해 명시적으로 초기화 한다.
initial_op = tf.global_variables_initializer()
# 5. 각 variable 은 그래프가 실행되는 동안 tf.Variable.initializer 를 사용하여 개별적으로 초기화 될 수 있다.
bias = tf.Variable(tf.zeros([100, 100]))
with tf.Session() as sess:
    sess.run(bias.initializer)
# 6. Saver class 를 사용해 변수들을 저장할 수 있다.
saver = tf.train.Saver()
# 7. placeholder 정의
tf.placeholder(dtype=tf.int32, shape=[3, 3], name='x')
# 8. placefolder 를 선언하는 동안 dtype 은 지정되어야 한다.
x = tf.placeholder(dtype=tf.float32)
y = 2 * x
data = tf.random_uniform(shape=[4, 5], minval=10)
with tf.Session() as sess:
    x_data = sess.run(data)
    print(sess.run(y, feed_dict={x: x_data}))

#todo How it works...