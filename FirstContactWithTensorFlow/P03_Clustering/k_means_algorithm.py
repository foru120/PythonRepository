###################################################################################################
## ▣ K-평균 알고리즘
##  - 주어진 데이터를 지정된 군집 개수(K)로 그룹화 한다.
##    한 군집 내의 데이터들은 동일한 성질을 가지며 다른 그룹과는 구별된다.
##    알고리즘의 결과는 중심이라고 부르는 K개의 점으로서, 이들은 각기 다른 그룹의 중심점을 나타내며 데이터들은
##    K개의 군집 중 하나에만 속할 수 있다.
##    한 군집 내의 모든 데이터들은 다른 어떤 중심들보다 자기 군집 중심과의 거리가 더 가깝다.
##  - 반복 개선 기법 사용
##   1. 초기 단계(0 단계) : K개 중심의 초기 집합을 결정 (K 개를 임의로 선택)
##   2. 할당 단계(1 단계) : 각 데이터를 가장 가까운 군집에 할당
##   3. 업데이트 단계(2 단계) : 각 그룹에 대해 새로운 중심을 계산
###################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# num_points = 2000
# vectors_set = []

# for i in range(num_points):
#     if np.random.random() > 0.5:
#         vectors_set.append([np.random.normal(0.0, 0.9), np.random.normal(0.0, 0.9)])
#     else:
#         vectors_set.append([np.random.normal(3.0, 0.5), np.random.normal(1.0, 0.5)])

vectors_set = np.loadtxt('data/academy.csv', delimiter=',')
vectors_set = vectors_set[:, [1,2]]
# df = pd.DataFrame({'x': [v[0] for v in vectors_set],
#                    'y': [v[1] for v in vectors_set]})
x = [v[0] for v in vectors_set]
y = [v[1] for v in vectors_set]

# plt.plot(x, y, 'ro')
# plt.show()

vectors = tf.constant(vectors_set)
k = 4

#  1단계 : 랜덤으로 K 개의 초기 집합을 설정
#   - tf.random_shuffle(object) : 상수 텐서를 랜덤으로 섞는다.
#   - tf.slice(object, begin, size) : 랜덤으로 얻어진 상수 텐서를 처음부터 k개 만큼 자른다.
#   - tf_Variable(object) : 상수 텐서를 변수 텐서로 변환.
centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0, 0], [k, -1]))

#   - expanded_vectors   : (2000, 2) --> (1, 2000, 2), expanded_vectors.shape
#   - expanded_centroids : (4, 2) --> (4, 1, 2), expanded_centroids.shape
#   ※ 크기가 1인 차원만 broadcasting 기능이 동작하므로, 두 텐서간의 차원 확장이 필요하다.
expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroids = tf.expand_dims(centroids, 1)

#  2단계 : 각 데이터를 가장 가까운 군집에 할당
#  → 중심과 각 점간의 거리를 구하기 위해 유클리드 제곱 거리 알고리즘을 선택.
#   - tf.subtract(tensor1, tensor2) : 두 개의 텐서사이의 빼기 연산을 수행.
#   - tf.square(tensor) : 각각의 텐서 값에 제곱을 수행.
#   - tf.reduce_sum(tensor, dimension) : 텐서의 dimension 차원에 해당하는 값을 더하고 차원을 축소. 여기서는 x, y 값을 더해주는 역할
#   - tf.argmin(tensor, dimension) : dimension(0:열, 1:행, 2:면...), 0 이면 열을 기준으로 가장 작은 값을 가지는 행의 index 값을 출력.

assignments = tf.argmin(tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2), 0)

#  3단계 : 각 그룹에 대해 새로운 중심을 계산
#   - tf.equal(tensor, cluster) : tensor 값에 대해 cluster 와 비교해서 같으면 True, 다르면 False 를 출력.
#   - tf.where(condition) : condition 을 만족하는 값에 대해 해당 인덱스를 출력.
#   - tf.reshape(tensor, shape) : tensor 의 차원을 shape 에 맞게 변형.
#   - tf.gather(tensor, indices) : tensor 에 대해 indices 를 만족하는 값을 출력.
#   - tf.reduce_mean(tensor, axis) : tensor 를 axis 에 대해 평균 값을 구해서 출력.
clustered_data = [tf.gather(vectors, tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1])) for c in range(k)]
means = tf.concat([tf.reduce_mean(data, axis=1) for data in clustered_data], 0)
# means = tf.concat([tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1])), axis=1) for c in range(k)], 0)

#   - tf.assign(tensor1, tensor2) : tensor1 을 tensor2 로 update 수행.
update_centroids = tf.assign(centroids, means)

#   - tf.global_variables_initializer() : 사용된 텐서 변수들에 대해 초기화.
init_op = tf.global_variables_initializer()

#   - tf.Session() : 텐서플로우 세션 생성, sess.run(init_op) : 텐서 변수 초기화 수행.
sess = tf.Session()
sess.run(init_op)

for step in range(500):
    _, centroid_values, assignment_values = sess.run([update_centroids, centroids, assignments])

colors = ["g.", "r.", "c.", "y."]

#  군집별로 점 출력
for i in range(k):
    temp_data = sess.run(clustered_data[i])
    plt.plot(temp_data[0, :, 0], temp_data[0, :, 1], colors[i], markersize=10)

#  중심값 출력
plt.scatter(centroid_values[:, 0], centroid_values[:, 1], marker='x', s=150, linewidths=5, zorder=10)
plt.show()

# print('vectors.shape : ', vectors.shape)
# print('centroids.shape : ', centroids.shape)
# print('expanded_vectors.shape : ', expanded_vectors.shape)
# print('expanded_centroids.shape : ', expanded_centroids.shape)
# print('assignments.shape : ', assignments.shape)
# print('assignments : ', sess.run(assignments))
# print('mean : ', sess.run(means))
# print('update_centroids : ', sess.run(update_centroids))