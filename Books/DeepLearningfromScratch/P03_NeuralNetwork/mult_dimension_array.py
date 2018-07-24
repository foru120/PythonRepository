import numpy as np

# 1차원 배열
A = np.array([1, 2, 3, 4])
print(A, np.ndim(A), A.shape, A.shape[0])  # ndim : 배열의 차원 수 확인

# 2차원 배열
B = np.array([[1, 2], [3, 4], [5, 6]])
print(B, np.ndim(B), B.shape, B.shape[0])

# 행렬의 내적
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(np.dot(A, B))  # dot 는 행렬의 내적을 구하는 함수
print(np.dot(B, A))

A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[1, 2], [3, 4], [5, 6]])
print(np.dot(A, B))

A = np.array([[1, 2], [3, 4], [5, 6]])
B = np.array([7, 8])
print(np.dot(A, B))

# 신경망의 내적
X = np.array([1, 2])
W = np.array([[1, 3, 5], [2, 4, 6]])
Y = np.dot(X, W)
print(Y)