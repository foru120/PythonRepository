import numpy as np

# 일차원 배열
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
print(x+y, x-y, x*y, x/y)
print(x/2.0)

# 다차원 배열
A = np.array([[1, 2], [3, 4]])
print(A)        # 배열 출력
print(A.shape)  # 배열 형태 출력
print(A.dtype)  # 배열 원소 데이터 타입

# element-wise product
B = np.array([[3, 0], [0, 6]])
print(A+B)
print(A*B)

# broadcast
print(A*6)

A = np.array([[1, 2], [3, 4]])
B = np.array([10, 20])
print(A*B)

X = np.array([[51, 55], [14, 19], [0, 4]])
print(X)
print(X[0], X[0][0])  # 원소 접근

for row in X:
    print(row)

X = X.flatten()  # 일차원 배열로 변환
print(X, X[np.array([0, 2, 4])])
print(X > 15)
print(X[X > 15])