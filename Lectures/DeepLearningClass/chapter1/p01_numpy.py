# 1. 배열 만들기
import numpy as np

a = np.array([[1, 2], [3, 4]])
print(a)

# 2. 사칙 연산
print('====================================================================================================')
print('== 문제 1. 아래의 a 배열에 모든 원소에 5를 더한 결과를 출력하시오!')
print('====================================================================================================\n')
a = np.array([[1, 2], [3, 4]])
print(a + 5)

print('====================================================================================================')
print('== 문제 2. 아래의 배열의 원소들의 평균값을 출력하시오!')
print('====================================================================================================\n')
a = np.array([1,2,4,5,5,7,10,13,18,21])
print(np.mean(a))

print('====================================================================================================')
print('== 문제 3. a 배열의 중앙값을 출력하시오!')
print('====================================================================================================\n')
print(np.median(a))

print('====================================================================================================')
print('== 문제 4. a 배열의 최대값과 최소값을 출력하시오!')
print('====================================================================================================\n')
print(np.max(a), np.min(a))

print('====================================================================================================')
print('== 문제 5. a 배열의 표준편차와 분산을 출력하시오!')
print('====================================================================================================\n')
print(np.std(a), np.var(a))

print('====================================================================================================')
print('== 문제 6. 아래의 행렬식을 numpy 로 구현하시오!')
print('====================================================================================================\n')
a = np.array([[1, 3, 7], [1, 0, 0]])
b = np.array([[0, 0, 5], [7, 5, 0]])
print(a + b)

print('====================================================================================================')
print('== 문제 7. 아래의 numpy 배열을 생성하고 원소중에 10 만 출력해보시오!')
print('====================================================================================================\n')
a = np.array([[1, 2, 3], [4, 10, 6], [8, 9, 20]])
print(a[1][1])

print('====================================================================================================')
print('== 문제 8. (점심시간 문제) 아래의 행렬 연산을 파이썬으로 구현하시오!')
print('====================================================================================================\n')
a = np.array([[1, 2], [3, 4]])
b = np.array([10, 20])
print(a * b)

print('====================================================================================================')
print('== 문제 9. 아래의 그림의 행렬 연산을 numpy 로 구현하시오!')
print('====================================================================================================\n')
a = np.array([[0], [10], [20], [30]])
b = np.array([0, 1, 2])
print(a + b)

print('====================================================================================================')
print('== 문제 10. 아래의 행렬식을 numpy 로 구현하고 아래의 요소에서 15 이상인것만 출력하시오!')
print('====================================================================================================\n')
a = np.array([[51, 55], [14, 19], [0, 4]])
a[a>=15]

print('====================================================================================================')
print('== 문제 11. 아래의 행렬식을 numpy 를 이용하지 않고 list 변수로 구현하고 아래의 행렬식에서 행의 개수가 몇 개 인지 출력하시오!')
print('====================================================================================================\n')
a = [[1, 3, 7], [1, 0, 0]]
print(len(a))

print('====================================================================================================')
print('== 문제 12. 아래의 행렬식을 numpy 를 이용하지 않고 list 변수로 구현하고 열의 개수가 몇개인지 출력하시오!')
print('====================================================================================================\n')
a = [[1, 3, 7], [1, 0, 0]]
print(len(a[0]))

print('====================================================================================================')
print('== 문제 13. 아래의 행렬식의 덧셈 연산을 numpy 를 이용하지 않고 수행하시오!')
print('====================================================================================================\n')
a = [[1, 3, 7], [1, 0, 0]]
b = [[0, 0, 5], [7, 5, 0]]
result = []

for row_idx in range(len(a)):
    temp = []
    for col_idx in range(len(a[row_idx])):
        temp.append(a[row_idx][col_idx] + b[row_idx][col_idx])
    result.append(temp)
print(result)

print('====================================================================================================')
print('== 문제 14. 아래의 행렬식을 numpy 이용하지 않고 구현하시오!')
print('====================================================================================================\n')
a = [[1, 2], [3, 4]]
b = [[5, 6], [7, 8]]
result = []

for row_idx in range(len(a)):
    temp = []
    for col_idx in range(len(a[row_idx])):
        temp.append(a[row_idx][col_idx] * b[row_idx][col_idx])
    result.append(temp)
print(result)

print('====================================================================================================')
print('== 문제 15. 아래의 행렬 연산을 numpy 와 numpy 를 이용하지 않았을 때 2가지 방법으로 구현하시오!')
print('====================================================================================================\n')
a = [[10, 20], [30, 40]]
b = [[5, 6], [7, 8]]
result = []

for row_idx in range(len(a)):
    temp = []
    for col_idx in range(len(a[row_idx])):
        temp.append(a[row_idx][col_idx] - b[row_idx][col_idx])
    result.append(temp)
print(result)

print('====================================================================================================')
print('== 문제 16. numpy 의 broadcast 를 사용한 연산을 numpy 를 이용하지 않는 방법으로 구현하시오!')
print('====================================================================================================\n')
a = np.array([[1, 2], [3, 4]])
b = np.array([10, 20])
c = [[v[col_idx]*b[col_idx] for col_idx in range(len(v))] for v in a]
print(c)

# ■ matplotlib 사용법
import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0, 12, 0.01)
print(t)

plt.figure()
plt.plot(t)
plt.show()

print('====================================================================================================')
print('== 문제 18. 위의 그래프에 x 축의 이름을 size 라고 하고 y 축의 이름을 cost 라고 하시오!')
print('====================================================================================================\n')
import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0, 12, 0.01)
print(t)

plt.figure()
plt.plot(t)
plt.grid()
plt.xlabel('size')
plt.ylabel('cost')
plt.show()

print('====================================================================================================')
print('== 문제 19. 위의 그래프에 전체 제목을 size & cost 라고 하시오!')
print('====================================================================================================\n')
import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0, 12, 0.01)
print(t)

plt.figure()
plt.plot(t)
plt.grid()
plt.xlabel('size')
plt.ylabel('cost')
plt.title('size & cost')
plt.show()

print('====================================================================================================')
print('== 문제 20. 아래의 numpy 배열로 산포도 그래프를 그리시오!')
print('====================================================================================================\n')
x = np.array([0,1,2,3,4,5,6,7,8,9])
y = np.array([9,8,7,9,8,3,2,4,3,4])
plt.figure()
plt.scatter(x, y)
plt.show()

print('====================================================================================================')
print('== 문제 21. 위의 그래프를 라인 그래프로 출력하시오!')
print('====================================================================================================\n')
x = np.array([0,1,2,3,4,5,6,7,8,9])
y = np.array([9,8,7,9,8,3,2,4,3,4])
plt.figure()
plt.plot(x, y)
plt.show()

print('====================================================================================================')
print('== 문제 22. 치킨집 년도별 창업건수를 가지고 라인 그래프를 그리시오.')
print('====================================================================================================\n')
start = np.loadtxt('D:\\KYH\\02.PYTHON\\PythonRepository\\DeepLearningClass\\chapter1\\data\\start.csv', skiprows=1, unpack=True, delimiter=',')
start_x = start[0]
start_y = start[4]
plt.figure(figsize=(6,4))
plt.plot(start_x, start_y)
plt.show()

print('====================================================================================================')
print('== 문제 23. 폐업건수도 위의 그래프에 겹치게 해서 출력하시오.')
print('====================================================================================================\n')
end = np.loadtxt('D:\\KYH\\02.PYTHON\\PythonRepository\\DeepLearningClass\\chapter1\\data\\end.csv', skiprows=1, unpack=True, delimiter=',')
end_x = end[0]
end_y = end[4]
plt.figure()
plt.plot(end_x, end_y, label='open')
plt.show()

print('====================================================================================================')
print('== 문제 24. 위의 그래프에 legend 도 출력하시오.')
print('====================================================================================================\n')
end = np.loadtxt('D:\\KYH\\02.PYTHON\\PythonRepository\\DeepLearningClass\\chapter1\\data\\end.csv', skiprows=1, unpack=True, delimiter=',')
end_x = end[0]
end_y = end[4]
plt.figure()
plt.plot(end_x, end_y, label='end')
plt.legend()
plt.show()

print('====================================================================================================')
print('== 문제 25. 책 44 페이지의 이미지 표시를 파이썬으로 구현하시오.')
print('====================================================================================================\n')
from matplotlib.image import imread
img = imread('data/lena.png')
plt.imshow(img)
plt.show()

print('====================================================================================================')
print('== 문제 26. 고양이 사진을 출력하시오.')
print('====================================================================================================\n')
from matplotlib.image import imread
img = imread('data/cat.png')
plt.imshow(img)
plt.show()
