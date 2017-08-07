import numpy as np

# 6x6 행렬 만들기
a = np.array([i for i in range(36)]).reshape(6,6)


# 3x3 필터 만들기
Filter = np.eye(3,3)

# 행렬 확인
print('---a\n',a)
print('---filter\n',Filter)

############################################################################

### 합성곱 연산 방법 1 ###

# 단일 곱셈-누산 vs 행렬곱 연산
d = np.array([[1,2,3],[4,5,6],[7,8,9]])
print('---d\n',d)
print('---단일 곱셈-누산 결과\n', np.sum(Filter * d)) # (1 * 1) + (-1 * 2) + (-1 * 3) + (1 * 4)
print('---행렬곱 연산 결과\n', np.dot(Filter, d))

# 넘파이 array indexing
print('---a[:,:]\n',a[:,:])            # a 전체 출력
print('---a[:,1:2]\n',a[:,0:3])      # a의 전체행 / 첫번째열~세번째열 출력
print('---a[0:3,4:5]\n',a[3:5,4:5])    # a의 네번재행~다섯번째행 / 다섯번째열 출력

# 스트라이드
for rn in range(len(a[0])-1):
    for cn in range(len(a[1])-1):
        print('---',[rn,cn],'\n',a[rn:rn+2, cn:cn+2])

# 합성곱 연산
result = []

for rn in range(len(a[0])-2):
    for cn in range(len(a[1])-2):
        result.append(np.sum(a[rn:rn+3, cn:cn+3] * Filter))

print('---result\n',result)
print('---len(result)\n', len(result))
len_a = int(np.sqrt(len(result)))
result = np.array(result).reshape(len_a,len_a)
print('---result.reshape\n', result)

# 패딩
a_pad = np.pad(a, pad_width=1, mode='constant', constant_values=0)
print('---a_pad\n',a_pad)

a_pad2 = np.pad(a, pad_width=2, mode='constant', constant_values=-1) # constant_values로 숫자 변경 가능
print('---a_pad2\n',a_pad2)

a_pad3 = np.pad(a, pad_width=((1,2),(3,4)), mode='constant', constant_values=0) # pad_width=( (위, 아래), (왼쪽, 오른쪽 패드 수) )
print('---a_pad3\n',a_pad3)

# 패딩 적용한 합성곱 연산
result2 = []

for rn in range(len(a_pad[0])-2):
    for cn in range(len(a_pad[1])-2):
        result2.append(np.sum(a_pad[rn:rn+3, cn:cn+3] * Filter))

print('---result2\n', result2)
print('---len(result2)\n', len(result2))
len_a2 = int(np.sqrt(len(result2)))
result2 = np.array(result2).reshape(len_a2, len_a2)
print('---result2.reshape\n', result2)


# 문제(1). 0부터 143까지 원소로 이뤄진 12x12 행렬을 만들고, 4x4 필터(단위 행렬)를 이용해 합성곱을 해보세요.
#         (단, 스트라이드는 1, 출력 행렬은 12x12가 되도록 패딩을 적용하세요)
x = np.arange(0, 144).reshape((-1, 12))
x = np.pad(x, pad_width=2, mode='constant', constant_values=0)
f = np.eye(4, 4)

result = []

for rn in range(len(x[0])-4):
    for cn in range(len(x[1])-4):
        result.append(np.sum(x[rn:rn+4, cn:cn+4] * f))
result = np.array(result).reshape(-1, 12)
print(result)
print(result.shape)

############################################################################

### 합성곱 연산 2 ###

# 단일 곱셈 누산 -> 행렬곱 연산
print('---Filter\n', Filter)
print('---Filter.flatten()\n', Filter.flatten())  # flatten은 행렬을 벡터로 만들어줌

# 행렬곱하기 좋게 행렬을 변환해주는 함수
def im2col_sliding_strided(A, filtersize, stepsize=1): # A = 변환할 행렬, filtersize = 필터 크기, stepsize = 스트라이드
    m, n = A.shape
    s0, s1 = A.strides
    BSZ = [m + 1 - filtersize[0], n + 1 - filtersize[1]]
    nrows = m - BSZ[0] + 1
    ncols = n - BSZ[1] + 1
    shp = BSZ[0], BSZ[1], nrows, ncols
    strd = s0, s1, s0, s1

    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view.reshape(BSZ[0] * BSZ[1], -1)[:, ::stepsize]

print('---변환 전 a\n', a)
print('---변환 후 a\n', im2col_sliding_strided(a, [3,3]))

# 행렬곱 연산을 이용한 합성곱

a_pad = np.pad(a, pad_width=1, mode='constant', constant_values=0)
a2 = im2col_sliding_strided(a_pad, [3,3])
Filter2 = Filter.flatten()
result = np.dot(a2, Filter2)
print('---합성곱 결과\n', result)
result = result.reshape(6,6)
print('---최종 결과\n', result)

# 문제(2). 앞에서 배운 두 가지 합성곱 방법을 각각 이용하여 0~1사이의 난수로 이루어진 300x300 행렬을
#          9x9 필터(단위행렬)를 이용해 합성곱을 해보세요. (단, 스트라이드는 1, 출력 행렬 크기는 300x300이 되도록 패딩을 적용하세요)

print('====================================================================================================')
print('== 문제 154. 0부터 15까지 원소로 이루어진 4x4 행렬을 만드시오!')
print('====================================================================================================\n')
a = np.arange(0, 16).reshape((4, 4))


print('====================================================================================================')
print('== 문제 155. 위에서 만든 행렬에 0 패딩 1을 수행하시오.')
print('====================================================================================================\n')
a = np.pad(a, pad_width=1, mode='constant', constant_values=0)


print('====================================================================================================')
print('== 문제 156. 0부터 35까지 원소로 이루어진 6x6 행렬을 만들고 0부터 15까지의 원소로 이루어진 4x4 필터를'
      '이용해서 합성곱을 하시오!')
print('====================================================================================================\n')
x = np.arange(0, 36).reshape((-1, 6))
x = np.pad(x, pad_width=2, mode='constant', constant_values=0)
f = np.arange(0, 16).reshape((4, 4))

result = []

for rn in range(x.shape[0]-4):
    for cn in range(x.shape[1]-4):
        result.append(np.sum(x[rn:rn+4, cn:cn+4] * f))
result = np.array(result).reshape(-1, 6)
print(result)
print(result.shape)


print('====================================================================================================')
print('== 문제 158. 아래와 같이 출력값(OH)와 Stride(S) 와 입력값(H)와 필터값(FH)를 입력하면 P(패딩)이 출력이 되는 함수를 생성하시오.')
print('====================================================================================================\n')
import math
padding = lambda OH,S,H,FH : math.ceil(((OH-1)*S-H+FH)/2)


print('====================================================================================================')
print('== 문제 159. 0부터 15까지 원소로 이루어진 4x4 행렬을 만들고 0부터 8까지의 원소로 이루어진 3x3 필터를'
      '이용해서 합성곱을 하시오!')
print('====================================================================================================\n')
x = np.arange(0, 16).reshape((-1, 4))
x = np.pad(x, pad_width=1, mode='constant', constant_values=0)
f = np.arange(0, 9).reshape((-1, 3))

result = []
x
for rn in range(x.shape[0]-2):
    for cn in range(x.shape[1]-2):
        result.append(np.sum(x[rn:rn+3, cn:cn+3] * f))
result = np.array(result).reshape(-1, 4)
print(result)
print(result.shape)


print('====================================================================================================')
print('== 문제 160. 0부터 35까지 원소로 이루어진 6x6 행렬을 만들고 0부터 8까지의 원소로 이루어진 3x3 필터를'
      '이용해서 합성곱을 하시오!')
print('====================================================================================================\n')
x = np.arange(0, 36).reshape((-1, 6))
x = np.pad(x, pad_width=1, mode='constant', constant_values=0)
f = np.arange(0, 9).reshape((-1, 3))

result = []
x
for rn in range(x.shape[0]-2):
    for cn in range(x.shape[1]-2):
        result.append(np.sum(x[rn:rn+3, cn:cn+3] * f))
result = np.array(result).reshape(-1, 6)
print(result)
print(result.shape)


print('====================================================================================================')
print('== 문제 160. 0부터 35까지 원소로 이루어진 6x6 행렬을 만들고 0부터 8까지의 원소로 이루어진 3x3 필터를'
      '이용해서 합성곱을 하시오!')
print('====================================================================================================\n')
x = np.arange(0, 25).reshape((-1, 5))
x = np.pad(x, pad_width=1, mode='constant', constant_values=0)
f = np.arange(0, 4).reshape((-1, 2))

result = []

for rn in range(x.shape[0]-2):
    for cn in range(x.shape[1]-2):
        result.append(np.sum(x[rn:rn+2, cn:cn+2] * f))
result = np.array(result).reshape(-1, 5)
print(result)
print(result.shape)


print('====================================================================================================')
print('== 문제 163. 입력 데이터를 15행의 8열로 0~119번까지의 원소로 생성해서 채널을 10개로 만든걸 입력 데이터를 만들고'
      '필터를 3행에 3열로 해서 0~8 까지의 원소로 생성해서 채널을 10개로 생성해서 합성곱을 구현하시오!')
print('====================================================================================================\n')
x = np.array([np.arange(0, 120).reshape((15, 8)) for _ in range(10)])
f = np.array([np.arange(0, 9).reshape((3, 3)) for _ in range(10)])

for idx in range(x.shape[0]):
    temp = np.pad(x[idx], pad_width=1, mode='constant', constant_values=0)
    result = []
    for rn in range(temp.shape[0]-2):
        for cn in range(temp.shape[1]-2):
            result.append(np.sum(temp[rn:rn+3, cn:cn+3] * f))
    result = np.array(result).reshape(-1, 8)
    print(result)
    print(result.shape)


print('====================================================================================================')
print('== NCS 문제 3. 0부터 80까지의 원소로 이루어진 9x9 행렬을 만들고 0부터 15까지의 원소로 이루어진 4x4 필터를 이용해서 합성곱을 하시오!')
print('====================================================================================================\n')
import numpy as np
x = np.arange(0, 81).reshape((-1, 9))
x = np.pad(x, pad_width=2, mode='constant', constant_values=0)
f = np.arange(0, 16).reshape((-1, 4))

result = []

for rn in range(x.shape[0]-4):
    for cn in range(x.shape[1]-4):
        result.append(np.sum(x[rn:rn+4, cn:cn+4] * f))
result = np.array(result).reshape(-1, 9)
print(result)
print(result.shape)


print('====================================================================================================')
print('== NCS 문제 4. 문제 3번을 다시 수행하는데 원소의 값은 똑같고 입력값의 채널이 3차원일때의 합성곱을 구하시오!')
print('====================================================================================================\n')
x = np.array([np.arange(0, 81).reshape((-1, 9)) for _ in range(10)])
f = np.array([np.arange(0, 16).reshape((-1, 4)) for _ in range(10)])

for idx in range(x.shape[0]):
    temp = np.pad(x[idx], pad_width=2, mode='constant', constant_values=0)
    result = []
    for rn in range(temp.shape[0]-4):
        for cn in range(temp.shape[1]-4):
            result.append(np.sum(temp[rn:rn+4, cn:cn+4] * f))
    result = np.array(result).reshape(-1, 9)
    print(result)
    print(result.shape)


print('====================================================================================================')
print('== 문제 164. (점심시간 문제) 문제 162번 최대 풀링을 파이썬으로 구현하시오.')
print('====================================================================================================\n')
a = np.array([[21, 8, 8, 12], [12, 19, 9, 7], [8, 10, 4, 3], [18, 12, 9, 10]])

def pooling(x):
    result = []
    w, h = x.shape
    for i in range(0, w, 2):
        for j in range(0, h, 2):
            result.append(np.max(x[i:i+2, j:j+2]))
    return np.array(result).reshape((2, 2))
print(pooling(a))


print('====================================================================================================')
print('== 문제 167. 0부터 15까지 원소로 이루어진 4x4 행렬을 만들고 0부터 8까지의 원소로 이루어진 3x3 필터를 이용해서'
      '합성곱을 하는데 이번에는 im2col 을 활용해서 수행하시오.')
print('====================================================================================================\n')
import numpy as np
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """다수의 이미지를 입력받아 2차원 배열로 변환한다(평탄화).

    Parameters
    ----------
    input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩

    Returns
    -------
    col : 2차원 배열
    """
    C, H, W = input_data.shape
    N = 1
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col

x = np.array([np.arange(0, 16).reshape((-1, 4)) for _ in range(10)])
x = np.pad(x, pad_width=1, mode='constant', constant_values=0)
f = np.array([np.arange(0, 9).reshape((-1, 3)) for _ in range(10)])

result = im2col(x, 3, 3, 1)
print(result)
print(result.shape)