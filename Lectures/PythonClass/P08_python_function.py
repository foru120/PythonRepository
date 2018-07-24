# 7장. 함수
#  * 목차
#   1. 함수 생성하는 함수
#   2. 기본값 매개변수와 키워드 매개변수
#   3. 가변 매개변수
#   4. 매개 변수로 함수를 사용하는 경우
#   5. 함수 밖의 변수, 함수 안의 변수
#   6. 재귀함수

print('')
print('====================================================================================================')
print('== 문제 146. 아래와 같이 이름을 입력해서 함수를 실행하면 해당 사원의 부서위치가 출력되게 하시오.')
print('====================================================================================================')
import pandas as pd

def find_loc(ename):
    empFrame = pd.read_csv('D:\\KYH\\02.PYTHON\\data\\emp.csv')
    deptFrame = pd.read_csv('D:\\KYH\\02.PYTHON\\data\\deptPD.csv')

    result = pd.merge(empFrame, deptFrame, on='deptno')
    return result[['loc']][result['ename'] == ename].values[0]

print(find_loc('SMITH'))


print('')
print('====================================================================================================')
print('== 문제 147. 미분계수를 구하는 함수를 생성하는 데, 함수 f(x) = 2x**2 + 1 일때 x 가 -2 일때의 기울기를 구하시오!')
print('====================================================================================================')
def numerical_diff(f, x):
    delta = 1e-4
    a = (f(x+delta)-f(x-delta))/2*delta
    return a

print(numerical_diff(lambda x: 2*pow(x, 2) + 1, -2))


print('')
print('====================================================================================================')
print('== 문제 148. 함수 f(x) = x**2-x+5 함수의 x 가 -2 일때의 미분계수를 구하시오.')
print('====================================================================================================')
def numerical_diff(f, x):
    delta = 1e-4
    a = (f(x+delta)-f(x-delta))/2*delta
    return a

print(numerical_diff(lambda x: pow(x, 2) - x + 5, -2))


# ■ 가변 매개변수
#  - 문자열.format() 함수처럼 매개변수의 수가 유동적인 함수를 만들고 싶을 때 사용하는 변수
#    함수 실행할 때 매개변수를 10개, 20개를 입력해도 제대로 동작을 한다.
def merge_string(*text_list):
    result = ''
    for s in text_list:
        result = result + s + ' '
    return result

print(merge_string('아버지가', '방에'))

# ■ 파이썬에서 * 가 쓰이는 경우
#  1. 가변 매개변수
#  2. 리스트 변수내의 요소들을 뽑아낼때


print('')
print('====================================================================================================')
print('== 문제 150. mit ttt 코드에서 보드판을 출력하는 printboard 함수를 분석하시오.')
print('====================================================================================================')
# States as integer : manual coding
EMPTY = 0
PLAYER_X = 1
PLAYER_O = 2
DRAW = 3
BOARD_FORMAT = """----------------------------
| {0} | {1} | {2} |
|--------------------------|
| {3} | {4} | {5} |
|--------------------------|
| {6} | {7} | {8} |
----------------------------"""
NAMES = [' ', 'X', 'O']

def printboard(state):
    """ Print the board from the internal state."""
    cells = []
    for i in range(3):
        for j in range(3):
            cells.append(NAMES[state[i][j]].center(6))
    print(cells)
    print(*cells)
    print(BOARD_FORMAT.format(*cells))

printboard([[1,2,0],[0,0,0],[0,0,0]])
print(BOARD_FORMAT.format('a','b','c','d','e','f','g','h','g'))
print(BOARD_FORMAT.format('x','o','x','o','x',' ',' ',' ',' '))
print(BOARD_FORMAT.format('x'.center(6),'o','x','o','x',' ',' ',' ',' '))


# ■ 매개 변수로 함수를 사용하는 경우
#  * return 뒤에 아무것도 적어주지 않으면 함수 종료
def stop_fun(num):
    for i in range(1, num+1):
        print('숫자 {0} 을 출력합니다'.format(i))
        if i == 5:
            return
stop_fun(10)
print('')
print('====================================================================================================')
print('== 문제 151. (점심시간 문제) 아래와 같이 숫자를 입력하고 함수를 실행하면 숫자가 세로로 출력되게 하시오.')
print('====================================================================================================')
def print_something(*args):
    for data in args:
        print(data)

print_something(1, 2, 3, 4, 5, 6, 7, 8)


# 5. 함수 밖의 변수, 함수 안의 변수
#  로컬 변수 ? 함수 내에서만 사용하는 변수
#  글로벌 변수 ? 함수 내,외 둘 다 사용 가능한 변수
#               특정함수에서 출력된 결과를 다른 함수에서 사용할 때 사용합니다.


print('')
print('====================================================================================================')
print('== 문제 152. 위의 스크립트에서 마지막 scope_test() 를 실행했을 때 a 가 1이 아니라 0 이 출력이 되려면')
print('==  scope_test() 함수를 생성할 때 어떻게 생성해야 했는가?')
print('====================================================================================================')
def scope_test():
    global a
    a = 1
    print('a : {0}'.format(a))
scope_test()


# * 스택 구조 : First in Last out
def some_func(count):
    if count > 0:
        some_func(count - 1)
    print(count)
print(some_func(10))


print('')
print('====================================================================================================')
print('== 문제 153. 10! 를 재귀함수로 구현해서 출력하시오!')
print('====================================================================================================')
def factorial2(num):
    if num > 1:
        num *= factorial2(num-1)
    return num

print(factorial2(10))


print('')
print('====================================================================================================')
print('== 문제 154. 16과 20의 최대공약수를 출력하는데 재귀함수를 이용해서 구현하시오!')
print('====================================================================================================')
def euclidean_gcd(num1, num2):
    rem = max(num1, num2) % min(num1, num2)
    if rem != 0:
        return euclidean_gcd(num1, rem)
    return min(num1, num2)

print(euclidean_gcd(108, 72))


print('')
print('====================================================================================================')
print('== 문제 155. 오늘 오전에 배운 가변 매개변수와 재귀 알고리즘을 이용해서 최대 공약수를 출력하는 함수를 생성하시오.')
print('====================================================================================================')
def prime_factorization_gcd(args):
    maxNum = max(args)
    for i in range(2, maxNum+1):
        cnt = 0
        for num in args:
            if (num % i) == 0:
                cnt += 1
        if len(args) == cnt:
            return i*prime_factorization_gcd([int(num/i) for num in args])
    return 1

print(prime_factorization_gcd([108, 72]))


# ■ 7.7. 중첩 함수
#  "파이썬에서는 함수 안에 함수를 정의하는 것이 가능하다"
#   중첩함수는 자신이 소속된 함수의 매개변수에 접근할 수 있다는 특징이 있다.
import math

def stddev(*args):
    def mean():
        return sum(args) / len(args)

    def variance(m):
        total = 0
        for arg in args:
            total += (arg - m) ** 2
        return total / (len(args)-1)

    v = variance(mean())
    return math.sqrt(v)

print(stddev(2.3, 1.7, 1.4, 0.7, 1.9))


print('')
print('====================================================================================================')
print('== 문제 156. 오늘 오전에 배운 가변 매개변수와 재귀 알고리즘을 이용해서 최대 공약수를 출력하는 함수를 생성하시오.')
print('====================================================================================================')
def list(*n):  # 가변 매개변수로 데이터를 여러개 입력받으면
    def gcd(a):  # 여러 수 중에서 최대공약수를 출력하는 알고리즘
        b = gcdtwo(max(a), min(a))  # 여러 수 중 두수를 뽑아서 최대공약수를 구하고
        # 다른 두수를 뽑아서 최대공약수를 구하고를 반복해서
        a.remove(min(a))  # 마지막에 남는 최대공약수가 전체의 최대공약수인 점을 이용
        a.remove(max(a))  # 정렬할 필요가 없게 전체 수에서 최대, 최소값을 뽑아서
        # 최대공약수를 구하는데 계산에 사용한 수는 제거하고
        a.append(b)  # 위에서 구한 최대공약수를 리스트에 추가
        if max(a) == min(a):  # 위 과정을 재귀를 통해 반복하면 최대공약수만 2개 남는데
            print('최대공약수는 : ', a[0])  # 그경우에서 재귀를 종료하고 최대공약수를 출력
        else:
            gcd(a)

    def gcdtwo(a, b):  # 두 수의 최대공약수를 출력
        # 분모가 0일경우 에러가 발생하므로 0인 경우를 따로 생각
        if min(a, b) == 0:  # 0과 A의 최대공약수는 무조건 A 이기 떄문에
            return max(a, b)  # 두 수중 최소값이 0인경우 두 수중 맥스값으로 최대공약수 출력
        return gcdtwo(b, a % b)  # 0이 아닌경우에 대해 유클리드호제법으로 재귀

    # 튜플 형태이기 때문에 데이터 변경이 불가능
    a = []  # 리스트를 생성하고 데이터 변경이 가능하도록
    for i in n:  # 튜플 데이터를 잘라서 리스트에 입력
        a.append(i)
    gcd(a)  # 최종적으로 생성한 리스트 변수를
    # 위에 생성한 최대공약수 함수에 입력해서 최대공약수 계산

list(1000, 500, 250, 100, 25, 25)


# 1. greedy (탐욕) 알고리즘
#  당장 눈 앞의 이익만 추구하는 것
#  먼 미래를 내다 보지 않고 지금 당장의 최선이 무엇인가만 판단


print('')
print('====================================================================================================')
print('== 문제 157. 탐욕 알고리즘을 이용하여 금액과 화폐가 주어졌을 때 가장 적은 화폐로 지불하시오!')
print('====================================================================================================')
def greedy_func(money, cash_type):
    if len(cash_type) != 0:
        print(str(cash_type[0]) + '원 : ' + str(int(money/cash_type[0])) + '개')
        money = int(money % cash_type[0])
        cash_type.remove(cash_type[0])
        greedy_func(money, cash_type)

money = int(input('액수를 입력하세요 : '))
cash_type = [int(type) for type in input('화폐 단위를 입력하세요 : ').split(' ')]
cash_type.sort(reverse=True)
greedy_func(money, cash_type)