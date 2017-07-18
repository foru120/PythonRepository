from PythonClass.common_func import CommonFunc

# ■ 10장. 예외

# 10.1. 예외란?
#  "프로그램에서 에러가 발생했을 때, 에러를 핸들링하는 기능"

def my_power():
    x = input('분자 숫자를 입력하세요 ~ ')
    y = input('분모 숫자를 입력하세요 ~ ')
    return int(x) / int(y)

print(my_power())


print('')
print('====================================================================================================')
print('== 문제 179. 위의 코드에 예외처리 코드를 입혀서 분모로 0 을 입력하면 0 으로는 나눌 수 없습니다. 라는 메세지가 출력되게 하시오!')
print('====================================================================================================')
def my_power():
    try:
        x = input('분자 숫자를 입력하세요 ~ ')
        y = input('분모 숫자를 입력하세요 ~ ')
        return int(x) / int(y)
    except ZeroDivisionError:
        print('0 으로 나눌 수 없습니다.')

print(my_power())


print('')
print('====================================================================================================')
print('== 문제 180. 이름을 물어보게하고 이름을 입력하면 해당 사원의 월급이 출력되는 함수를 생성하시오.')
print('====================================================================================================')
from PythonClass.common_func import CommonFunc
def find_sal():
    ename = input('월급을 알고 싶은 사원명을 입력하세요 : ')

    for empData in CommonFunc.returnCsvData('emp2.csv'):
        if empData[1] == ename.upper():
            return empData[5]
    return None

print(find_sal())


print('')
print('====================================================================================================')
print('== 문제 181. 위의 코드에 exception 코드를 입혀서 없는 사원 이름을 입력하면 해당 사원은 없습니다. 라는 메세지가 출력되게 하시오!')
print('====================================================================================================')
import pandas as pd

def find_sal():
    ename = input('월급을 알고 싶은 사원명을 입력하세요 : ')
    empFrame = pd.read_csv('D:\\KYH\\02.PYTHON\\data\\emp.csv')

    try:
        return empFrame[['sal']][empFrame['ename'] == ename.upper()].values[0]
    except:
        print('해당 사원은 존재하지 않습니다.')

print(find_sal())


print('')
print('====================================================================================================')
print('== 문제 182. 이름을 물어보게 하는데 이름을 입력하지 않으면 다시 질문을 하게 하시오!')
print('====================================================================================================')
import pandas as pd

def find_sal():
    while True:
        ename = input('월급을 알고 싶은 사원명을 입력하세요 : ')
        if ename == '':
            continue

        empFrame = pd.read_csv('D:\\KYH\\02.PYTHON\\data\\emp.csv')

        try:
            return empFrame[['sal']][empFrame['ename'] == ename.upper()].values[0]
        except:
            print('해당 사원은 존재하지 않습니다.')
            return None

print(find_sal())


print('')
print('====================================================================================================')
print('== 문제 183. 위의 코드를 수정해서 없는 사원명을 입력하면 해당 사원은 없습니다. 라고 출력되게 하시오!')
print('====================================================================================================')
import pandas as pd

def find_sal():
    while True:
        ename = input('월급을 알고 싶은 사원명을 입력하세요 : ')
        if ename == '':
            continue

        empFrame = pd.read_csv('D:\\KYH\\02.PYTHON\\data\\emp.csv')

        try:
            return empFrame[['sal']][empFrame['ename'] == ename.upper()].values[0]
        except:
            print('해당 사원은 존재하지 않습니다.')
            return None

print(find_sal())


print('')
print('====================================================================================================')
print('== 문제 184. 직업을 물어보게하고 직업을 입력하면 해당 직업의 토탈월급이 출력되게하는데 아무것도 입력하지 않으면')
print('==  계속 물어보게하고 잘못된 직업명을 입력하면 해당 직업은 없습니다. 라는 메세지가 출력되게 하시오.')
print('====================================================================================================')
import pandas as pd

def tot_sal():
    while True:
        job = input('직업을 입력하세요 : ')
        if job == '':
            continue

        empFrame = pd.read_csv('D:\\KYH\\02.PYTHON\\data\\emp.csv')

        try:
            result = empFrame[['sal', 'job']][empFrame['job'] == job.upper()]
            return result.groupby(['job'])['sal'].sum()
        except:
            print('해당 직업은 존재하지 않습니다.')
            return None

print(tot_sal())


# 10.3. 복수개의 except 절 사용하기
def my_power():
    try:
        x = input('분자 숫자를 입력하세요 : ')
        y = input('분자 숫자를 입력하세요 : ')
        return int(x) / int(y)
    except ZeroDivisionError as err:
        print('0 으로 나눌 수 없습니다.', err)
    except:
        print('다른 예외입니다.')

print(my_power())


# 10.4. try 절을 무사히 실행하면 만날 수 있는 else
print('')
print('====================================================================================================')
print('== 문제 185. 이름을 물어보게 하고 해당 사원의 월급이 출력되게 하는데 이름이 없으면 해당 사원은 없습니다. 라는')
print('==  메세지가 나오게하고 만약 있어서 성공했다면 월급 추출에 성공했습니다. 라는 메세지가 출력되게 하시오.')
print('====================================================================================================')
import pandas as pd

def find_sal2():
    try:
        emp = pd.read_csv('D:\\KYH\\02.PYTHON\\data\\emp.csv')
        name = ''
        while name == '':
            name = input('월급을 알고 싶은 사원명을 입력하세요 : ')
        sal = emp['sal'][emp['ename'] == name.upper()].values[0]
        return sal
    except Exception as err:
        print('해당 사원은 없습니다.')
    finally:
        print('월급 추출에 성공했습니다.')

print(find_sal2())


print('')
print('====================================================================================================')
print('== 문제 186. 방금 사용한 else 를 이용해서 아래의 나눈값을 출력되게하는데 두 수를 물어보게 하고 나눈 값을 출력할때')
print('==  정상적으로 나눠지면 나눈값을 잘 추출했습니다가 출력되게하고 0 으로 나누면 0 으로 나눌수 없습니다가 출력되게 하시오!')
print('====================================================================================================')
def my_power():
    try:
        x = input('분자 숫자를 입력하세요 : ')
        y = input('분자 숫자를 입력하세요 : ')
        num = int(x) / int(y)
    except ZeroDivisionError as err:
        print('0 으로 나눌 수 없습니다.', err)
    else:
        print('나눈값을 잘 추출했습니다.')
        return num

print(my_power())


# 10.4. 어떤일이 있어도 반드시 실행되는 finally
def my_power():
    try:
        x = input('분자 숫자를 입력하세요 : ')
        y = input('분자 숫자를 입력하세요 : ')
        num = int(x) / int(y)
        return num
    except ZeroDivisionError as err:
        print('0 으로 나눌 수 없습니다.', err)
    finally:
        print('저는 무조건 수행됩니다.')

print(my_power())


# 10.5. Exception 클래스
#  "클래스의 상속관계에서 자식 클래스는 부모 클래스로 간주할 수 있다는 특징으로 인해서 exception 클래스에 대한
#   예외 처리절이 다른 예외 처리절에 앞서 위치하면 나머지 예외 처리절은 모두 무시된다."
def my_power():
    try:
        x = input('분자 숫자를 입력하세요 : ')
        y = input('분자 숫자를 입력하세요 : ')
        num = int(x) / int(y)
        return num
    except Exception as err:
        print('예외가 발생했습니다.', err)
    except ZeroDivisionError as err:
        print('0 으로 나눌 수 없습니다.', err)
    finally:
        print('저는 무조건 수행됩니다.')

print(my_power())


# ■ NotImplementedError 예외사항 테스트
#  "추상 클래스를 상속받았을 때 처럼 상위 클래스에 NotImplementedError 예외가 정의되어있으면 상속받은 자식 클래스에서는
#   반드시 오버라이딩을 해야 에러가 안나게 강제하는 예외 사항"
class Bird:
    def fly(self):
        raise NotImplementedError

class Eagle(Bird):
    pass

eagle = Eagle()
eagle.fly()


print('')
print('====================================================================================================')
print('== 문제 187. eagle.fly() 를 실행할 때 에러가 안나고 very fast 라는 말이 출력될 수 있도록 오버라이딩 하시오.')
print('====================================================================================================')
class Bird:
    def fly(self):
        raise NotImplementedError

class Eagle(Bird):
    def fly(self):
        print('very fast')

eagle = Eagle()
eagle.fly()


# 10.7. 사용자 정의 예외 처리
#  "파이썬 입장에서 봤을 때는 오류가 아닌데 프로그래머가 이건 오류이다라고 raise 문을 써서 예외처리를 하는 경우"
print('')
print('====================================================================================================')
print('== 문제 188. 이름을 입력하면 월급을 출력하는 함수를 만드는데 월급이 3000 이상인 사원들은 해당 사원의 월급을 볼 수 없습니다.')
print('==  라는 에러 메세지가 출력되게 하시오.')
print('====================================================================================================')
import pandas as pd
def find_sal():
    ename = input('이름을 입력하세요 : ')
    empFrame = pd.read_csv('D:\\KYH\\02.PYTHON\\data\\emp.csv')
    sal = empFrame['sal'][empFrame['ename'] == ename.upper()].values[0]

    if sal >= 3000:
        raise Exception('해당 사원의 월급을 볼 수 없습니다.')
    else:
        return sal

print(find_sal())


print('')
print('====================================================================================================')
print('== 문제 189. 1부터 9사이에 숫자를 받게해서 해당 숫자를 출력하는 함수를 생성하는데 1부터 9사이가 아니면 잘 못 입력하였습니다.')
print('==  라는 에러 메세지가 출력되게 하는 함수를 생성하시오.')
print('====================================================================================================')
def get_number():
    num = int(input('숫자를 입력하세요 : '))
    if 1 <= num <= 9:
        return num
    else:
        raise Exception('잘못 입력하였습니다.')

print(get_number())


print('')
print('====================================================================================================')
print('== 문제 190. 위의 코드를 수정해서 1번부터 9번 사이에 숫자를 입력하지 않았으면 숫자를 다시 물어보게 하시오.')
print('====================================================================================================')
def get_number():
    while True:
        num = int(input('숫자를 입력하세요 : '))
        if 1 <= num <= 9:
            return num
        else:
            continue

print(get_number())


print('')
print('====================================================================================================')
print('== 문제 191. 머신 러닝화 하지 않은 MIT 코드 ttt를 수행해서 숫자를 1~9번 외의 번호를 입력하세요.')
print('====================================================================================================')
import random
from copy import copy, deepcopy
# deepcopy : 메모리를 완전히 새롭게 생성
# copy : 껍데기만 카피, 내용은 동일한 곳을 가리킴
EMPTY = 0
PLAYER_X = 1
PLAYER_O = 2
DRAW = 3
BOARD_FORMAT = "----------------------------\n| {0} | {1} | {2} |\n|--------------------------|\n| {3} | {4} | {5} |\n|--------------------------|\n| {6} | {7} | {8} |\n----------------------------"
NAMES = [' ', 'X', 'O']

# 보드 출력
def printboard(state):
    cells = []
    for i in range(3):
        for j in range(3):
            cells.append(NAMES[state[i][j]].center(6))
    print(BOARD_FORMAT.format(*cells))

# 빈 판
def emptystate():
    return [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]

def gameover(state):
    # 가로/세로로 한 줄 완성한 플레이어가 있다면 그 플레이어 리턴
    for i in range(3):
        if state[i][0] != EMPTY and state[i][0] == state[i][1] and state[i][0] == state[i][2]:
            return state[i][0]
        if state[0][i] != EMPTY and state[0][i] == state[1][i] and state[0][i] == state[2][i]:
            return state[0][i]
    # 좌우 대각선
    if state[0][0] != EMPTY and state[0][0] == state[1][1] and state[0][0] == state[2][2]:
        return state[0][0]
    if state[0][2] != EMPTY and state[0][2] == state[1][1] and state[0][2] == state[2][0]:
        return state[0][2]
    # 판이 비었는지
    for i in range(3):
        for j in range(3):
            if state[i][j] == EMPTY:
                return EMPTY
    return DRAW
# 사람
class Human(object):
    def __init__(self, player):
        self.player = player
    # 착수
    def action(self, state):
        printboard(state)
        action = None
        while action not in range(1, 10):
            action = int(input('Your move? '))
        switch_map = {
            1: (0, 0),
            2: (0, 1),
            3: (0, 2),
            4: (1, 0),
            5: (1, 1),
            6: (1, 2),
            7: (2, 0),
            8: (2, 1),
            9: (2, 2)
        }
        return switch_map[action]
    def episode_over(self, winner):
        if winner == DRAW:
            print('Game over! It was a draw.')
        else:
            print('Game over! Winner: Player {0}'.format(winner))
def play(agent1, agent2):
    state = emptystate()
    for i in range(9):
        if i % 2 == 0:
            move = agent1.action(state)
        else:
            move = agent2.action(state)
        state[move[0]][move[1]] = (i % 2) + 1
        winner = gameover(state)
        if winner != EMPTY:
            return winner
    return winner
if __name__ == "__main__":
    p1 = Human(1)
    p2 = Human(2)
    while True:
        winner = play(p1, p2)
        p1.episode_over(winner)
        p2.episode_over(winner)


print('')
print('====================================================================================================')
print('== 문제 192. 이번에는 1번부터 9번 사이외에 숫자를 넣으면 다시 물어보게 할 뿐만 아니라 입력을 안해도 다시 물어보게 하시오.')
print('====================================================================================================')
def action(self, state):
    printboard(state)
    action = None
    while action not in range(1, 10) and action != '':
        action = int(input('Your move? '))
    switch_map = {
        1: (0, 0),
        2: (0, 1),
        3: (0, 2),
        4: (1, 0),
        5: (1, 1),
        6: (1, 2),
        7: (2, 0),
        8: (2, 1),
        9: (2, 2)
    }

    return switch_map[action]


print('')
print('====================================================================================================')
print('== 문제 193. 생각해야할 문제에 경원이가 올린 코드를 수정해서 아무것도 입력 안했을 때 다시 물어보게 하시오')
print('====================================================================================================')
def action(self, state):
    printboard(state)
    action = None
    while action not in range(1, 10):
        try:
            action = int(input('Your move?'))
        except ValueError:
            continue
    switch_map = {
        1: (0, 0),
        2: (0, 1),
        3: (0, 2),
        4: (1, 0),
        5: (1, 1),
        6: (1, 2),
        7: (2, 0),
        8: (2, 1),
        9: (2, 2)
    }

    return switch_map[action]