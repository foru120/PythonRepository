
from math import *
import random
import copy
import time

BOARD_FORMAT = "|-|---|1|------|2|------|3|------|4|------|5|------|6|------|7|------|8|------|9|---\n" \
               "|1| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} |\n|" \
               "-|---------------------------------------------------------------------------------\n" \
               "|2| {9} | {10} | {11} | {12} | {13} | {14} | {15} | {16} | {17} |\n|" \
               "-|---------------------------------------------------------------------------------\n" \
               "|3| {18} | {19} | {20} | {21} | {22} | {23} | {24} | {25} | {26} |\n|" \
               "-|---------------------------------------------------------------------------------\n" \
               "|4| {27} | {28} | {29} | {30} | {31} | {32} | {33} | {34} | {35} |\n|" \
               "-|---------------------------------------------------------------------------------\n" \
               "|5| {36} | {37} | {38} | {39} | {40} | {41} | {42} | {43} | {44} |\n|" \
               "-|---------------------------------------------------------------------------------\n" \
               "|6| {45} | {46} | {47} | {48} | {49} | {50} | {51} | {52} | {53} |\n|" \
               "-|---------------------------------------------------------------------------------\n" \
               "|7| {54} | {55} | {56} | {57} | {58} | {59} | {60} | {61} | {62} |\n|" \
               "-|---------------------------------------------------------------------------------\n" \
               "|8| {63} | {64} | {65} | {66} | {67} | {68} | {69} | {70} | {71} |\n|" \
               "-|---------------------------------------------------------------------------------\n" \
               "|9| {72} | {73} | {74} | {75} | {76} | {77} | {78} | {79} | {80} |\n|" \
               "-|---------------------------------------------------------------------------------"
NAMES = [' ', 'X', 'O']

# 수 계산에 사용할 조건인 승리(end) 조건을 담는 부분
end=[]

# 가로 직선
for a in range(0,5,1):
    for b in range(a,a+73,9):
        end.append(tuple([b,b+1,b+2,b+3,b+4]))
# print(len(end))

# 세로 직선
for c in range(0,45,1):
    end.append(tuple([c,c+9,c+18,c+27,c+36]))
# print(len(end))

# 대각선 ↘
for d in range(0,37,9):
    for e in range(d,d+5,1):
        end.append(tuple([e,e+10,e+20,e+30,e+40]))
# print(len(end))

# 대각선 ↙
for i in range(4,41,9):
    for j in range(i,i+5,1):
        end.append(tuple([j,j+8,j+16,j+24,j+32]))

class TicTacToe:
    def __init__(self, state=None):
        if state is None:
            state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.playerJustMoved = 2
        self.state = state

    def Clone(self):                # 판 상태를 복사해서 지속적으로 게임을 진행하게 복사를 하는 함수
        state = TicTacToe()         # TicTacToe 클래스를 state에 담는다
        state.state = self.state[:] # state의 상태가 self.state
        state.playerJustMoved = self.playerJustMoved
        return state

    def DoMove(self, move):         # 턴을 정하는 함수
        assert int(move) >= 0 and int(move) <= 80 and self.state[move] == 0     # assert : 뒤 조건이 아니면
        self.playerJustMoved = 3 - self.playerJustMoved     # 턴을 넘기는 곳. 플레이어 1이 두면 3-self.playerJustMoved는 1이므로 새 self.playerJustMoved에 2가 저장된다
        self.state[move] = self.playerJustMoved             # 그렇다면 수를 둘(state[move]) 플레이어의 값이 2로 저장되어 턴이 넘어 간다.

    def GetMoves(self):             # 수를 둘 수 있는 빈칸을 찾아서 그 칸의 자리가 입력되면 그 자리에 수를 둘 수 있게 하는 함수
        if self.checkState() != 0:  # checkstate가 0이 아닌경우 빈 리스트를 반환한다.
            return []

        else:                       # 그게 아닌 모든 경우(checkstate가 값이 있을 경우) 빈 리스트 moves를 생성하고
            moves = []              # 0부터 80까지 숫자에 대해 self.state[i] 값이 0이면 해당 i값을 moves에 입력한다.
            for i in range(81):
                if self.state[i] == 0:
                    moves.append(i)

            return moves            # 그리고 moves를 반환한다.

    def GetResult(self, playerjm):
        result = self.checkState()
        assert result != 0
        if result == -1:
            return 0.5

        elif result == playerjm:
            return 2.0
        else:
            return 0.0

    def checkState(self):
        for (x, y, z, p, q) in end:
            if self.state[x] == self.state[y] == self.state[z] == self.state[p] == self.state[q]:
                if self.state[x] == 1:
                    return 1
                elif self.state[x] == 2:
                    return 2

        if [i for i in range(81) if self.state[i] == 0] == []:
            return -1
        return 0

    def __repr__(self):
        cells = []
        for i in range(81):
            cells.append(NAMES[self.state[i]].center(6))
        return BOARD_FORMAT.format(*cells)


class Node:
    def __init__(self, move=None, parent=None, state=None):
        self.move = move
        self.parentNode = parent
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.untriedMoves = state.GetMoves()
        self.playerJustMoved = state.playerJustMoved

    def UCTSelectChild(self):
        s = sorted(self.childNodes, key=lambda c: c.wins / c.visits + sqrt(2 * log(self.visits) / c.visits))
        return s[-1]

    def AddChild(self, m, s):
        n = Node(move=m, parent=self, state=copy.deepcopy(s))
        self.untriedMoves.remove(m)
        self.childNodes.append(n)
        return n

    def Update(self, result):
        self.visits += 1
        self.wins += result

    def __repr__(self):
        return "[M" + str(self.move) + " W/V " + str(self.wins) + "/" + str(self.visits) + " U" + str(
            self.untriedMoves) + "]"

    def ChildrenToString(self):
        s = ""
        for c in self.childNodes:
            s += str(c) + "\n"
        return s


def UCT(rootstate, itermax):
    rootnode = Node(state=rootstate)
    sTime = time.time()
    for i in range(itermax):
        node = rootnode
        state = copy.deepcopy(rootstate)

        # selection
        while node.untriedMoves == [] and node.childNodes != []:
            node = node.UCTSelectChild()
            state.DoMove(node.move)

        # Expansion
        if node.untriedMoves != []:
            m = random.choice(node.untriedMoves)
            state.DoMove(m)
            node = node.AddChild(m, state)

        # simulation
        while state.GetMoves() != []:
            state.DoMove(random.choice(state.GetMoves()))

        # BackPropagation
        while node != None:
            node.Update(state.GetResult(node.playerJustMoved))
            node = node.parentNode

    print(rootnode.ChildrenToString())
    eTime = time.time()
    print('AI가 수를 계산하는데 걸린 시간 : ', eTime - sTime)
    s = sorted(rootnode.childNodes, key=lambda c: c.wins / c.visits)
    return sorted(s, key=lambda c: c.visits)[-1].move


def UCTPlayGame():
    state = TicTacToe()
    while state.GetMoves() != []:
        print(str(state))
        if state.playerJustMoved == 1:
            rootstate = copy.deepcopy(state)
            m = UCT(rootstate, itermax=10000)

        else:
            m1 = input("수를 둘 행(1-9)을 입력하세요 : ")
            m2 = input("수를 둘 열(1-9)을 입력하세요 : ")
            m=m1+m2
            m = int(m)
            m = m-10-int(m1)
        state.DoMove(m)

    if state.GetResult(state.playerJustMoved) == 2.0:
        print("Player " + str(state.playerJustMoved) + " Wins!!")

    elif state.GetResult(state.playerJustMoved) == 0.0:
        print("Player " + str(3 - state.playerJustMoved) + " Wins!!")

    else:
        print("Draw!!")


if __name__ == "__main__":
    UCTPlayGame()


# Assult
# http://egloos.zum.com/spaurh/v/4068754
# 프로그래머가 확인하고 싶은 조건을 걸어두고 그냥 실행만 하면 조건에 맞지 않는 값이 들어왔을 때, 프로그래머가 디버깅을 하지 않아도 오류가
# 발생한 위치를 정확하게 알 수 있다. 일반적으로 ASSERT는 다음과 같은 상황에서 사용한다.
#
# 1. 내가 작성한 메서드에 넘어온 매개 변수를 확인하고 싶을 때
# 2. 내가 호출한 메서드에서 반환한 값을 확인하고 싶을 때
# 3. 내가 호출하는 메서드의 매개 변수를 확인하고 싶을 때
#
# 첫번째, ''내가 작성한 메서드에 넘어온 매개 변수를 확인하고 싶을 때''가 ASSERT를 사용하는 가장 흔한 경우이다. 앞에서 소개한 예제 코드도 바로 이런 경우에 속한다고 볼 수 있다.
# 이 경우에는 대부분 메서드의 코드 시작 부분에서 오류가 발생할 수 있는 모든 상황을 검증하는  것이 일반적이다. 왜냐하면 시작 부분에서 완벽하게 검증하지 않고 ASSERT 코드가 분산되어
# 있다면 매개 변수로 넘어온 값이 내가 작성한 코드에 의해서 영향을 받게 되어 ASSERT가 실패할 수 있기 때문이다. 이런 경우에는 비록 ASSERT로 오류가 발생하는 위치를 찾았다고 하더라도
#  디버깅하기 위해서 처음부터 코드를 다시 살펴봐야 하기 때문이다. 따라서 매개 변수는 함수 시작 부분에서 확인한다라고 알아두시면 되겠다.
#
# 두번째 ''내가 작성한 메서드에서 반환한 값을 확인하고 싶은 경우''는 우리가 잊어버리기가 굉장히 쉽다. 하지만 첫번째 상황보다는 ASSERT를 해야 한다라는 인식면에서 볼때 더 많이 알려진 경우
# 라고 볼 수 있다. 사실 프로그래머가 되면서 가장 많이 듣는 말이 리턴 값을 검사해야 한다는 말이다.  리턴 값을 넘기지 않는 함수들도 많지만, 함수가 제대로 작성되어 있다면 적어도 성공 또는 실패라는
# 정도는 알려주어야 한다고 생각한다. 리턴 값이 없는 함수들(리턴 값이 void인 함수들)을 작성한 프로그래머는 거의 모든 상황에서 함수가 성공하며 이 함수를 사용하는 사람은 함수의 성공 여부에
# 관심을 가질 필요가 없다라고 생각하기 때문일 것이다. 하지만 그건 함수를 작성하는 사람의 생각일 뿐이지, 함수가 리턴값으로 성공이라고 알려준다고 해도 전혀 해가 될것이 없고 언젠가는 함수의
# 결과를 확인해야 하는 순간이 올 수 있기 때문에 될 수 있으면 리턴 값을 넘기도록 함수를 작성하는 것이 좋다고 생각한다. 결국 내가 호출하는 어떠한 함수도 실패할 수 있고 내가 원하는 대로 함수가
# 작동하는지 확인해야 할 필요가 있기 때문에 함수의 리턴값은 함수를 호출한 다음 곧바로 검사할 수 있도록 해야한다.
#
# 마지막으로 ''내가 호출하는 메서드의 매개 변수를 확인하고 싶을 때''에는 일반적으로 자주 일어나는  경우는 아니지만, 다른 사람이 작성한 함수가 잘 작동할 수 있도록 배려하는 차원에서 확인한다고
#  보면 되겠다. 하지만 내가 아무리 정확하게 검증한다고 하더라도 해당 함수를 작성한 사람이 함수  안에서 내가 전달한 매개 변수를 검증하는 것보다는 확실하지 못할 것이다. 결국 첫번째 경우를
# 모두가 지켜준다면 세번째 경우는 필요가 없다.