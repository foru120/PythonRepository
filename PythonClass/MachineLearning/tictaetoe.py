import csv
import random
import numpy as np

MODE = 1  # 0 : 학습, 1: Com vs Com
MATCH_CNT = 100
LEARNING_CNT = 100000
MATRIX = 5

FILE_LOC = "D:\\KYH\\02.PYTHON\\data\\TTT5_5_100000.csv"
EMPTY = 0
PLAYER_X = 1
PLAYER_O = 2
DRAW = 3
BOARD_FORMAT = "| "+"{:^6s} | "*MATRIX
BOARD_FORMAT = (BOARD_FORMAT+'\n')*MATRIX
BOARD_FORMAT = "---------"*MATRIX + "\n" + BOARD_FORMAT + "---------"*MATRIX
NAMES = [' ', 'X', 'O']

# state 판 출력하는 함수
def printboard(state):
    print(BOARD_FORMAT.format(*[NAMES[int(idx)] for idx in state.flatten()]))

# 비어있는 state 리턴하는 함수
def emptystate():
    return np.zeros((MATRIX, MATRIX))

# 게임이 끝났는지 아닌지 확인하는 함수
def gameover(state):
    check_cnt = 0

    for i in range(MATRIX):
        # 가로 체크
        if state[i][0] != EMPTY:
            for j in range(1, MATRIX):
                if state[i][0] == state[i][j]:
                    check_cnt += 1

            if check_cnt == (MATRIX-1):
                return state[i][0]
            check_cnt = 0

        # 세로 체크
        if state[0][i] != EMPTY:
            for j in range(1, MATRIX):
                if state[0][i] == state[j][i]:
                    check_cnt += 1

            if check_cnt == (MATRIX-1):
                return state[0][i]
            check_cnt = 0

    check_cnt = 0

    # 대각선 체크(↘)
    if state[0][0] != EMPTY:
        for i in range(1, MATRIX):
            if state[0][0] == state[i][i]:
                check_cnt += 1

        if check_cnt == (MATRIX-1):
            return state[0][0]
        check_cnt = 0

    # 대각선 체크(↙)
    if state[0][MATRIX-1] != EMPTY:
        for i in range(1, MATRIX):
            if state[0][MATRIX-1] == state[i][MATRIX-i]:
                check_cnt += 1

        if check_cnt == (MATRIX-1):
            return state[0][2]

    # 빈 공간이 있는지 체크
    for i in range(MATRIX):
        for j in range(MATRIX):
            if state[i][j] == EMPTY:
                return EMPTY

    return DRAW

class Common(object):
    def __init__(self, player, verbose=False, lossval=0, learning=True):
        self.values = {}
        self.player = player
        self.verbose = verbose
        self.lossval = lossval
        self.learning = learning
        self.epsilon = 0.1
        self.alpha = 0.99
        self.prevstate = None
        self.prevscore = 0
        self.count = 0
        self.rownum = 0
        self.gamenum = 0

    # 랜덤으로 수를 선택
    def random(self, state):
        available = []
        for i in range(MATRIX):
            for j in range(MATRIX):
                if state[i][j] == EMPTY:
                    available.append((i, j))
        return random.choice(available)

    def greedy(self, state):
        maxval = -50000
        maxmove = None
        if self.verbose:
            cells = []
        for i in range(MATRIX):
            for j in range(MATRIX):
                if state[i][j] == EMPTY:
                    state[i][j] = self.player
                    val = self.lookup(state)
                    state[i][j] = EMPTY
                    if val > maxval:
                        maxval = val
                        maxmove = (i, j)
                    if self.verbose:
                        cells.append('{0:.3f}'.format(val).center(6))
                elif self.verbose:
                    cells.append(NAMES[int(state[i][j])].center(6))
        if self.verbose:
            print(BOARD_FORMAT.format(*cells))

        return maxmove, maxval

    def lookup(self, state):
        key = self.statetuple(state)
        if not key in self.values:
            self.add(key)
        return self.values[key]

    def statetuple(self, state):
        return tuple(tuple(row) for row in state.tolist())

    def add(self, state):
        winner = gameover(state)
        self.values[state] = self.winnerval(winner)

    def winnerval(self, winner):
        if winner == self.player:
            return 1
        elif winner == EMPTY:
            return 0.5
        elif winner == DRAW:
            return 0
        else:
            return self.lossval


# 컴퓨터가 학습하는 클래스
class Agent(Common):
    def __init__(self, player, verbose=False, lossval=0, learning=True):
        super().__init__(player, verbose, lossval, learning)
        self.dataset = []

    def episode_over(self, winner):
        self.backup(self.winnerval(winner))
        self.prevstate = None
        self.prevscore = 0
        self.gamenum += 1

    def action(self, state):
        r = random.random()
        if r < self.epsilon:
            move = self.random(state)
            # self.log('>>>>>>> Exploratory action: ' + str(move))
        else:
            move, maxval = self.greedy(state)
            self.backup(maxval)
            # self.log('>>>>>>> Best action: ' + str(move))
        state[move[0]][move[1]] = self.player
        self.prevstate = self.statetuple(state)
        self.prevscore = self.lookup(state)
        state[move[0]][move[1]] = EMPTY
        return move

    def backup(self, nextval):
        if self.prevstate != None and self.learning:
            prevval = self.values[self.prevstate]
            self.values[self.prevstate] += self.alpha * (nextval - self.prevscore)
            temp = []
            for i in range(MATRIX):
                for j in range(MATRIX):
                    temp.append(self.prevstate[i][j])

            self.dataset.append(temp + [prevval, self.values[self.prevstate], self.player, self.gamenum, self.rownum])
            self.rownum += 1

    def datatofile(self):
        Fn = open(FILE_LOC, 'a')
        w = csv.writer(Fn, delimiter=',', lineterminator='\n')
        for data in self.dataset:
            w.writerow(data)
        Fn.close()
        self.dataset.clear()


# 컴퓨터가 게임하는 클래스
class Computer(Common):
    def __init__(self, player, filename):
        super().__init__(player, True)
        self.filename = filename
        self.readCSV()

    def readCSV(self):
        print(str(self.player) + ' 님의 데이터를 초기화 하는 중입니다...')
        file = open("D:\\KYH\\02.PYTHON\\data\\"+self.filename, 'r')
        ttt_list = csv.reader(file)
        for t in ttt_list:
            try:
                self.values[tuple(tuple(t[v] for v in range(sidx, sidx+MATRIX)) for sidx in range(0, MATRIX*MATRIX, MATRIX))] = float(t[MATRIX+1])
            except ValueError:
                continue

    # 컴퓨터가 착수
    def action(self, state):
        printboard(state)
        move, maxval = self.greedy(state)
        state[move[0]][move[1]] = self.player
        return move


def episode_over(winner):
    if winner == DRAW:
        print('Game over! It was a draw.')
    else:
        print('Game over! Winner: Player {0}'.format(winner))

def play(agent1, agent2):
    state = emptystate()
    for i in range(MATRIX*MATRIX):
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
    if MODE == 0:
        p1 = Agent(1, lossval=-1)
        p2 = Agent(2, lossval=-1)

        for i in range(LEARNING_CNT):
            if i % 10 == 0:
                print('Game: {0}'.format(i))

            winner = play(p1, p2)
            p1.episode_over(winner)
            p2.episode_over(winner)

            if (i != 0) and (i % 10000 == 0):
                p1.datatofile()
                p2.datatofile()
        p1.datatofile()
        p2.datatofile()
    elif MODE == 1:
        p1 = Computer(1, 'TTT5_5_10000.csv')
        p2 = Computer(2, 'TTT5_5_100000.csv')

        for i in range(MATCH_CNT):
            if i % 10 == 0:
                print('Game: {0}'.format(i))

            winner = play(p1, p2)
            episode_over(winner)