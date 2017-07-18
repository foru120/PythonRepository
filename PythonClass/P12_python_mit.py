# ■ 12장. Mit TTT 코드 분석
#  1. Tic Tac Toe 프로그램을 만들기 위해서 필요한 기능
print('')
print('====================================================================================================')
print('== 문제 194. print 를 찍어서 디버깅하시오.')
print('====================================================================================================')
# EMPTY = 0
# PLAYER_X = 1
# PLAYER_O = 2
# DRAW = 3
# BOARD_FORMAT = "----------------------------\n| {0} | {1} | {2} |\n|--------------------------|\n| {3} | {4} | {5} |\n|--------------------------|\n| {6} | {7} | {8} |\n----------------------------"
# NAMES = [' ', 'X', 'O']
#
# state = [[1, 2, 0], [0, 0, 0], [0, 0, 0]]

# 보드 출력
# def printboard(state):
#     cells = []
#     for i in range(3):
#         for j in range(3):
#             cells.append(NAMES[state[i][j]].center(6))
#     print(BOARD_FORMAT.format(*cells))
#
# print(printboard(state))


print('')
print('====================================================================================================')
print('== 문제 195. 함수의 매개변수로 함수를 사용할 수 있다고 했다. 그러므로 printboard 에 매개변수로 emptystate()')
print('==  함수를 사용하면 결과가 어떻게 출력되는지 결과를 출력해보세요.')
print('====================================================================================================')
# state = [[1, 2, 0], [0, 0, 0], [0, 0, 0]]

# 보드 출력
# def printboard(state):
#     cells = []
#     for i in range(3):
#         for j in range(3):
#             cells.append(NAMES[state[i][j]].center(6))
#     print(BOARD_FORMAT.format(*cells))
#
# # 빈 판
# def emptystate():
#     return [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]
#
# def gameover(state):
#     # 가로/세로로 한 줄 완성한 플레이어가 있다면 그 플레이어 리턴
#     for i in range(3):
#         if state[i][0] != EMPTY and state[i][0] == state[i][1] and state[i][0] == state[i][2]:  # 가로
#             return state[i][0]
#         if state[0][i] != EMPTY and state[0][i] == state[1][i] and state[0][i] == state[2][i]:  # 세로
#             return state[0][i]
#     # 좌우 대각선
#     if state[0][0] != EMPTY and state[0][0] == state[1][1] and state[0][0] == state[2][2]:
#         return state[0][0]
#     if state[0][2] != EMPTY and state[0][2] == state[1][1] and state[0][2] == state[2][0]:
#         return state[0][2]
#     # 판이 비었는지
#     for i in range(3):
#         for j in range(3):
#             if state[i][j] == EMPTY:
#                 return EMPTY
#     return DRAW

# printboard(emptystate())


print('')
print('====================================================================================================')
print('== 문제 196. 아래의 7번째 자리에 x 를 찍어주고 리셋되게 하시오!')
print('====================================================================================================')
import random
from copy import copy, deepcopy

# deepcopy : 메모리를 완전히 새롭게 생성
# copy : 껍데기만 카피, 내용은 동일한 곳을 가리킴
# 보드 출력
# def printboard(state):
#     cells = []
#     for i in range(3):
#         for j in range(3):
#             cells.append(NAMES[state[i][j]].center(6))
#     print(BOARD_FORMAT.format(*cells))


# 빈 판
# def emptystate():
#     return [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]
#
#
# def gameover(state):
#     # 가로/세로로 한 줄 완성한 플레이어가 있다면 그 플레이어 리턴
#     for i in range(3):
#         if state[i][0] != EMPTY and state[i][0] == state[i][1] and state[i][0] == state[i][2]:
#             printboard(state)
#             return state[i][0]
#         if state[0][i] != EMPTY and state[0][i] == state[1][i] and state[0][i] == state[2][i]:
#             printboard(state)
#             return state[0][i]
#     # 좌우 대각선
#     if state[0][0] != EMPTY and state[0][0] == state[1][1] and state[0][0] == state[2][2]:
#         printboard(state)
#         return state[0][0]
#     if state[0][2] != EMPTY and state[0][2] == state[1][1] and state[0][2] == state[2][0]:
#         printboard(state)
#         return state[0][2]
#     # 판이 비었는지
#     for i in range(3):
#         for j in range(3):
#             if state[i][j] == EMPTY:
#                 return EMPTY
#     return DRAW


# action : 어디에 수를 둘지 물어보는 함수
# def action():
#     # printboard(state)
#     action = None
#     while action not in range(1, 10):
#         action = int(input('Your move? '))
#     switch_map = {
#         1: (0, 0),
#         2: (0, 1),
#         3: (0, 2),
#         4: (1, 0),
#         5: (1, 1),
#         6: (1, 2),
#         7: (2, 0),
#         8: (2, 1),
#         9: (2, 2)
#     }
#     return switch_map[action]
#
# print(action())


print('')
print('====================================================================================================')
print('== 문제 197. 아무것도 입력하지 않으면 계속 물어보게 하시오!')
print('====================================================================================================')
# def action():
#     # printboard(state)
#     action = None
#     while action not in range(1, 10):
#         try:
#             action = int(input('Your move? '))
#         except ValueError:
#             continue
#     switch_map = {
#         1: (0, 0),
#         2: (0, 1),
#         3: (0, 2),
#         4: (1, 0),
#         5: (1, 1),
#         6: (1, 2),
#         7: (2, 0),
#         8: (2, 1),
#         9: (2, 2)
#     }
#     return switch_map[action]
#
# print(action())


# episode_over : 게임 종료시 누가 이겼는지 비겼는지 메세지를 출력하는 함수
# def episode_over(winner):
#     if winner == DRAW:
#         print('Game over! It was a draw.')
#     else:
#         print('Game over! Winner: Player {0}'.format(winner))
#
# print(episode_over(1))


# play : 실제로 게임을 진행하는 함수, 게임 종료 여부를 확인하고 결과를 리턴하는 함수
# def play(agent1, agent2):
#     state = emptystate()
#     for i in range(9):
#         if i % 2 == 0:
#             move = agent1.action(state)
#         else:
#             move = agent2.action(state)
#         state[move[0]][move[1]] = (i % 2) + 1
#         winner = gameover(state)
#         if winner != EMPTY:
#             return winner
#     return winner


print('')
print('====================================================================================================')
print('== 문제 198. 아래 코드를 이용해서 main 함수를 생성하는데 무한 루프가 게임횟수를 아래와 같이 지정해서 수행하게 하시오.')
print('====================================================================================================')
# 사람
# class Human(object):
#     def __init__(self, player):
#         self.player = player
#
#     # 착수
#     def action(self, state):
#         printboard(state)
#         action = None
#         while action not in range(1, 10):
#             action = int(input('Your move? '))
#         switch_map = {
#             1: (0, 0),
#             2: (0, 1),
#             3: (0, 2),
#             4: (1, 0),
#             5: (1, 1),
#             6: (1, 2),
#             7: (2, 0),
#             8: (2, 1),
#             9: (2, 2)
#         }
#         return switch_map[action]
#
#     def episode_over(self, winner):
#         if winner == DRAW:
#             print('Game over! It was a draw.')
#         else:
#             print('Game over! Winner: Player {0}'.format(winner))


# def play(agent1, agent2):
#     state = emptystate()
#     for i in range(9):
#         if i % 2 == 0:
#             move = agent1.action(state)
#         else:
#             move = agent2.action(state)
#         state[move[0]][move[1]] = (i % 2) + 1
#         winner = gameover(state)
#         if winner != EMPTY:
#             return winner
#     return winner


# class Computer(object):
#     def __init__(self, player):
#         self.player = player
#
#     def random(self, state):
#         available = []
#         for i in range(3):
#             for j in range(3):
#                 if state[i][j] == EMPTY:
#                     available.append((i, j))
#         return random.choice(available)
#
#     # 컴퓨터가 착수
#     def action(self, state):
#         printboard(state)
#         action = None
#         move = self.random(state)
#         state[move[0]][move[1]] = self.player
#         return move
#
#     def episode_over(self, winner):
#         if winner == DRAW:
#             print('Game over! It was a draw.')
#         else:
#             print('Game over! Winner: Player {0}'.format(winner))
#
# def main(cnt)
#     p1 = Human(1)
#     p2 = Computer(2)
#     while cnt >= 0:
#         winner = play(p1, p2)
#         p1.episode_over(winner)
#         p2.episode_over(winner)
#         cnt -= 1


print('')
print('====================================================================================================')
print('== 문제 199. 사람과 컴퓨터와 게임을 할 수 있게 하시오!')
print('====================================================================================================')
import random
from copy import copy, deepcopy

# deepcopy : 메모리를 완전히 새롭게 생성
# copy : 껍데기만 카피, 내용은 동일한 곳을 가리킴

# 보드 출력
# def printboard(state):
#     cells = []
#     for i in range(3):
#         for j in range(3):
#             cells.append(NAMES[state[i][j]].center(6))
#     print(BOARD_FORMAT.format(*cells))
#
#
# # 빈 판
# def emptystate():
#     return [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]
#
#
# def gameover(state):
#     # 가로/세로로 한 줄 완성한 플레이어가 있다면 그 플레이어 리턴
#     for i in range(3):
#         if state[i][0] != EMPTY and state[i][0] == state[i][1] and state[i][0] == state[i][2]:
#             return state[i][0]
#         if state[0][i] != EMPTY and state[0][i] == state[1][i] and state[0][i] == state[2][i]:
#             return state[0][i]
#     # 좌우 대각선
#     if state[0][0] != EMPTY and state[0][0] == state[1][1] and state[0][0] == state[2][2]:
#         return state[0][0]
#     if state[0][2] != EMPTY and state[0][2] == state[1][1] and state[0][2] == state[2][0]:
#         return state[0][2]
#     # 판이 비었는지
#     for i in range(3):
#         for j in range(3):
#             if state[i][j] == EMPTY:
#                 return EMPTY
#     return DRAW
#
#
# # 사람
# class Human(object):
#     def __init__(self, player):
#         self.player = player
#
#     # 착수
#     def action(self, state):
#         printboard(state)
#         action = None
#         while action not in range(1, 10):
#             action = int(input('Your move? '))
#         switch_map = {
#             1: (0, 0),
#             2: (0, 1),
#             3: (0, 2),
#             4: (1, 0),
#             5: (1, 1),
#             6: (1, 2),
#             7: (2, 0),
#             8: (2, 1),
#             9: (2, 2)
#         }
#         return switch_map[action]
#
#     def episode_over(self, winner):
#         if winner == DRAW:
#             print('Game over! It was a draw.')
#         else:
#             print('Game over! Winner: Player {0}'.format(winner))
#
#
# def play(agent1, agent2):
#     state = emptystate()
#     for i in range(9):
#         if i % 2 == 0:
#             move = agent1.action(state)
#         else:
#             move = agent2.action(state)
#         state[move[0]][move[1]] = (i % 2) + 1
#         winner = gameover(state)
#         if winner != EMPTY:
#             return winner
#     return winner
#
#
# class Computer(object):
#     def __init__(self, player):
#         self.player = player
#
#     def random(self, state):
#         available = []
#         for i in range(3):
#             for j in range(3):
#                 if state[i][j] == EMPTY:
#                     available.append((i, j))
#         return random.choice(available)
#
#     # 컴퓨터가 착수
#     def action(self, state):
#         printboard(state)
#         action = None
#         move = self.random(state)
#         state[move[0]][move[1]] = self.player
#         return move
#
#     def episode_over(self, winner):
#         if winner == DRAW:
#             print('Game over! It was a draw.')
#         else:
#             print('Game over! Winner: Player {0}'.format(winner))
#
#
# if __name__ == "__main__":
#     p1 = Human(1)
#     p2 = Computer(2)
#     while True:
#         winner = play(p1, p2)
#         p1.episode_over(winner)
#         p2.episode_over(winner)


print('')
print('====================================================================================================')
print('== 문제 201. 컴퓨터(랜덤)와 컴퓨터(랜덤)과의 대결의 게임 진행 데이터를 D 드라이브 밑에 test200.csv 로 생성되게 하시오!')
print('====================================================================================================')
# import random
# import csv
# from copy import copy, deepcopy
# EMPTY = 0
# PLAYER_X = 1
# PLAYER_O = 2
# DRAW = 3
# BOARD_FORMAT = "----------------------------\n| {0} | {1} | {2} |\n|--------------------------|\n| {3} | {4} | {5} |\n|--------------------------|\n| {6} | {7} | {8} |\n----------------------------"
# NAMES = [' ', 'X', 'O']
#
# # deepcopy : 메모리를 완전히 새롭게 생성
# # copy : 껍데기만 카피, 내용은 동일한 곳을 가리킴
#
# # 보드 출력
# def printboard(state):
#     cells = []
#     for i in range(3):
#         for j in range(3):
#             cells.append(NAMES[state[i][j]].center(6))
#     print(BOARD_FORMAT.format(*cells))
#
#
# # 빈 판
# def emptystate():
#     return [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]
#
#
# def gameover(state):
#     # 가로/세로로 한 줄 완성한 플레이어가 있다면 그 플레이어 리턴
#     for i in range(3):
#         if state[i][0] != EMPTY and state[i][0] == state[i][1] and state[i][0] == state[i][2]:
#             return state[i][0]
#         if state[0][i] != EMPTY and state[0][i] == state[1][i] and state[0][i] == state[2][i]:
#             return state[0][i]
#     # 좌우 대각선
#     if state[0][0] != EMPTY and state[0][0] == state[1][1] and state[0][0] == state[2][2]:
#         return state[0][0]
#     if state[0][2] != EMPTY and state[0][2] == state[1][1] and state[0][2] == state[2][0]:
#         return state[0][2]
#     # 판이 비었는지
#     for i in range(3):
#         for j in range(3):
#             if state[i][j] == EMPTY:
#                 return EMPTY
#     return DRAW
#
#
# # 사람
# class Human(object):
#     def __init__(self, player):
#         self.player = player
#
#     # 착수
#     def action(self, state):
#         printboard(state)
#         action = None
#         while action not in range(1, 10):
#             action = int(input('Your move? '))
#         switch_map = {
#             1: (0, 0),
#             2: (0, 1),
#             3: (0, 2),
#             4: (1, 0),
#             5: (1, 1),
#             6: (1, 2),
#             7: (2, 0),
#             8: (2, 1),
#             9: (2, 2)
#         }
#         return switch_map[action]
#
#     def episode_over(self, winner):
#         if winner == DRAW:
#             print('Game over! It was a draw.')
#         else:
#             print('Game over! Winner: Player {0}'.format(winner))
#
#
# def play(agent1, agent2):
#     state = emptystate()
#     for i in range(9):
#         if i % 2 == 0:
#             move = agent1.action(state)
#         else:
#             move = agent2.action(state)
#         state[move[0]][move[1]] = (i % 2) + 1
#         winner = gameover(state)
#         if winner != EMPTY:
#             return winner
#     return winner
#
#
# class Computer(object):
#     def __init__(self, player):
#         self.player = player
#
#     def random(self, state):
#         available = []
#         for i in range(3):
#             for j in range(3):
#                 if state[i][j] == EMPTY:
#                     available.append((i, j))
#         return random.choice(available)
#
#     # 컴퓨터가 착수
#     def action(self, state):
#         printboard(state)
#         action = None
#         move = self.random(state)
#         state[move[0]][move[1]] = self.player
#         Fn = ('D:\\KYH\\02.PYTHON\\data\\Test200.csv')
#         w = csv.writer(open(Fn, 'a'), delimiter=',')
#         w.writerow([state[0][0],
#                     state[0][1],
#                     state[0][2],
#                     state[1][0],
#                     state[1][1],
#                     state[1][2],
#                     state[2][0],
#                     state[2][1],
#                     state[2][2]])
#         return move
#
#     def episode_over(self, winner):
#         if winner == DRAW:
#             print('Game over! It was a draw.')
#         else:
#             print('Game over! Winner: Player {0}'.format(winner))
#
#
# if __name__ == "__main__":
#     p1 = Computer(1)
#     p2 = Computer(2)
#     while True:
#         winner = play(p1, p2)
#         p1.episode_over(winner)
#         p2.episode_over(winner)


print('')
print('====================================================================================================')
print('== 문제 203. 위의 코드를 수정해서 게임의 승패여부도 출력되게 하시오.')
print('====================================================================================================')
# import random
# import csv
# from copy import copy, deepcopy
#
# # deepcopy : 메모리를 완전히 새롭게 생성
# # copy : 껍데기만 카피, 내용은 동일한 곳을 가리킴
#
# EMPTY = 0
# PLAYER_X = 1
# PLAYER_O = 2
# DRAW = 3
# BOARD_FORMAT = "----------------------------\n| {0} | {1} | {2} |\n|--------------------------|\n| {3} | {4} | {5} |\n|--------------------------|\n| {6} | {7} | {8} |\n----------------------------"
# NAMES = [' ', 'X', 'O']
#
# # 보드 출력
# def printboard(state):
#     cells = []
#     for i in range(3):
#         for j in range(3):
#             cells.append(NAMES[state[i][j]].center(6))
#     print(BOARD_FORMAT.format(*cells))
#
#
# # 빈 판
# def emptystate():
#     return [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]
#
#
# def gameover(state):
#     # 가로/세로로 한 줄 완성한 플레이어가 있다면 그 플레이어 리턴
#     for i in range(3):
#         if state[i][0] != EMPTY and state[i][0] == state[i][1] and state[i][0] == state[i][2]:
#             return state[i][0]
#         if state[0][i] != EMPTY and state[0][i] == state[1][i] and state[0][i] == state[2][i]:
#             return state[0][i]
#     # 좌우 대각선
#     if state[0][0] != EMPTY and state[0][0] == state[1][1] and state[0][0] == state[2][2]:
#         return state[0][0]
#     if state[0][2] != EMPTY and state[0][2] == state[1][1] and state[0][2] == state[2][0]:
#         return state[0][2]
#     # 판이 비었는지
#     for i in range(3):
#         for j in range(3):
#             if state[i][j] == EMPTY:
#                 return EMPTY
#     return DRAW
#
#
# # 사람
# class Human(object):
#     def __init__(self, player):
#         self.player = player
#
#     # 착수
#     def action(self, state):
#         printboard(state)
#         action = None
#         while action not in range(1, 10):
#             action = int(input('Your move? '))
#         switch_map = {
#             1: (0, 0),
#             2: (0, 1),
#             3: (0, 2),
#             4: (1, 0),
#             5: (1, 1),
#             6: (1, 2),
#             7: (2, 0),
#             8: (2, 1),
#             9: (2, 2)
#         }
#         return switch_map[action]
#
#     def episode_over(self, winner):
#         if winner == DRAW:
#             print('Game over! It was a draw.')
#         else:
#             print('Game over! Winner: Player {0}'.format(winner))
#
#
# def play(agent1, agent2):
#     state = emptystate()
#     for i in range(9):
#         if i % 2 == 0:
#             move = agent1.action(state)
#         else:
#             move = agent2.action(state)
#         state[move[0]][move[1]] = (i % 2) + 1
#         winner = gameover(state)
#         if winner != EMPTY:
#             Fn = ('D:\\KYH\\02.PYTHON\\data\\Test400.csv')
#             w = csv.writer(open(Fn, 'a'), delimiter=',')
#             w.writerow([state[0][0],
#                         state[0][1],
#                         state[0][2],
#                         state[1][0],
#                         state[1][1],
#                         state[1][2],
#                         state[2][0],
#                         state[2][1],
#                         state[2][2],
#                         winner])
#             return winner
#     return winner
#
#
#  class Computer(object):
#     def __init__(self, player):
#         self.player = player
#
#     def random(self, state):
#         available = []
#         for i in range(3):
#             for j in range(3):
#                 if state[i][j] == EMPTY:
#                     available.append((i, j))
#         return random.choice(available)
#
#     # 컴퓨터가 착수
#     def action(self, state):
#         printboard(state)
#         action = None
#         move = self.random(state)
#         state[move[0]][move[1]] = self.player
#         # Fn = ('D:\\KYH\\02.PYTHON\\data\\Test200.csv')
#         # w = csv.writer(open(Fn, 'a'), delimiter=',')
#         # w.writerow([state[0][0],
#         #             state[0][1],
#         #             state[0][2],
#         #             state[1][0],
#         #             state[1][1],
#         #             state[1][2],
#         #             state[2][0],
#         #             state[2][1],
#         #             state[2][2]])
#         return move
#
#     def episode_over(self, winner):
#         if winner == DRAW:
#             print('Game over! It was a draw.')
#         else:
#             print('Game over! Winner: Player {0}'.format(winner))
#
#
# if __name__ == "__main__":
#     p1 = Computer(1)
#     p2 = Computer(2)
#     while True:
#         winner = play(p1, p2)
#         p1.episode_over(winner)
#         p2.episode_over(winner)


print('')
print('====================================================================================================')
print('== 문제 204. 무성이가 만든 오목 파이썬 코드를 구현하시오.')
print('====================================================================================================')
# # 사람 대 사람 오목 게임
# EMPTY = 0  # 비어있는 칸은 0으로
# DRAW = 3  # 비긴 경우는 3으로
#
# # 9x9 바둑판 만들기
#
# BOARD_FORMAT = "----1----2----3----4----5----6----7----8----9--\n1| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} |\n|" \
#                "-----------------------------------------------\n2| {9} | {10} | {11} | {12} | {13} | {14} | {15} | {16} | {17} |\n|" \
#                "-----------------------------------------------\n3| {18} | {19} | {20} | {21} | {22} | {23} | {24} | {25} | {26} |\n|" \
#                "-----------------------------------------------\n4| {27} | {28} | {29} | {30} | {31} | {32} | {33} | {34} | {35} |\n|" \
#                "-----------------------------------------------\n5| {36} | {37} | {38} | {39} | {40} | {41} | {42} | {43} | {44} |\n|" \
#                "-----------------------------------------------\n6| {45} | {46} | {47} | {48} | {49} | {50} | {51} | {52} | {53} |\n|" \
#                "-----------------------------------------------\n7| {54} | {55} | {56} | {57} | {58} | {59} | {60} | {61} | {62} |\n|" \
#                "-----------------------------------------------\n8| {63} | {64} | {65} | {66} | {67} | {68} | {69} | {70} | {71} |\n|" \
#                "-----------------------------------------------\n9| {72} | {73} | {74} | {75} | {76} | {77} | {78} | {79} | {80} |\n" \
#                "------------------------------------------------"
# # Names[0] -> 비어있는 경우 ' ', Names[1] -> Player1 의 돌은 흑돌, Names[2] -> Player2 의 돌은 백돌
# Names = ['  ', '●', '○']
#
#
# # 바둑판에 돌 놓기
# def printboard(state):
#     ball = []  # 임의의 리스트 ball 을 만들어서
#     for i in range(9):
#         for j in range(9):
#             ball.append(Names[state[i][j]])  # 바둑판 좌표 state[i][j] 의 값이 1이면 Names[1] -> 흑돌을 리스트에 입력
#     print(BOARD_FORMAT.format(*ball))  # 처음에 만든 BOARD_FORMAT 바둑판에 돌 놓기
#
#
# # 초기 바둑판 좌표값
# def emptyboard():
#     empty_board = [[EMPTY for i in range(9)] for i in range(9)]
#     return empty_board  # [ [EMPTY,EMPTY,EMPTY... EMPTY 총 9개], [EMPTY,EMPTY,EMPTY... EMPTY 총 9개], ...]
#     # [ [EMPTY x 9] x 9 ]  로 이뤄진 리스트 만들기. 즉 모든 값이 EMPTY 인 9x9의 리스트 생성
#
#
# # 게임이 종료되는 조건 함수 생성
# def gameover(state):
#     for i in range(9):
#         for j in range(9):
#             try:
#                 # 한쪽이 이겨서 게임 종료되는 경우
#
#                 # 가로로 다섯칸 모두 1인 경우(player1의 흑돌이 가로로 연속 다섯칸에 놓인 경우)
#                 if state[i][j] * state[i][j + 1] * state[i][j + 2] * state[i][j + 3] * state[i][j + 4] == 1:
#                     return 1
#                     # 가로로 다섯칸 모두 2인 경우(player2의 백돌이 가로로 연속 다섯칸에 놓인 경우)
#                 if state[i][j] * state[i][j + 1] * state[i][j + 2] * state[i][j + 3] * state[i][j + 4] == 32:
#                     return 2
#                     # 세로로 다섯칸 모두 1인 경우(player1의 흑돌이 세로로 연속 다섯칸에 놓인 경우)
#                 if state[j][i] * state[j + 1][i] * state[j + 2][i] * state[j + 3][i] * state[j + 4][i] == 1:
#                     return 1
#                     # 세로로 다섯칸 모두 2인 경우(player2의 백돌이 가로로 연속 다섯칸에 놓인 경우)
#                 if state[j][i] * state[j + 1][i] * state[j + 2][i] * state[j + 3][i] * state[j + 4][i] == 32:
#                     return 2
#                     # 대각선으로 다섯칸 모두 1인 경우(player1의 흑돌이 대각선으로 연속 다섯칸에 놓인 경우)
#                 if state[i][j] * state[i + 1][j + 1] * state[i + 2][j + 2] * state[i + 3][j + 3] * state[i + 4][
#                             j + 4] == 1:
#                     return 1
#                 if state[i][j + 4] * state[i + 1][j + 3] * state[i + 2][j + 2] * state[i + 3][j + 1] * state[i + 4][
#                     j] == 1:
#                     return 1
#                     # 대각선으로 다섯칸 모두 2인 경우(player2의 백돌이 대각선으로 연속 다섯칸에 놓인 경우)
#                 if state[i][j] * state[i + 1][j + 1] * state[i + 2][j + 2] * state[i + 3][j + 3] * state[i + 4][
#                             j + 4] == 32:
#                     return 2
#                 if state[i][j + 4] * state[i + 1][j + 3] * state[i + 2][j + 2] * state[i + 3][j + 1] * state[i + 4][
#                     j] == 32:
#                     return 2
#
#             except IndexError:  # range(9)로 인덱스 범위 넘어가는 경우 continue 로 예외처리하여 에러 안 뜨게 함
#                 continue
#
#                 # 한쪽이 이겨서 게임이 종료된 경우가 아니며, 빈칸이 존재하는 경우 계속 진행
#     for i in range(9):
#         for j in range(9):
#             if state[i][j] == EMPTY:
#                 return EMPTY
#                 # 한쪽이 이겨서 게임이 종료된 경우가 아니며, 빈칸도 없는 경우 비김
#     return DRAW
#
#
# class human():
#     def __init__(self, player):
#         self.player = player
#
#     # 돌 놓기
#     def Action(self, state):
#         printboard(state)  # 바둑판 출력
#         action = None
#         switch_map = {}  # 바둑판 좌표 딕셔너리
#         for i in range(1, 10):
#             for j in range(1, 10):
#                 switch_map[10 * i + j] = (i, j)
#
#                 # 인풋 받기
#         while action not in range(11, 100) or state[switch_map[action][0] - 1][switch_map[action][1] - 1] != EMPTY:
#             try:
#                 action = int(input('Player{}의 차례입니다. '.format(self.player)))
#             except ValueError:
#                 continue
#
#         return switch_map[action]
#
#     # 게임 종료시 출력 문구
#     def episode_over(self, winner):
#         if winner == DRAW:
#             print('무승부입니다.')
#         else:
#             print('승자는 Player{} 입니다.'.format(winner))
#
# class Computer(object):
#     def __init__(self, player):
#         self.player = player
#
#     def random(self, state):
#         available = []
#         for i in range(3):
#             for j in range(3):
#                 if state[i][j] == EMPTY:
#                     available.append((i, j))
#         return random.choice(available)
#
#     # 컴퓨터가 착수
#     def action(self, state):
#         printboard(state)
#         action = None
#         move = self.random(state)
#         state[move[0]][move[1]] = self.player
#         return move
#
#     def episode_over(self, winner):
#         if winner == DRAW:
#             print('Game over! It was a draw.')
#         else:
#             print('Game over! Winner: Player {0}'.format(winner))
#
#
# # 게임 진행
# def play(p1, p2):
#     state = emptyboard()
#     for i in range(81):
#         if i % 2 == 0:
#             move = p1.Action(state)
#         else:
#             move = p2.Action(state)
#         state[move[0] - 1][move[1] - 1] = i % 2 + 1
#         winner = gameover(state)
#         if winner != EMPTY:
#             printboard(state)
#             return winner
#     return winner
#
#
# if __name__ == '__main__':
#     p1 = human(1)
#     p2 = human(2)
#     while True:
#         winner = play(p1, p2)
#         p1.episode_over(winner)
#         if winner != '':
#             break  # winner 가 있을 경우 루프 벗어남


print('')
print('====================================================================================================')
print('== 문제 206. 10만번 학습시킨 csv 파일을 생성하시오!')
print('====================================================================================================')
import csv
import random
from copy import copy, deepcopy

EMPTY = 0
PLAYER_X = 1
PLAYER_O = 2
DRAW = 3
BOARD_FORMAT = "----------------------------\n| {0} | {1} | {2} |\n|--------------------------|\n| {3} | {4} | {5} |\n|--------------------------|\n| {6} | {7} | {8} |\n----------------------------"
NAMES = [' ', 'X', 'O']


def printboard(state):
    cells = []
    for i in range(3):
        for j in range(3):
            cells.append(NAMES[state[i][j]].center(6))
    print(BOARD_FORMAT.format(*cells))


def emptystate():
    return [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]


def gameover(state):
    for i in range(3):
        if state[i][0] != EMPTY and state[i][0] == state[i][1] and state[i][0] == state[i][2]:
            return state[i][0]
        if state[0][i] != EMPTY and state[0][i] == state[1][i] and state[0][i] == state[2][i]:
            return state[0][i]
    if state[0][0] != EMPTY and state[0][0] == state[1][1] and state[0][0] == state[2][2]:
        return state[0][0]
    if state[0][2] != EMPTY and state[0][2] == state[1][1] and state[0][2] == state[2][0]:
        return state[0][2]
    for i in range(3):
        for j in range(3):
            if state[i][j] == EMPTY:
                return EMPTY
    return DRAW


def last_to_act(state):
    countx = 0
    counto = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] == PLAYER_X:
                countx += 1
            elif state[i][j] == PLAYER_O:
                counto += 1
    if countx == counto:
        return PLAYER_O
    if countx == (counto + 1):
        return PLAYER_X
    return -1


def enumstates(state, idx, agent):
    if idx > 8:
        player = last_to_act(state)
        if player == agent.player:
            agent.add(state)
    else:
        winner = gameover(state)
        if winner != EMPTY:
            return
        i = int(idx / 3)
        j = idx % 3
        for val in range(3):
            state[i][j] = val
            enumstates(state, idx + 1, agent)


class Agent(object):
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
        self.dataset = []
        enumstates(emptystate(), 0, self)

    def episode_over(self, winner):
        self.backup(self.winnerval(winner))
        self.prevstate = None
        self.prevscore = 0
        self.gamenum += 1

    def action(self, state):
        r = random.random()
        if r < self.epsilon:
            move = self.random(state)
            self.log('>>>>>>> Exploratory action: ' + str(move))
        else:
            move = self.greedy(state)
            self.log('>>>>>>> Best action: ' + str(move))
        state[move[0]][move[1]] = self.player
        self.prevstate = self.statetuple(state)
        self.prevscore = self.lookup(state)
        state[move[0]][move[1]] = EMPTY
        return move

    def random(self, state):
        available = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == EMPTY:
                    available.append((i, j))
        return random.choice(available)

    def greedy(self, state):
        maxval = -50000
        maxmove = None
        if self.verbose:
            cells = []
        for i in range(3):
            for j in range(3):
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
                    cells.append(NAMES[state[i][j]].center(6))
        if self.verbose:
            print(BOARD_FORMAT.format(*cells))
        self.backup(maxval)
        return maxmove


    def lookup(self, state):
        key = self.statetuple(state)
        if not key in self.values:
            self.add(key)
        return self.values[key]

    def add(self, state):
        winner = gameover(state)
        tup = self.statetuple(state)
        print(tup)
        self.values[tup] = self.winnerval(winner)

    def backup(self, nextval):
        if self.prevstate != None and self.learning:
            prevval = self.values[self.prevstate]
            self.values[self.prevstate] += self.alpha * (nextval - self.prevscore)
            self.dataset.append([self.prevstate[0][0],
                                 self.prevstate[0][1],
                                 self.prevstate[0][2],
                                 self.prevstate[1][0],
                                 self.prevstate[1][1],
                                 self.prevstate[1][2],
                                 self.prevstate[2][0],
                                 self.prevstate[2][1],
                                 self.prevstate[2][2],
                                 prevval,
                                 self.values[self.prevstate],
                                 self.player,
                                 self.gamenum,
                                 self.rownum])
            self.rownum += 1

    def datatofile(self):
        Fn = open("D:\\KYH\\02.PYTHON\\data\\Test0508.csv", 'a')
        w = csv.writer(Fn, delimiter=',', lineterminator='\n')
        for data in self.dataset:
            w.writerow(data)
        Fn.close()

    def winnerval(self, winner):
        if winner == self.player:
            return 1
        elif winner == EMPTY:
            return 0.5
        elif winner == DRAW:
            return 0
        else:
            return self.lossval

    def printvalues(self):
        vals = deepcopy(self.values)
        for key in vals:
            state = [list(key[0]), list(key[1]), list(key[2])]
            cells = []
            for i in range(3):
                for j in range(3):
                    if state[i][j] == EMPTY:
                        state[i][j] = self.player
                        cells.append(str(self.lookup(state)).center(3))
                        state[i][j] = EMPTY
                    else:
                        cells.append(NAMES[state[i][j]].center(3))
            print(BOARD_FORMAT.format(*cells))

    def statetuple(self, state):
        return (tuple(state[0]), tuple(state[1]), tuple(state[2]))

    def log(self, s):
        if self.verbose:
            print(s)


class Human(object):
    def __init__(self, player):
        self.player = player

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
    p1 = Agent(1, lossval=-1)
    p2 = Agent(2, lossval=-1)

    for i in range(100000):
        if i % 10 == 0:
            print('Game: {0}'.format(i))

        winner = play(p1, p2)
        p1.episode_over(winner)
        p2.episode_over(winner)

    p1.datatofile()
    p2.datatofile()

    while True:
        p2.verbose = True
        p1 = Human(1)
        winner = play(p1, p2)
        p1.episode_over(winner)
        p2.episode_over(winner)


print('')
print('====================================================================================================')
print('== 문제 207. ttt_data 테이블을 만들고 데이터를 입력하시오.')
print('====================================================================================================')
# create table ttt_data
# (
#     c1 number(2),
#     c2 number(2),
#     c3 number(2),
#     c4 number(2),
#     c5 number(2),
#     c6 number(2),
#     c7 number(2),
#     c8 number(2),
#     c9 number(2),
#     val_old number(20,17),
#     val_new number(20,17),
#     player number(2),
#     game_num number(10),
#     learning_order number(10)
# );


# - mit 코드 총정리 1 (사람 vs 사람)
# - mit 코드 총정리 2 (사람 vs 컴퓨터(랜덤))
# - mit 코드 총정리 3 (컴퓨터(랜덤) vs 컴퓨터(랜덤))
# - mit 코드 총정리 4 (사람 vs 컴퓨터(학습))
# - mit 코드 총정리 5 (사람 vs 컴퓨터(스스로 학습))


print('')
print('====================================================================================================')
print('== 문제 208. mit 총정리 5번째 코드를 수정해서 agent 가 학습할 때 printboard 가 계속 출력되게 하시오.')
print('====================================================================================================')
import csv
import random
from copy import copy, deepcopy

EMPTY = 0
PLAYER_X = 1
PLAYER_O = 2
DRAW = 3
BOARD_FORMAT = "----------------------------\n| {0} | {1} | {2} |\n|--------------------------|\n| {3} | {4} | {5} |\n|--------------------------|\n| {6} | {7} | {8} |\n----------------------------"
NAMES = [' ', 'X', 'O']


def printboard(state):
    cells = []
    for i in range(3):
        for j in range(3):
            cells.append(NAMES[state[i][j]].center(6))
    print(BOARD_FORMAT.format(*cells))


def emptystate():
    return [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]


def gameover(state):
    for i in range(3):
        if state[i][0] != EMPTY and state[i][0] == state[i][1] and state[i][0] == state[i][2]:
            return state[i][0]
        if state[0][i] != EMPTY and state[0][i] == state[1][i] and state[0][i] == state[2][i]:
            return state[0][i]
    if state[0][0] != EMPTY and state[0][0] == state[1][1] and state[0][0] == state[2][2]:
        return state[0][0]
    if state[0][2] != EMPTY and state[0][2] == state[1][1] and state[0][2] == state[2][0]:
        return state[0][2]
    for i in range(3):
        for j in range(3):
            if state[i][j] == EMPTY:
                return EMPTY
    return DRAW


def last_to_act(state):
    countx = 0
    counto = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] == PLAYER_X:
                countx += 1
            elif state[i][j] == PLAYER_O:
                counto += 1
    if countx == counto:
        return PLAYER_O
    if countx == (counto + 1):
        return PLAYER_X
    return -1


def enumstates(state, idx, agent):
    if idx > 8:
        player = last_to_act(state)
        if player == agent.player:
            agent.add(state)
    else:
        winner = gameover(state)
        if winner != EMPTY:
            return
        i = int(idx / 3)
        j = idx % 3
        for val in range(3):
            state[i][j] = val
            enumstates(state, idx + 1, agent)


class Agent(object):
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
        self.dataset = []
        enumstates(emptystate(), 0, self)

    def episode_over(self, winner):
        self.backup(self.winnerval(winner))
        self.prevstate = None
        self.prevscore = 0
        self.gamenum += 1

    def action(self, state):
        r = random.random()
        if r < self.epsilon:
            move = self.random(state)
            self.log('>>>>>>> Exploratory action: ' + str(move))
        else:
            move = self.greedy(state)
            self.log('>>>>>>> Best action: ' + str(move))
        state[move[0]][move[1]] = self.player
        self.prevstate = self.statetuple(state)
        self.prevscore = self.lookup(state)
        state[move[0]][move[1]] = EMPTY

        return move

    def random(self, state):
        available = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == EMPTY:
                    available.append((i, j))
        return random.choice(available)

    def greedy(self, state):
        maxval = -50000
        maxmove = None
        if self.verbose:
            cells = []
        for i in range(3):
            for j in range(3):
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
                    cells.append(NAMES[state[i][j]].center(6))
        if self.verbose:
            print(BOARD_FORMAT.format(*cells))
        self.backup(maxval)
        return maxmove


    def lookup(self, state):
        key = self.statetuple(state)
        if not key in self.values:
            self.add(key)
        return self.values[key]

    def add(self, state):
        winner = gameover(state)
        tup = self.statetuple(state)
        print(tup)
        self.values[tup] = self.winnerval(winner)

    def backup(self, nextval):
        if self.prevstate != None and self.learning:
            prevval = self.values[self.prevstate]
            self.values[self.prevstate] += self.alpha * (nextval - self.prevscore)
            self.dataset.append([self.prevstate[0][0],
                                 self.prevstate[0][1],
                                 self.prevstate[0][2],
                                 self.prevstate[1][0],
                                 self.prevstate[1][1],
                                 self.prevstate[1][2],
                                 self.prevstate[2][0],
                                 self.prevstate[2][1],
                                 self.prevstate[2][2],
                                 prevval,
                                 self.values[self.prevstate],
                                 self.player,
                                 self.gamenum,
                                 self.rownum])
            self.rownum += 1

    def datatofile(self):
        Fn = open("D:\\KYH\\02.PYTHON\\data\\Test0508.csv", 'a')
        w = csv.writer(Fn, delimiter=',', lineterminator='\n')
        for data in self.dataset:
            w.writerow(data)
        Fn.close()

    def winnerval(self, winner):
        if winner == self.player:
            return 1
        elif winner == EMPTY:
            return 0.5
        elif winner == DRAW:
            return 0
        else:
            return self.lossval

    def printvalues(self):
        vals = deepcopy(self.values)
        for key in vals:
            state = [list(key[0]), list(key[1]), list(key[2])]
            cells = []
            for i in range(3):
                for j in range(3):
                    if state[i][j] == EMPTY:
                        state[i][j] = self.player
                        cells.append(str(self.lookup(state)).center(3))
                        state[i][j] = EMPTY
                    else:
                        cells.append(NAMES[state[i][j]].center(3))
            print(BOARD_FORMAT.format(*cells))

    def statetuple(self, state):
        return (tuple(state[0]), tuple(state[1]), tuple(state[2]))

    def log(self, s):
        if self.verbose:
            print(s)


class Human(object):
    def __init__(self, player):
        self.player = player

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
            printboard(state)
        else:
            move = agent2.action(state)
            printboard(state)
        state[move[0]][move[1]] = (i % 2) + 1
        winner = gameover(state)
        if winner != EMPTY:
            return winner
    return winner


if __name__ == "__main__":
    p1 = Agent(1, lossval=-1)
    p2 = Agent(2, lossval=-1)

    for i in range(100000):
        if i % 10 == 0:
            print('Game: {0}'.format(i))

        winner = play(p1, p2)
        p1.episode_over(winner)
        p2.episode_over(winner)

    p1.datatofile()
    p2.datatofile()

    # while True:
    #     p2.verbose = True
    #     p1 = Human(1)
    #     winner = play(p1, p2)
    #     p1.episode_over(winner)
    #     p2.episode_over(winner)


print('')
print('====================================================================================================')
print('== 문제 209. 학습할때도 보드판에 확률이 출력이 되게 하려면?')
print('====================================================================================================')
import csv
import random
from copy import copy, deepcopy

EMPTY = 0
PLAYER_X = 1
PLAYER_O = 2
DRAW = 3
BOARD_FORMAT = "----------------------------\n| {0} | {1} | {2} |\n|--------------------------|\n| {3} | {4} | {5} |\n|--------------------------|\n| {6} | {7} | {8} |\n----------------------------"
NAMES = [' ', 'X', 'O']


def printboard(state):
    cells = []
    for i in range(3):
        for j in range(3):
            cells.append(NAMES[state[i][j]].center(6))
    print(BOARD_FORMAT.format(*cells))


def emptystate():
    return [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]


def gameover(state):
    for i in range(3):
        if state[i][0] != EMPTY and state[i][0] == state[i][1] and state[i][0] == state[i][2]:
            return state[i][0]
        if state[0][i] != EMPTY and state[0][i] == state[1][i] and state[0][i] == state[2][i]:
            return state[0][i]
    if state[0][0] != EMPTY and state[0][0] == state[1][1] and state[0][0] == state[2][2]:
        return state[0][0]
    if state[0][2] != EMPTY and state[0][2] == state[1][1] and state[0][2] == state[2][0]:
        return state[0][2]
    for i in range(3):
        for j in range(3):
            if state[i][j] == EMPTY:
                return EMPTY
    return DRAW


def last_to_act(state):
    countx = 0
    counto = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] == PLAYER_X:
                countx += 1
            elif state[i][j] == PLAYER_O:
                counto += 1
    if countx == counto:
        return PLAYER_O
    if countx == (counto + 1):
        return PLAYER_X
    return -1


def enumstates(state, idx, agent):
    if idx > 8:
        player = last_to_act(state)
        if player == agent.player:
            agent.add(state)
    else:
        winner = gameover(state)
        if winner != EMPTY:
            return
        i = int(idx / 3)
        j = idx % 3
        for val in range(3):
            state[i][j] = val
            enumstates(state, idx + 1, agent)


class Agent(object):
    def __init__(self, player, verbose=True, lossval=0, learning=True):
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
        self.dataset = []
        enumstates(emptystate(), 0, self)

    def episode_over(self, winner):
        self.backup(self.winnerval(winner))
        self.prevstate = None
        self.prevscore = 0
        self.gamenum += 1

    def action(self, state):
        r = random.random()
        if r < self.epsilon:
            move = self.random(state)
            self.log('>>>>>>> Exploratory action: ' + str(move))
        else:
            move = self.greedy(state)
            self.log('>>>>>>> Best action: ' + str(move))
        state[move[0]][move[1]] = self.player
        self.prevstate = self.statetuple(state)
        self.prevscore = self.lookup(state)
        state[move[0]][move[1]] = EMPTY

        return move

    def random(self, state):
        available = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == EMPTY:
                    available.append((i, j))
        return random.choice(available)

    def greedy(self, state):
        maxval = -50000
        maxmove = None
        if self.verbose:
            cells = []
        for i in range(3):
            for j in range(3):
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
                    cells.append(NAMES[state[i][j]].center(6))
        if self.verbose:
            print(BOARD_FORMAT.format(*cells))
        self.backup(maxval)
        return maxmove

    def lookup(self, state):
        key = self.statetuple(state)
        if not key in self.values:
            self.add(key)
        return self.values[key]

    def add(self, state):
        winner = gameover(state)
        tup = self.statetuple(state)
        print(tup)
        self.values[tup] = self.winnerval(winner)

    def backup(self, nextval):
        if self.prevstate != None and self.learning:
            prevval = self.values[self.prevstate]
            self.values[self.prevstate] += self.alpha * (nextval - self.prevscore)
            self.dataset.append([self.prevstate[0][0],
                                 self.prevstate[0][1],
                                 self.prevstate[0][2],
                                 self.prevstate[1][0],
                                 self.prevstate[1][1],
                                 self.prevstate[1][2],
                                 self.prevstate[2][0],
                                 self.prevstate[2][1],
                                 self.prevstate[2][2],
                                 prevval,
                                 self.values[self.prevstate],
                                 self.player,
                                 self.gamenum,
                                 self.rownum])
            self.rownum += 1

    def datatofile(self):
        Fn = open("D:\\KYH\\02.PYTHON\\data\\Test0508.csv", 'a')
        w = csv.writer(Fn, delimiter=',', lineterminator='\n')
        for data in self.dataset:
            w.writerow(data)
        Fn.close()

    def winnerval(self, winner):
        if winner == self.player:
            return 1
        elif winner == EMPTY:
            return 0.5
        elif winner == DRAW:
            return 0
        else:
            return self.lossval

    def printvalues(self):
        vals = deepcopy(self.values)
        for key in vals:
            state = [list(key[0]), list(key[1]), list(key[2])]
            cells = []
            for i in range(3):
                for j in range(3):
                    if state[i][j] == EMPTY:
                        state[i][j] = self.player
                        cells.append(str(self.lookup(state)).center(3))
                        state[i][j] = EMPTY
                    else:
                        cells.append(NAMES[state[i][j]].center(3))
            print(BOARD_FORMAT.format(*cells))

    def statetuple(self, state):
        return (tuple(state[0]), tuple(state[1]), tuple(state[2]))

    def log(self, s):
        if self.verbose:
            print(s)


class Human(object):
    def __init__(self, player):
        self.player = player

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
            printboard(state)
        else:
            move = agent2.action(state)
            printboard(state)
        state[move[0]][move[1]] = (i % 2) + 1
        winner = gameover(state)
        if winner != EMPTY:
            return winner
    return winner


if __name__ == "__main__":
    p1 = Agent(1, lossval=-1)
    p2 = Agent(2, lossval=-1)

    for i in range(100000):
        if i % 10 == 0:
            print('Game: {0}'.format(i))

        winner = play(p1, p2)
        p1.episode_over(winner)
        p2.episode_over(winner)

    p1.datatofile()
    p2.datatofile()


print('')
print('====================================================================================================')
print('== 문제 210. 만번 학습한 agent1 과 10만번 학습한 agent 를 각각 만들어서 대결하려면?')
print('====================================================================================================')
import csv
import random
from copy import copy, deepcopy

EMPTY = 0
PLAYER_X = 1
PLAYER_O = 2
DRAW = 3
BOARD_FORMAT = "----------------------------\n| {0} | {1} | {2} |\n|--------------------------|\n| {3} | {4} | {5} |\n|--------------------------|\n| {6} | {7} | {8} |\n----------------------------"
NAMES = [' ', 'X', 'O']


def printboard(state):
    cells = []
    for i in range(3):
        for j in range(3):
            cells.append(NAMES[state[i][j]].center(6))
    print(BOARD_FORMAT.format(*cells))


def emptystate():
    return [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]


def gameover(state):
    for i in range(3):
        if state[i][0] != EMPTY and state[i][0] == state[i][1] and state[i][0] == state[i][2]:
            return state[i][0]
        if state[0][i] != EMPTY and state[0][i] == state[1][i] and state[0][i] == state[2][i]:
            return state[0][i]
    if state[0][0] != EMPTY and state[0][0] == state[1][1] and state[0][0] == state[2][2]:
        return state[0][0]
    if state[0][2] != EMPTY and state[0][2] == state[1][1] and state[0][2] == state[2][0]:
        return state[0][2]
    for i in range(3):
        for j in range(3):
            if state[i][j] == EMPTY:
                return EMPTY
    return DRAW


def last_to_act(state):
    countx = 0
    counto = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] == PLAYER_X:
                countx += 1
            elif state[i][j] == PLAYER_O:
                counto += 1
    if countx == counto:
        return PLAYER_O
    if countx == (counto + 1):
        return PLAYER_X
    return -1


def enumstates(state, idx, agent):
    if idx > 8:
        player = last_to_act(state)
        if player == agent.player:
            agent.add(state)
    else:
        winner = gameover(state)
        if winner != EMPTY:
            return
        i = int(idx / 3)
        j = idx % 3
        for val in range(3):
            state[i][j] = val
            enumstates(state, idx + 1, agent)


class Agent(object):
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
        self.dataset = []
        enumstates(emptystate(), 0, self)

    def episode_over(self, winner):
        self.backup(self.winnerval(winner))
        self.prevstate = None
        self.prevscore = 0
        self.gamenum += 1

    def action(self, state):
        r = random.random()
        if r < self.epsilon:
            move = self.random(state)
            self.log('>>>>>>> Exploratory action: ' + str(move))
        else:
            move = self.greedy(state)
            self.log('>>>>>>> Best action: ' + str(move))
        state[move[0]][move[1]] = self.player
        self.prevstate = self.statetuple(state)
        self.prevscore = self.lookup(state)
        state[move[0]][move[1]] = EMPTY

        return move

    def random(self, state):
        available = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == EMPTY:
                    available.append((i, j))
        return random.choice(available)

    def greedy(self, state):
        maxval = -50000
        maxmove = None
        if self.verbose:
            cells = []
        for i in range(3):
            for j in range(3):
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
                    cells.append(NAMES[state[i][j]].center(6))
        if self.verbose:
            print(BOARD_FORMAT.format(*cells))
        self.backup(maxval)
        return maxmove

    def lookup(self, state):
        key = self.statetuple(state)
        if not key in self.values:
            self.add(key)
        return self.values[key]

    def add(self, state):
        winner = gameover(state)
        tup = self.statetuple(state)
        print(tup)
        self.values[tup] = self.winnerval(winner)

    def backup(self, nextval):
        if self.prevstate != None and self.learning:
            prevval = self.values[self.prevstate]
            self.values[self.prevstate] += self.alpha * (nextval - self.prevscore)
            self.dataset.append([self.prevstate[0][0],
                                 self.prevstate[0][1],
                                 self.prevstate[0][2],
                                 self.prevstate[1][0],
                                 self.prevstate[1][1],
                                 self.prevstate[1][2],
                                 self.prevstate[2][0],
                                 self.prevstate[2][1],
                                 self.prevstate[2][2],
                                 prevval,
                                 self.values[self.prevstate],
                                 self.player,
                                 self.gamenum,
                                 self.rownum])
            self.rownum += 1

    def datatofile(self):
        Fn = open("D:\\KYH\\02.PYTHON\\data\\Test0508.csv", 'a')
        w = csv.writer(Fn, delimiter=',', lineterminator='\n')
        for data in self.dataset:
            w.writerow(data)
        Fn.close()

    def winnerval(self, winner):
        if winner == self.player:
            return 1
        elif winner == EMPTY:
            return 0.5
        elif winner == DRAW:
            return 0
        else:
            return self.lossval

    def printvalues(self):
        vals = deepcopy(self.values)
        for key in vals:
            state = [list(key[0]), list(key[1]), list(key[2])]
            cells = []
            for i in range(3):
                for j in range(3):
                    if state[i][j] == EMPTY:
                        state[i][j] = self.player
                        cells.append(str(self.lookup(state)).center(3))
                        state[i][j] = EMPTY
                    else:
                        cells.append(NAMES[state[i][j]].center(3))
            print(BOARD_FORMAT.format(*cells))

    def statetuple(self, state):
        return (tuple(state[0]), tuple(state[1]), tuple(state[2]))

    def log(self, s):
        if self.verbose:
            print(s)


class Human(object):
    def __init__(self, player):
        self.player = player

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
            printboard(state)
        else:
            move = agent2.action(state)
            printboard(state)
        state[move[0]][move[1]] = (i % 2) + 1
        winner = gameover(state)
        if winner != EMPTY:
            return winner
    return winner


if __name__ == "__main__":
    p1 = Agent(1, lossval=-1)
    p2 = Agent(2, lossval=-1)
    p3 = Agent(1, lossval=-1)
    p4 = Agent(2, lossval=-1)

    for i in range(10000):
        if i % 10 == 0:
            print('Game: {0}'.format(i))

        winner = play(p1, p2)
        p1.episode_over(winner)
        p2.episode_over(winner)

    p1.datatofile()
    p2.datatofile()

    for i in range(100000):
        if i % 10 == 0:
            print('Game: {0}'.format(i))

        winner = play(p3, p4)
        p1.episode_over(winner)
        p2.episode_over(winner)

    p3.datatofile()
    p4.datatofile()

    while True:
        p2.verbose = True
        p4.verbose = True
        winner = play(p2, p4)
        p2.episode_over(winner)
        p4.episode_over(winner)


print('')
print('====================================================================================================')
print('== 문제 210. 만번 학습한 agent 의 학습 데이터를 csv 로 내리고(agent_10000.csv) 십만번 학습한 agent 의 학습 데이터를 ')
print('==  csv(agent_100000.csv) 로 내리시오.')
print('====================================================================================================')
import csv
import random
from copy import copy, deepcopy

EMPTY = 0
PLAYER_X = 1
PLAYER_O = 2
DRAW = 3
BOARD_FORMAT = "----------------------------\n| {0} | {1} | {2} |\n|--------------------------|\n| {3} | {4} | {5} |\n|--------------------------|\n| {6} | {7} | {8} |\n----------------------------"
NAMES = [' ', 'X', 'O']


def printboard(state):
    cells = []
    for i in range(3):
        for j in range(3):
            cells.append(NAMES[state[i][j]].center(6))
    print(BOARD_FORMAT.format(*cells))


def emptystate():
    return [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]


def gameover(state):
    for i in range(3):
        if state[i][0] != EMPTY and state[i][0] == state[i][1] and state[i][0] == state[i][2]:
            return state[i][0]
        if state[0][i] != EMPTY and state[0][i] == state[1][i] and state[0][i] == state[2][i]:
            return state[0][i]
    if state[0][0] != EMPTY and state[0][0] == state[1][1] and state[0][0] == state[2][2]:
        return state[0][0]
    if state[0][2] != EMPTY and state[0][2] == state[1][1] and state[0][2] == state[2][0]:
        return state[0][2]
    for i in range(3):
        for j in range(3):
            if state[i][j] == EMPTY:
                return EMPTY
    return DRAW


def last_to_act(state):
    countx = 0
    counto = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] == PLAYER_X:
                countx += 1
            elif state[i][j] == PLAYER_O:
                counto += 1
    if countx == counto:
        return PLAYER_O
    if countx == (counto + 1):
        return PLAYER_X
    return -1


def enumstates(state, idx, agent):
    if idx > 8:
        player = last_to_act(state)
        if player == agent.player:
            agent.add(state)
    else:
        winner = gameover(state)
        if winner != EMPTY:
            return
        i = int(idx / 3)
        j = idx % 3
        for val in range(3):
            state[i][j] = val
            enumstates(state, idx + 1, agent)


class Agent(object):
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
        self.dataset = []
        enumstates(emptystate(), 0, self)

    def episode_over(self, winner):
        self.backup(self.winnerval(winner))
        self.prevstate = None
        self.prevscore = 0
        self.gamenum += 1

    def action(self, state):
        r = random.random()
        if r < self.epsilon:
            move = self.random(state)
            self.log('>>>>>>> Exploratory action: ' + str(move))
        else:
            move = self.greedy(state)
            self.log('>>>>>>> Best action: ' + str(move))
        state[move[0]][move[1]] = self.player
        self.prevstate = self.statetuple(state)
        self.prevscore = self.lookup(state)
        state[move[0]][move[1]] = EMPTY

        return move

    def random(self, state):
        available = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == EMPTY:
                    available.append((i, j))
        return random.choice(available)

    def greedy(self, state):
        maxval = -50000
        maxmove = None
        if self.verbose:
            cells = []
        for i in range(3):
            for j in range(3):
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
                    cells.append(NAMES[state[i][j]].center(6))
        if self.verbose:
            print(BOARD_FORMAT.format(*cells))
        self.backup(maxval)
        return maxmove

    def lookup(self, state):
        key = self.statetuple(state)
        if not key in self.values:
            self.add(key)
        return self.values[key]

    def add(self, state):
        winner = gameover(state)
        tup = self.statetuple(state)
        print(tup)
        self.values[tup] = self.winnerval(winner)

    def backup(self, nextval):
        if self.prevstate != None and self.learning:
            prevval = self.values[self.prevstate]
            self.values[self.prevstate] += self.alpha * (nextval - self.prevscore)
            self.dataset.append([self.prevstate[0][0],
                                 self.prevstate[0][1],
                                 self.prevstate[0][2],
                                 self.prevstate[1][0],
                                 self.prevstate[1][1],
                                 self.prevstate[1][2],
                                 self.prevstate[2][0],
                                 self.prevstate[2][1],
                                 self.prevstate[2][2],
                                 prevval,
                                 self.values[self.prevstate],
                                 self.player,
                                 self.gamenum,
                                 self.rownum])
            self.rownum += 1

    def datatofile(self):
        Fn = open("D:\\KYH\\02.PYTHON\\data\\agent_1000.csv", 'a')
        w = csv.writer(Fn, delimiter=',', lineterminator='\n')
        for data in self.dataset:
            w.writerow(data)
        Fn.close()

    def winnerval(self, winner):
        if winner == self.player:
            return 1
        elif winner == EMPTY:
            return 0.5
        elif winner == DRAW:
            return 0
        else:
            return self.lossval

    def printvalues(self):
        vals = deepcopy(self.values)
        for key in vals:
            state = [list(key[0]), list(key[1]), list(key[2])]
            cells = []
            for i in range(3):
                for j in range(3):
                    if state[i][j] == EMPTY:
                        state[i][j] = self.player
                        cells.append(str(self.lookup(state)).center(3))
                        state[i][j] = EMPTY
                    else:
                        cells.append(NAMES[state[i][j]].center(3))
            print(BOARD_FORMAT.format(*cells))

    def statetuple(self, state):
        return (tuple(state[0]), tuple(state[1]), tuple(state[2]))

    def log(self, s):
        if self.verbose:
            print(s)


class Human(object):
    def __init__(self, player):
        self.player = player

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
    p1 = Agent(1, lossval=-1)
    p2 = Agent(2, lossval=-1)

    for i in range(1000):
        if i % 10 == 0:
            print('Game: {0}'.format(i))

        winner = play(p1, p2)
        p1.episode_over(winner)
        p2.episode_over(winner)

    p1.datatofile()
    p2.datatofile()


print('')
print('====================================================================================================')
print('== 문제 211. 천번 학습한 agent 와 대결하시오! ')
print('====================================================================================================')
import random, csv
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

class Computer(object):
    def __init__(self, player):
        self.player = player
        self.values = {} # csv 에 있는 파일의 내용(9개의 판(수)과 가중치)를 읽어서 저장할 딕셔너리 변수
        self.readCSV() # init 할때 values 에 값 채워넣을려고 함수를 실행함
        self.verbose = True
        #print(self.values) # {((0, 2, 1), (2, 2, 0), (1, 1, 0)): -0.999999,...
                           #  위와 같이 values 딕셔너리 변수에 저장되어 있는지 확인한다.

    def readCSV(self):
        file = open("D:\\KYH\\02.PYTHON\\data\\agent_10000.csv", 'r')
        ttt_list = csv.reader(file)
        for t in ttt_list:
            try:
                self.values[((int(t[0]) ,int(t[1]) ,int(t[2])),(int(t[3]) ,int(t[4]) ,int(t[5])) ,(int(t[6]) ,int(t[7])
                             ,int(t[8])))] = float(t[10])
            except ValueError:    # {((0, 2, 1), (2, 2, 0), (1, 1, 0)): -0.999999,..
                continue

    def random(self, state):  # 남아있는 비어 있는 수들 중에서 한수를 random 으로 고르기 위한 함수
        available = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == EMPTY:
                    available.append((i ,j))
        return random.choice(available)

    def greedy(self, state):
        maxval = -50000  # 남아있는 수중에 가장 좋은 수의 가중치를 담기 위해  선언
        maxmove = None   # 남아있는 수중에 가장 좋은 수를 담기 위해 선언
        if self.verbose:  # 수를 둘때마다 남아있는 수들의 확률을 확인하기 위해서 사용하는 코드
            cells = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == EMPTY: # 남아있는 수중에 비어있는 수를 찾아서
                    state[i][j] = self.player # 거기에 플레이어의 숫자를 넣은후
                    val = self.lookup(state) # values 에 없으면 새로 0.5 를
                    #print(val)               # values 에 넣어주고 그 값을 다시 여기로 가져온다
                                              # 있으면 바로 values 에서 가져온다. (-0.9606 )
                    state[i][j] = EMPTY      # 그 수를 다시 비워준다.

                    if val > maxval:
                        maxval = val
                        # print (maxval) # 남아있는 수중에 가장 큰게 0.029698 (0.030) 이었음
                        maxmove = (i, j)
                        #print(maxmove)   # 남아있는 수중에 가장 가장치가 큰 자리 (2,0)
                    if self.verbose:  #
                        cells.append('{0:.3f}'.format(val).center(6))
                elif self.verbose:
                    cells.append(NAMES[state[i][j]].center(6))
        if self.verbose:
            print (BOARD_FORMAT.format(*cells))
           # ---------------------------    verbose 는 이 결과를 출력하기 위한 코드임
           # | 0.000 | -1.000 | 0.000 |
           # | -------------------------- |
           # | -1.000 | X | -0.961 |
           # | -------------------------- |
           # | 0.030 | -1.000 | 0.000 |
           # ----------------------------
        # print(maxmove)  # (2,0) 을 출력 ( 남아있는 수중에 가장 좋은수 )
        return maxmove

    def lookup(self, state):
        key = self.statetuple(state) # 리스트를 튜플로 바꿔주는 역활
        #print(key)  # x (player 1) 가 5번에 두었을때 o (player 2) 가 둘수있는 남아있는 수 출력
        # ((2, 0, 0), (0, 1, 0), (0, 0, 0))
        # ((0, 2, 0), (0, 1, 0), (0, 0, 0))
        # ((0, 0, 2), (0, 1, 0), (0, 0, 0))
        # ((0, 0, 0), (2, 1, 0), (0, 0, 0))
        # ((0, 0, 0), (0, 1, 2), (0, 0, 0))
        # ((0, 0, 0), (0, 1, 0), (2, 0, 0))
        # ((0, 0, 0), (0, 1, 0), (0, 2, 0))
        # ((0, 0, 0), (0, 1, 0), (0, 0, 2))
        if not key in self.values: # 위의  key 수들이 csv 에서 읽어온 수들중에 없다면
            self.add(key)  # values 에 없으며 add 함수로 추가
        #print (self.values) # {((0, 2, 1), (2, 2, 0), (1, 1, 0)): -0.999999, ...
        #print (self.values[key]) # -0.999847, 0.0, -0.999996, .......
        return self.values[key]  # 있으면 그거 리턴, 없으면 만들고 리턴

    def add(self, state):
        winner = gameover(state)
        tup = self.statetuple(state)
        self.values[tup] = self.winnerval(winner) # 1,-1,0.5, 0 (비긴것)

    def statetuple(self, state):
        return (tuple(state[0]) ,tuple(state[1]) ,tuple(state[2]))

    # 컴퓨터가 착수
    def action(self, state):
        printboard(state)
        action = None
        move = self.greedy(state)
        state[move[0]][move[1]] = self.player
        return move

    def winnerval(self, winner):
        if winner == self.player:
            return 1
        elif winner == EMPTY:
            return 0.5
        elif winner == DRAW:
            return 0
        else:
            return self.lossval

    def episode_over(self, winner):
        if winner == DRAW:
            print('Game over! It was a draw.')
        else:
            print('Game over! Winner: Player {0}'.format(winner))

if __name__ == "__main__":
    p1 = Human(1)
    p2 = Computer(2)
    while True:
        winner = play(p1, p2)
        p1.episode_over(winner)
        p2.episode_over(winner)


print('')
print('====================================================================================================')
print('== 문제 213. 위의 대결의 결과를 csv 로 남기시오! (1000번 학습 vs 십만번 학습)')
print('====================================================================================================')
import random, csv
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


class Computer(object):
    def __init__(self, player, filename):
        self.player = player
        self.values = {}  # csv 에 있는 파일의 내용(9개의 판(수)과 가중치)를 읽어서 저장할 딕셔너리 변수
        self.filename = filename
        self.readCSV()  # init 할때 values 에 값 채워넣을려고 함수를 실행함
        self.verbose = True
        # print(self.values) # {((0, 2, 1), (2, 2, 0), (1, 1, 0)): -0.999999,...
        #  위와 같이 values 딕셔너리 변수에 저장되어 있는지 확인한다.

    def readCSV(self):
        file = open("D:\\KYH\\02.PYTHON\\data\\" + self.filename, 'r')
        ttt_list = csv.reader(file)
        for t in ttt_list:
            try:
                self.values[((int(t[0]), int(t[1]), int(t[2])), (int(t[3]), int(t[4]), int(t[5])), (int(t[6]), int(t[7])
                                                                                                    , int(
                    t[8])))] = float(t[10])
            except ValueError:  # {((0, 2, 1), (2, 2, 0), (1, 1, 0)): -0.999999,..
                continue

    def random(self, state):  # 남아있는 비어 있는 수들 중에서 한수를 random 으로 고르기 위한 함수
        available = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == EMPTY:
                    available.append((i, j))
        return random.choice(available)

    def greedy(self, state):
        maxval = -50000  # 남아있는 수중에 가장 좋은 수의 가중치를 담기 위해  선언
        maxmove = None  # 남아있는 수중에 가장 좋은 수를 담기 위해 선언
        if self.verbose:  # 수를 둘때마다 남아있는 수들의 확률을 확인하기 위해서 사용하는 코드
            cells = []
        for i in range(3):
            for j in range(3):
                if state[i][j] == EMPTY:  # 남아있는 수중에 비어있는 수를 찾아서
                    state[i][j] = self.player  # 거기에 플레이어의 숫자를 넣은후
                    val = self.lookup(state)  # values 에 없으면 새로 0.5 를
                    # print(val)               # values 에 넣어주고 그 값을 다시 여기로 가져온다
                    # 있으면 바로 values 에서 가져온다. (-0.9606 )
                    state[i][j] = EMPTY  # 그 수를 다시 비워준다.

                    if val > maxval:
                        maxval = val
                        # print (maxval) # 남아있는 수중에 가장 큰게 0.029698 (0.030) 이었음
                        maxmove = (i, j)
                        # print(maxmove)   # 남아있는 수중에 가장 가장치가 큰 자리 (2,0)
                    if self.verbose:  #
                        cells.append('{0:.3f}'.format(val).center(6))
                elif self.verbose:
                    cells.append(NAMES[state[i][j]].center(6))
        if self.verbose:
            print(BOARD_FORMAT.format(*cells))
            # ---------------------------    verbose 는 이 결과를 출력하기 위한 코드임
            # | 0.000 | -1.000 | 0.000 |
            # | -------------------------- |
            # | -1.000 | X | -0.961 |
            # | -------------------------- |
            # | 0.030 | -1.000 | 0.000 |
            # ----------------------------
        # print(maxmove)  # (2,0) 을 출력 ( 남아있는 수중에 가장 좋은수 )
        return maxmove

    def lookup(self, state):
        key = self.statetuple(state)  # 리스트를 튜플로 바꿔주는 역활
        # print(key)  # x (player 1) 가 5번에 두었을때 o (player 2) 가 둘수있는 남아있는 수 출력
        # ((2, 0, 0), (0, 1, 0), (0, 0, 0))
        # ((0, 2, 0), (0, 1, 0), (0, 0, 0))
        # ((0, 0, 2), (0, 1, 0), (0, 0, 0))
        # ((0, 0, 0), (2, 1, 0), (0, 0, 0))
        # ((0, 0, 0), (0, 1, 2), (0, 0, 0))
        # ((0, 0, 0), (0, 1, 0), (2, 0, 0))
        # ((0, 0, 0), (0, 1, 0), (0, 2, 0))
        # ((0, 0, 0), (0, 1, 0), (0, 0, 2))
        if not key in self.values:  # 위의  key 수들이 csv 에서 읽어온 수들중에 없다면
            self.add(key)  # values 에 없으며 add 함수로 추가
        # print (self.values) # {((0, 2, 1), (2, 2, 0), (1, 1, 0)): -0.999999, ...
        # print (self.values[key]) # -0.999847, 0.0, -0.999996, .......
        return self.values[key]  # 있으면 그거 리턴, 없으면 만들고 리턴

    def add(self, state):
        winner = gameover(state)
        tup = self.statetuple(state)
        self.values[tup] = self.winnerval(winner)  # 1,-1,0.5, 0 (비긴것)

    def statetuple(self, state):
        return (tuple(state[0]), tuple(state[1]), tuple(state[2]))

    # 컴퓨터가 착수
    def action(self, state):
        printboard(state)
        action = None
        move = self.greedy(state)
        state[move[0]][move[1]] = self.player
        return move

    def winnerval(self, winner):
        if winner == self.player:
            return 1
        elif winner == EMPTY:
            return 0.5
        elif winner == DRAW:
            return 0
        else:
            return self.lossval

    def episode_over(self, winner):
        if winner == DRAW:
            print('Game over! It was a draw.')
        else:
            print('Game over! Winner: Player {0}'.format(winner))


if __name__ == "__main__":
    p1 = Computer(1, 'agent_1000.csv')
    p2 = Computer(2, 'agent_10000.csv')
    result_list = []

    for i in range(1000):
        winner = play(p2, p1)
        result_list.append(winner)
        p1.episode_over(winner)
        p2.episode_over(winner)

    Fn = open("D:\\KYH\\02.PYTHON\\data\\result.csv", 'a')
    w = csv.writer(Fn, delimiter=',', lineterminator='\n')
    for data in result_list:
        w.writerow([data])
    Fn.close()