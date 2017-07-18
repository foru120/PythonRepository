from tkinter import *
import math
import time
import random
import csv

class Map() :   # 지형 설정
    def __init__(self, canvas):
        self.canvas = canvas
        self.canvas.create_polygon(0,   100,    0,      150, 700, 200, 700, 150)
        self.canvas.create_polygon(100, 300,    100,    350, 800, 300, 800, 250)
        self.canvas.create_polygon(0,   400,    0,      450, 700, 500, 700, 450)
        self.canvas.create_polygon(100, 600,    100,    650, 800, 600, 800, 550)

class Ball() :  # 장애물역할인 공 설정
    def __init__(self, canvas):
        self.canvas = canvas
        self.w_width = self.canvas.winfo_width()
        self.w_height = self.canvas.winfo_height()
        self.ball_list = []
        self.counter = 100

    def reset(self):    # 게임 초기화
        for i in self.ball_list :
            self.canvas.delete(i[0])
        self.ball_list = []
        self.counter = 0

    def get_center(self, pos):  # 공의 하단 중앙 좌표 반환
        return (int(round((pos[0]+pos[2])/2)), int(pos[3]))

    def draw(self): # 공 그리기
        self.timing = 100   # 100 틱마다 공을 추가함
        if (len(self.ball_list) < 8) and (self.counter % self.timing == 0): # 공의 최대 갯수 8개
            self.ball_list.append([self.canvas.create_oval(1,1,25,25), 0, 3])
            self.counter = 0
        self.counter += 1

        for j in range(len(self.ball_list)-1, -1, -1) : # 공을 담은 리스트를 순회하며 공을 그림
            i = self.ball_list[j]
            id = i[0]

            # 공의 궤적은 직선 4개와 그 사이의 낙하
            pos = self.canvas.coords(i[0])
            base = self.get_center(pos)
            x = i[1]
            y=3
            if 550<=base[1]<=600 and 0<= base[0] < 100 :
                x = -5
                y = 0
            elif 550<=base[1]<=600 and base[0] >= 100 :
                if x == 0 : x = -5
                y = math.ceil(-1*(base[0]-100)/14+600) - base[1]
            elif 250 <= base[1] <=300 and base[0] >= 100 :
                if x == 0 : x = -5
                y = math.ceil(-1*(base[0]-100)/14+300) - base[1]
            elif 400<=base[1]<=450 and base[0] <= 700 :
                if x == 0 : x = 5
                y = math.ceil((base[0]/14)+400) - base[1]
            elif 100 <= base[1] <= 150 and base[0] <= 700:
                if x == 0 : x = 5
                y = math.ceil(base[0]/14+100) - base[1]
            else : y +=5

            if pos[3] < 570 and (pos[0]+x <= 0 or pos[2]+x >= self.w_width) :
                if x > 0:
                    self.canvas.move(id, self.w_width-pos[2]-2, y)
                elif x < 0:
                    self.canvas.move(id, -pos[0]+1, y)
                x = 0

            self.canvas.move(id, x, y)  # 공 위치 변경
            i[1], i[2] = x, y

            if pos[1] > self.w_height : # 공이 화면 밖을 벗어나면 삭제
                self.canvas.delete(i[0])
                del self.ball_list[j]

class Man():    # 캐릭터 설정

    dict = {}   # 데이터를 담을 딕셔너리

    def __init__(self, canvas, ball):
        self.canvas = canvas
        self.w_width = self.canvas.winfo_width()
        self.w_height = self.canvas.winfo_height()
        self.id = self.canvas.create_rectangle(5, 550, 35, 599)
        self.goal = self.canvas.create_rectangle(450,15, 475, 100)
        self.x = 0
        self.y = 5
        self.ball = ball
        self.canvas.bind_all('<KeyPress-j>', self.switch)
        # self.canvas.bind_all('<KeyPress-k>', self.jump_left)
        # self.canvas.bind_all('<KeyPress-l>', self.jump_right)
        self.val = True
        self.pre_pos = None
        self.jump = False
        self.move_list = [self.jump_right, self.jump_left]
        self.cycle_list = []
        self.game_num = 0
        self.load_csv()

    # csv 파일 저장, 불러오기 함수
    def save_csv(self):
        if self.game_num % 100 == 0 :
            file = open('learning_data/3000game_data.csv', 'w', newline = '')
            writer = csv.writer(file, delimiter = ',')
            writer.writerow([self.game_num])
            keys = Man.dict.keys()
            for key in keys :
                writer.writerow([int(key[0][0]), int(key[0][1]), int(key[1][0]), int(key[1][1]), int(key[2]), float(Man.dict[key])])
            file.close()
            print(self.game_num, 'game data is saved!')

    def load_csv(self):
        try :
            file = open('learning_data/35000game_data.csv', 'r')
            self.game_num = int(file.readline().split(',')[0])
            reader = csv.reader(file, delimiter = ',')
            for key in reader :
                Man.dict[((int(key[0]),int(key[1])), (int(key[2]),int(key[3])), int(key[4]))]=float(key[5])
            file.close()
            print('load!')
            print(Man.dict)
        except Exception:
            print('load fail!')

    # 게임 속도를 조절 - j 키 바인딩
    def switch(self, evt):
        self.val = not self.val

    def play(self):
        if not self.jump :
            i = random.random()
            print(i)
            state = self.make_state()
            state = self.move(state, i)
            move = state[2] # 조작값
            self.cycle_list.append(state)   # 한 사이클의 게임 데이터를 담음
            func = self.move_list[move] # 조작값에 따라 다른 조작함수 실행
            func()

    def move(self, state, num):  # 학습에 의한 조작값 선택
        temp_list = []
        for i in range(2) :
            temp = state[:] # 게임상태 복사
            temp.append(i)  # 0~ 2 조작값을 게임상태에 추가하여 게임데이터 생성
            temp = tuple(temp)
            if temp not in Man.dict.keys() :    # 게임 데이터 추가
                Man.dict[temp] = 0
                print('New Key!')
            temp_list.append((temp, Man.dict[temp]))    # 게임 데이터와 가중치를 리스트에 추가
        random.shuffle(temp_list)   # 가중치가 같다면 무작위 선택이 되도록 리스트 섞음
        if num < 0.1 :
            print(temp_list, temp_list[0])
            return temp_list[0][0]
        else :
            print(temp_list, max(temp_list, key = (lambda x : x[1])))
            return max(temp_list, key = (lambda x : x[1]))[0]   # 가중치가 큰 게임데이터를 선택함

    def make_state(self):   # 게임 상태 생성
        ball_id = self.find_ball()  # 캐릭터와 가장 가까운 공 선택
        self.canvas.itemconfig(ball_id, fill = 'red')   # 공에 붉게 표시
        ball_position = self.get_pos(ball_id)   # 공의 위치 반환
        pos = self.get_pos()    # 캐릭터 위치 반환
        return [tuple(int(i) for i in pos), tuple(int(i) for i in ball_position)] # (캐릭터 위치, 공위치) 반환

    def find_ball(self):    # 캐릭터와 가장 가까운 공 반환
        if len(self.ball.ball_list) > 0 and self.jump == False:	# 점프중이 아닐때만
            ball_over = []
            pos = self.get_pos()    # 캐릭터 위치 반환
            for i in self.ball.ball_list :
                ball_pos = self.canvas.coords(i[0])
                if ball_pos[3] <= pos[1] :  # 캐릭터보다 위에 위치한 공들을 리스트에 추가
                    ball_over.append((i[0], ball_pos[3]))
            return max(ball_over, key = (lambda x : x[1]))[0]   # y 값이 가장 큰 공을 반환

    def reset(self):    # 게임 초기화
        self.canvas.delete(self.id)
        self.pre_pos = None
        self.jump = False
        self.id = self.canvas.create_rectangle(5, 550, 35, 599)
        self.cycle_list = []

    def get_pos(self, x = None):    # 대상 혹은 캐릭터의 위치 반환
        x = x or self.id
        pos = self.canvas.coords(x)
        return (int((pos[0]+pos[2])/2), int(pos[3]))

    def draw(self): # 캐릭터 위치 변경
        base = self.get_pos()
        pos = self.canvas.coords(self.id)
        if not self.jump :  # 점프상태가 아닐때만 점프한 위치를 업데이트
            self.pre_pos = None

        if self.y == 5 :    # 점프중이 아닐때를 의미
            if 525 <= base[1] <= 600 and base[0] >= 100 :
                if self.y + base[1] >= math.ceil(-1 * (base[0] - 100) / 14 + 600) :
                    self.y = math.ceil(-1 * (base[0] - 100) / 14 + 600) - base[1]
                    self.jump = False
            elif 225 <= base[1] <= 300 and base[0] >= 100 :
                if self.y + base[1] >= math.ceil(-1 * (base[0] - 100) / 14 + 300) :
                    self.y = math.ceil(-1 * (base[0] - 100) / 14 + 300) - base[1]
                    self.jump = False
            elif 375 <= base[1] <= 450 and base[0] <= 700 :
                if self.y + base[1] >= math.ceil((base[0] / 14) + 400) :
                    self.y = math.ceil((base[0] / 14) + 400) - base[1]
                    self.jump = False
            elif 75 <= base[1] <= 150 and base[0] <= 700 :
                if self.y + base[1] >= math.ceil(base[0] / 14 + 100) :
                    self.y = math.ceil(base[0] / 14 + 100) - base[1]
                    self.jump = False
            elif (0 <= base[0] <= 100) and 525 <= base[1] <= 620:
                if self.y + base[1] > 599 :
                    self.y = 599 - base[1]
                    self.jump = False	# 착지한 상태이므로 점프상태를 False 로 변경
        else :  # 점프중일때
            self.jump = True
            self.pre_pos = self.pre_pos or self.get_pos()   # 점프한 지점의 위치를 저장
            if (base[1] == self.pre_pos[1] - 50) and self.y == -5 : # 점프한 지점에서 50픽셀만큼 뛴후부터 속력 감소
                self.y += 1
            elif (base[1] <= self.pre_pos[1] - 50) :    # y 속력이 점프 초기속력으로 회복될때까지 속력 변경
                self.y += 1

        if pos[0]+self.x < 0 or pos[2]+self.x > 799 :   # 화면 밖으로 나가지 않도록 설정
            if self.x < 0 :
                self.x = -pos[0]
            elif self.x > 0 :
                self.x = 799-pos[2]
        self.canvas.move(self.id, self.x, self.y)   # 캐릭터 위치 변경
        if not self.jump : self.y = 5   # 속도 초기화

    def collision(self):    # 게임 결과 판단
        pos = self.canvas.coords(self.id)
        goal_pos = self.canvas.coords(self.goal)
        for i in self.ball.ball_list :  # 공과 충돌시
            ball_pos = self.canvas.coords(i[0])
            if ball_pos[0] <= pos[2] and ball_pos[2] >= pos[0]:
                if ball_pos[1] <= pos[3] and ball_pos[3] >= pos[1] :
                    self.learning((600-pos[3])/100, True) # 올라간 높이만큼의 보상과 함께 처벌
                    self.reset()    # 게임 초기화
                    self.ball.reset()
                    self.game_num += 1  # 게임 수 증가
                    print(self.game_num)
                    self.save_csv() # 게임 데이터 저장
                    return
        if goal_pos[0] <= pos[2] and goal_pos[2] >= pos[0]: # 목표에 도착
            if goal_pos[1] <= pos[3] and goal_pos[3] >= pos[1]:
                self.learning((600-pos[3])/100, False) # 올라간 높이만큼 보상
                self.reset()
                self.ball.reset()
                self.game_num += 1
                print(self.game_num)
                self.save_csv()

    def learning(self, reward, dead): # 학습
        print(reward)
        if dead :	# 공과 부딪혔을때
            p = -8
            key1 = self.cycle_list.pop()
            Man.dict[key1] += p	# 공과 부딪힌 게임데이터 가중치에 -8
            p *= 0.9
		    # 어느 조작값을 갖던 죽는 상황에서는 그 상황을 만들어낸 직전 게임 데이터에 벌점
            while  Man.dict[(key1[0], key1[1], 0)] <0 and Man.dict[(key1[0], key1[1], 1)] < 0:
                key1 = self.cycle_list.pop()
                Man.dict[key1] += p
                p*= 0.9

        temp = []
        val = reward	# 보상
        self.cycle_list.reverse()
        for i in self.cycle_list:
            if temp.count(i) == 0 :
                pre_val = Man.dict[i]
                Man.dict[i] += 0.99 * (val - pre_val)
                val *= 0.99
                temp.append(i)

    # 캐릭터 조작 함수
    def jump_left(self):
        if not self.jump:
            base = self.get_pos()
            #self.jump = True
            if (((600 < base[0] < 700) and (base[1] > 550 or 250 < base[1] < 300))
                or ((100 < base[0] < 200) and (400 < base[1] < 450))):
                self.y = -15
            else:
                self.y = -5
            self.x = -4

    def jump_right(self):
        if not self.jump:
            base = self.get_pos()
            #self.jump = True
            if (((600 < base[0] < 700) and (base[1] > 550 or 250 < base[1] < 300))
                or ((100 < base[0] < 200) and (400 < base[1] < 450))):
                self.y = -15
            else:
                self.y = -5
            self.x = 4

tk = Tk()
tk.title('Game')
tk.resizable(0, 0)
tk.geometry("800x600+0+0")

canvas = Canvas(tk, width = 800, height = 600, bd = 0, highlightthickness= 0)
canvas.pack()
tk.update()
map = Map(canvas)
ball = Ball(canvas)
man = Man(canvas, ball)
time.sleep(2)

while True :
    man.play()
    ball.draw()
    man.draw()
    man.collision()
    tk.update_idletasks()
    tk.update()
    if man.val :
        time.sleep(0.01)