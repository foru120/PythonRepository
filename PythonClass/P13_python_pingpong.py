print('')
print('====================================================================================================')
print('== 문제 223. 완성된 pingpong 게임을 수행하시오.')
print('====================================================================================================')
from tkinter import *
import random
import time

class Ball:
    def __init__(self, canvas, paddle, color):
        self.canvas = canvas
        self.paddle = paddle
        self.id = canvas.create_oval(10, 10, 25, 25, fill=color)  # 공 크기 및 색깔

        self.canvas.move(self.id, 245, 100)  # 공을 캔버스 중앙으로 이동
        starts = [-3, -2, -1, 1, 2, 3]  # 공의 속도를 랜덤으로 구성하기 위해 준비한 리스트
        random.shuffle(starts)  # starts 리스트 중에 숫자를 랜덤으로 골라서
        self.x = starts[0]  # 처음 공이 패들에서 움직일때 왼쪽으로 올라갈지 오른쪽으로 올라갈지 랜덤으로 결정되는 부분
        self.y = -3  # 처음 공이 패들에서 움직일때 위로 올라가는 속도
        self.canvas_height = self.canvas.winfo_height()  # 캔버스의 현재 높이를 반환한다.(공이 화면에서 사라지지 않기위해)
        self.canvas_width = self.canvas.winfo_width()  # 캔버스의 현재 넓이를 반환한다.(공이 화면에서 사라지지 않기위해)
        self.hit_bottom = False

    def hit_paddle(self, pos):  # 패들에 공이 튀기게 하는 함수
        paddle_pos = self.canvas.coords(self.paddle.id)
        if pos[2] >= paddle_pos[0] and pos[0] <= paddle_pos[2]:  # 공이 패들에 내려오기 직전 좌표
            if pos[3] >= paddle_pos[1] and pos[3] <= paddle_pos[3]:  # 공이 패들에 닿았을때 좌표
                return True
        return False

    def draw(self):
        self.canvas.move(self.id, self.x, self.y)  # 공을 움직이게 하는 부분
        # 공이 화면 밖으로 나가지 않게 해준다
        pos = self.canvas.coords(self.id)  # 볼의 현재 좌표를 출력해준다. 공 좌표( 서쪽(0) , 남쪽(1) , 동쪽(2), 북쪽(3) )
        # [ 255,29,270,44]

        if pos[1] <= 0:  # 공의 남쪽이 가리키는 좌표가 0보다 작아진다면 공이 위쪽 화면 밖으로 나가버리므로
            self.y = 3  # 공을 아래로 떨어뜨린다. (공이 위로 올라갈수로 y 의 값이 작아지므로 아래로 내리려면 다시 양수로)
        if pos[3] >= self.canvas_height:  # 공의 북쪽이 가리키는 좌표가 캔버스의 높이보다 더 크다면 화면 아래로 나가버려서
            self.y = -3  # 공을 위로 올린다. (공이 아래로 내려갈수록 y 값이 커지므로 공을 위로 올릴려면 다시 음수로)
        if pos[0] <= 0:  # 공의 서쪽이 가리키는 좌표가 0보다 작으면 공이 화면 왼쪽으로 나가버리므로
            self.x = 3  # 공을 오른쪽으로 돌린다.
        if pos[2] >= self.canvas_width:  # 공의 동쪽이 가리키는 좌표가 공의 넓이보다 크다면 공이 화면 오른쪽으로 나가버림
            self.x = -3  # 공을 왼쪽으로 돌린다.
        if self.hit_paddle(pos) == True:  # 패들 판에 부딪히면 위로 튕겨올라가게
            self.y = -3  # 공을 위로 올린다.


class Paddle:
    def __init__(self, canvas, color):
        self.canvas = canvas
        self.id = canvas.create_rectangle(0, 0, 100, 10, fill=color)  # 패들의 높이와 넓이 그리고 색깔
        self.canvas.move(self.id, 200, 300)  # 패들 사각형을 200,300 에 위치
        self.x = 0  # 패들이 처음 시작할때 움직이지 않게 0으로 설정
        self.canvas_width = self.canvas.winfo_width()  # 캔버스의 넓이를 반환한다. 캔버스 밖으로 패들이 나가지 않도록
        self.canvas.bind_all('<KeyPress-Left>', self.turn_left)  # 왼쪽 화살표 키를 '<KeyPress-Left>'  라는 이름로 바인딩
        self.canvas.bind_all('<KeyPress-Right>', self.turn_right)  # 오른쪽도 마찬가지로 바인딩한다.

    def draw(self):
        self.canvas.move(self.id, self.x, 0)  # 시작할때 패들이 위아래로 움직이지 않도록 0 으로 설정
        pos = self.canvas.coords(self.id)
        print(pos)
        if pos[0] <= 0:  # 공의 서쪽이 가리키는 좌표가 0보다 작으면 공이 화면 왼쪽으로 나가버리므로
            self.x = 0  # 패들을 멈춰버린다.
        elif pos[2] >= self.canvas_width:  # 공의 동쪽이 캔버스의 넓이 보다 크면 공이 화면 오른쪽으로 나가버리므로
            self.x = 0  # 패들을 멈춰버린다

            # 패들이 화면의 끝에 부딪히면 공처럼 튕기는게 아니라 움직임이 멈춰야한다.
            # 그래서 왼쪽 x 좌표(pos[0]) 가 0 과 같거나 작으면 self.x = 0 처럼 x 변수에 0 을
            # 설정한다.  같은 방법으로 오른쪽 x 좌표(pos[2]) 가 캔버스의 폭과 같거나 크면
            # self.x = 0 처럼 변수에 0 을 설정한다.

    def turn_left(self, evt):  # 패들의 방향을 전환하는 함수
        self.x = -3

    def turn_right(self, evt):
        self.x = 3


tk = Tk()  # tk 를 인스턴스화 한다.
tk.title("Game")  # tk 객체의 title 메소드(함수)로 게임창에 제목을 부여한다.
tk.resizable(0, 0)  # 게임창의 크기는 가로나 세로로 변경될수 없다라고 말하는것이다.
tk.wm_attributes("-topmost", 1)  # 다른 모든 창들 앞에 캔버스를 가진 창이 위치할것을 tkinter 에게 알려준다.

canvas = Canvas(tk, width=500, height=400, bd=0, highlightthickness=0)
# bg=0,highlightthickness=0 은 캔버스 외곽에 둘러싼
# 외곽선이 없도록 하는것이다. (게임화면이 좀더 좋게)

canvas.pack()  # 앞의 코드에서 전달된 폭과 높이는 매개변수에 따라 크기를 맞추라고 캔버스에에 말해준다.
tk.update()  # tkinter 에게 게임에서의 애니메이션을 위해 자신을 초기화하라고 알려주는것이다.
paddle = Paddle(canvas, 'blue')
ball = Ball(canvas, paddle, 'red')
start = False
# 공을 약간 움직이고 새로운 위치로 화면을 다시 그리며, 잠깐 잠들었다가 다시 시작해 ! "
while 1:
    if ball.hit_bottom == False:
        ball.draw()
        paddle.draw()

    tk.update_idletasks()  # 우리가 창을 닫으라고 할때까지 계속해서 tkinter 에게 화면을 그려라 !
    tk.update()  # tkinter 에게 게임에서의 애니메이션을 위해 자신을 초기화하라고 알려주는것이다.
    time.sleep(0.01)  # 무한 루프중에 100분의 1초마다 잠들어라 !


print('')
print('====================================================================================================')
print('== 문제 225. 캔버스를 그리시오.')
print('====================================================================================================')
from tkinter import *
import random
import time

tk = Tk()      # 1. tk 를 인스턴스화 한다.
tk.title("Game")  # 2. tk 객체의 title 메소드(함수)로 게임창에 제목을 부여한다.
tk.resizable(0, 0) # 3. 게임창의 크기는 가로나 세로로 변경될수 없다라고 말하는것이다.
tk.wm_attributes("-topmost", 1) #4. 다른 모든 창들 앞에 캔버스를 가진 창이 위치할것을 tkinter 에게 알려준다.
canvas = Canvas(tk, width=500, height=400, bd=0, highlightthickness=0)
canvas.configure(background='black')
# bg=0,highlightthickness=0 은 캔버스 외곽에 둘러싼
# 외곽선이 없도록 하는것이다. (게임화면이 좀더 좋게)
canvas.pack()       # 앞의 코드에서 전달된 폭과 높이는 매개변수에 따라 크기를 맞추라고 캔버스에에 말해준다.
tk.update()   # tkinter 에게 게임에서의 애니메이션을 위해 자신을 초기화하라고 알려주는것이다.

#창이 바로 꺼지는것을 막을려면 ?   mainloop() 라 불리는 애니메이션 루프를 추가해야한다.

tk.mainloop()


print('')
print('====================================================================================================')
print('== 문제 229. pingpong 소스를 분석하시오.')
print('====================================================================================================')
#  1. 캔버스 클래스
#   - 캔버스 크기와 색깔
#  2. 공 클래스
#   - init 함수
#    1) 공의 크기, 색깔
#    2) 게임이 시작할 때 공의 첫 위치
#    3) 게임이 시작할 때 공이 움직이는 방향(랜덤으로)
#    4) 게임이 시작할 때 공이 위로 움직이는 속도
#    5) 공이 화면에서 사라지지 않게 하려고 할 때 필요한 정보를 모으는 코드
#   - 공을 움직이게 하는 함수
#   - 공이 패들에서 튀기게 하는 함수
#  3. 패들 클래스
#   - init 함수
#   - 패들을 움직이는 함수
#   - 패들이 화면 밖으로 안나가게하는 함수

from tkinter import *
import random
import time

class Ball:

    def __init__(self, canvas, paddle, color):
        self.canvas = canvas
        self.paddle = paddle
        self.id = canvas.create_oval(10, 10, 25, 25, fill=color)
        canvas.configure(background='black')
        self.canvas.move(self.id, 245, 100)
        starts = [-3, -2, -1, 1, 2, 3]
        random.shuffle(starts)

        self.x = starts[0]
        self.y = -3
        self.canvas_height = self.canvas.winfo_height()
        self.canvas_width = self.canvas.winfo_width()
        self.hit_bottom = False

    def draw(self):
        self.canvas.move(self.id, self.x, self.y)

        pos = self.canvas.coords(self.id)
        if pos[1] <= 0:
            self.y = 3
        if pos[3] >= self.canvas_height:
            self.hit_bottom = True
        if pos[0] <= 0:
            self.x = 3
        if pos[2] >= self.canvas_width:
            self.x = -3
        if self.hit_paddle(pos) == True:
            self.y = -3


    def hit_paddle(self,pos):
        paddle_pos = self.canvas.coords(self.paddle.id)
        if pos[2] >= paddle_pos[0] and pos[0] <= paddle_pos[2]:
            if pos[3] >= paddle_pos[1] and pos[1] <= paddle_pos[3]:
                return True
        return False


class Paddle:

    def __init__(self,canvas,color):
        self.canvas = canvas
        self.id = canvas.create_rectangle(0,0,100,10,fill=color)
        self.canvas.move(self.id, 200, 400)
        self.x = 0
        self.canvas_width = self.canvas.winfo_width()
        self.canvas.bind_all('<KeyPress-Left>',self.turn_left)
        self.canvas.bind_all('<KeyPress-Right>',self.turn_right)

    def draw(self):
        self.canvas.move(self.id, self.x, 0)
        pos = self.canvas.coords(self.id)
        if pos[0] <= 0:
            self.x = 0
        elif pos[2] >= self.canvas_width:
            self.x = 0

    def turn_left(self,evt):
        self.x = -9

    def turn_right(self,evt):
        self.x = 9

tk = Tk()
tk.title("Game")
tk.resizable(0, 0)
tk.wm_attributes("-topmost", 1)
canvas = Canvas(tk, width=600, height=500, bd=0, highlightthickness=0)
canvas.pack()
tk.update()
paddle = Paddle(canvas,'white')
ball = Ball(canvas, paddle, 'white')

while 1:

    ball.draw()
    paddle.draw()
    tk.update_idletasks()
    tk.update()
    time.sleep(0.02)


print('')
print('====================================================================================================')
print('== 문제 230. 캔버스의 공의 첫 시작 위치가 천장위가 되게 하시오.')
print('====================================================================================================')
from tkinter import *
import random
import time

class Ball:

    def __init__(self, canvas, paddle, color):
        self.canvas = canvas
        self.paddle = paddle
        self.id = canvas.create_oval(10, 10, 25, 25, fill=color)
        canvas.configure(background='black')
        self.canvas.move(self.id, 245, 100) # 공 시작 위치 설정
        starts = [-3, -2, -1, 1, 2, 3]
        random.shuffle(starts)

        self.x = starts[0]
        self.y = -3
        self.canvas_height = self.canvas.winfo_height()
        self.canvas_width = self.canvas.winfo_width()
        self.hit_bottom = False

    def draw(self):
        self.canvas.move(self.id, self.x, self.y)

        pos = self.canvas.coords(self.id)
        if pos[1] <= 0:
            self.y = 3
        if pos[3] >= self.canvas_height:
            self.hit_bottom = True
        if pos[0] <= 0:
            self.x = 3
        if pos[2] >= self.canvas_width:
            self.x = -3
        if self.hit_paddle(pos) == True:
            self.y = -3


    def hit_paddle(self,pos):
        paddle_pos = self.canvas.coords(self.paddle.id)
        if pos[2] >= paddle_pos[0] and pos[0] <= paddle_pos[2]:
            if pos[3] >= paddle_pos[1] and pos[1] <= paddle_pos[3]:
                return True
        return False


class Paddle:

    def __init__(self,canvas,color):
        self.canvas = canvas
        self.id = canvas.create_rectangle(0,0,100,10,fill=color)
        self.canvas.move(self.id, 200, 400)
        self.x = 0
        self.canvas_width = self.canvas.winfo_width()
        self.canvas.bind_all('<KeyPress-Left>',self.turn_left)
        self.canvas.bind_all('<KeyPress-Right>',self.turn_right)

    def draw(self):
        self.canvas.move(self.id, self.x, 0)
        pos = self.canvas.coords(self.id)
        if pos[0] <= 0:
            self.x = 0
        elif pos[2] >= self.canvas_width:
            self.x = 0

    def turn_left(self,evt):
        self.x = -9

    def turn_right(self,evt):
        self.x = 9

tk = Tk()
tk.title("Game")
tk.resizable(0, 0)
tk.wm_attributes("-topmost", 1)
canvas = Canvas(tk, width=600, height=500, bd=0, highlightthickness=0)
canvas.pack()
tk.update()
paddle = Paddle(canvas,'white')
ball = Ball(canvas, paddle, 'white')

while 1:

    ball.draw()
    paddle.draw()
    tk.update_idletasks()
    tk.update()
    time.sleep(0.02)


print('')
print('====================================================================================================')
print('== 문제 230. 게임이 시작할 때 공이 왼쪽, 오른쪽 중 랜덤으로 가게하지말고 무조건 오른쪽으로 가게 하려면?')
print('====================================================================================================')
from tkinter import *
import random
import time

class Ball:

    def __init__(self, canvas, paddle, color):
        self.canvas = canvas
        self.paddle = paddle
        self.id = canvas.create_oval(10, 10, 25, 25, fill=color)
        canvas.configure(background='black')
        self.canvas.move(self.id, 245, 100) # 공 시작 위치 설정
        starts = [1, 2, 3]  # x축 방향 설정
        random.shuffle(starts)

        self.x = starts[0]
        self.y = -3
        self.canvas_height = self.canvas.winfo_height()
        self.canvas_width = self.canvas.winfo_width()
        self.hit_bottom = False  # 바닥에 닿으면 게임 끝나는 코드를 구현하기 위해서 쓰는 변수
        self.canvas.bind_all('<KeyPress-Left>', self.turn_left)
        self.canvas.bind_all('<KeyPress-Right>', self.turn_right)
        self.canvas.bind_all('<KeyPress-Up>', self.turn_up)
        self.canvas.bind_all('<KeyPress-Down>', self.turn_down)

    def draw(self):
        self.canvas.move(self.id, self.x, self.y)

        pos = self.canvas.coords(self.id)
        if pos[1] <= 0:
            self.y = 3
        if pos[3] >= self.canvas_height:
            self.hit_bottom = True
        if pos[0] <= 0:
            self.x = 3
        if pos[2] >= self.canvas_width:
            self.x = -3
        if self.hit_paddle(pos) == True:
            self.y = -3

    def turn_left(self,evt):
        self.x = -9

    def turn_right(self,evt):
        self.x = 9

    def turn_up(self,evt):
        self.y = -9

    def turn_down(self,evt):
        self.y = 9


    def hit_paddle(self,pos):
        paddle_pos = self.canvas.coords(self.paddle.id)
        if pos[2] >= paddle_pos[0] and pos[0] <= paddle_pos[2]:
            if pos[3] >= paddle_pos[1] and pos[1] <= paddle_pos[3]:
                return True
        return False


class Paddle:

    def __init__(self,canvas,color):
        self.canvas = canvas
        self.id = canvas.create_rectangle(0,0,100,10,fill=color)
        self.canvas.move(self.id, 200, 400)
        self.x = 0
        self.canvas_width = self.canvas.winfo_width()
        self.canvas.bind_all('<KeyPress-Left>',self.turn_left)
        self.canvas.bind_all('<KeyPress-Right>',self.turn_right)

    def draw(self):
        self.canvas.move(self.id, self.x, 0)
        pos = self.canvas.coords(self.id)
        if pos[0] <= 0:
            self.x = 0
        elif pos[2] >= self.canvas_width:
            self.x = 0

    def turn_left(self,evt):
        self.x = -9

    def turn_right(self,evt):
        self.x = 9

tk = Tk()
tk.title("Game")
tk.resizable(0, 0)
tk.wm_attributes("-topmost", 1)
canvas = Canvas(tk, width=600, height=500, bd=0, highlightthickness=0)
canvas.pack()
tk.update()
paddle = Paddle(canvas,'white')
ball = Ball(canvas, paddle, 'white')

while 1:

    ball.draw()
    paddle.draw()
    tk.update_idletasks()
    tk.update()
    time.sleep(0.02)


print('')
print('====================================================================================================')
print('== 문제 230. 공이 패들에 닿으면 공이 딱 멈춰지게하고 스페이스 바를 누르면 게임이 다시 시작되게 하시오.')
print('====================================================================================================')
from tkinter import *
import random
import time


class Ball:
    def __init__(self, canvas, paddle, color):
        self.canvas = canvas
        self.paddle = paddle
        self.id = canvas.create_oval(10, 10, 25, 25, fill=color)
        canvas.configure(background='black')
        self.canvas.move(self.id, 245, 100)  # 공 시작 위치 설정
        starts = [-3, -2, -1, 1, 2, 3]  # x축 방향 설정
        random.shuffle(starts)

        self.x = starts[0]
        self.y = -3
        self.canvas_height = self.canvas.winfo_height()
        self.canvas_width = self.canvas.winfo_width()
        self.hit_bottom = False  # 바닥에 닿으면 게임 끝나는 코드를 구현하기 위해서 쓰는 변수

        self.canvas.bind_all('<Key>', self.event_method)

    def draw(self):
        self.canvas.move(self.id, self.x, self.y)

        pos = self.canvas.coords(self.id)
        if pos[1] <= 0:
            self.y = 3
        if pos[3] >= self.canvas_height:
            self.hit_bottom = True
        if pos[0] <= 0:
            self.x = 3
        if pos[2] >= self.canvas_width:
            self.x = -3
        if self.hit_paddle(pos) == True:
            self.x = 0
            self.y = 0

    def event_method(self, evt):
        x_starts = [-3, -2, -1, 1, 2, 3]
        y_starts = [-1, -2, -3]
        random.shuffle(x_starts)
        random.shuffle(y_starts)
        self.x = x_starts[0]
        self.y = y_starts[0]

    def hit_paddle(self, pos):
        paddle_pos = self.canvas.coords(self.paddle.id)
        if pos[2] >= paddle_pos[0] and pos[0] <= paddle_pos[2]:
            if pos[3] >= paddle_pos[1] and pos[1] <= paddle_pos[3]:
                return True
        return False


class Paddle:
    def __init__(self, canvas, color):
        self.canvas = canvas
        self.id = canvas.create_rectangle(0, 0, 100, 10, fill=color)
        self.canvas.move(self.id, 200, 400)
        self.x = 0
        self.canvas_width = self.canvas.winfo_width()
        self.canvas.bind_all('<KeyPress-Left>', self.turn_left)
        self.canvas.bind_all('<KeyPress-Right>', self.turn_right)

    def draw(self):
        self.canvas.move(self.id, self.x, 0)
        pos = self.canvas.coords(self.id)
        if pos[0] <= 0:
            self.x = 0
        elif pos[2] >= self.canvas_width:
            self.x = 0

    def turn_left(self, evt):
        self.x = -9

    def turn_right(self, evt):
        self.x = 9


tk = Tk()
tk.title("Game")
tk.resizable(0, 0)
tk.wm_attributes("-topmost", 1)
canvas = Canvas(tk, width=600, height=500, bd=0, highlightthickness=0)
canvas.pack()
tk.update()
paddle = Paddle(canvas, 'white')
ball = Ball(canvas, paddle, 'white')

while 1:
    ball.draw()
    paddle.draw()
    tk.update_idletasks()
    tk.update()
    time.sleep(0.02)


print('')
print('====================================================================================================')
print('== 문제 238. 공이 멈춰진 이후에 패들을 따라 움직여 지게 하시오!')
print('====================================================================================================')
from tkinter import *
import random
import time


class Ball:
    def __init__(self, canvas, paddle, color):
        self.canvas = canvas
        self.paddle = paddle
        self.id = canvas.create_oval(10, 10, 25, 25, fill=color)
        canvas.configure(background='black')
        self.canvas.move(self.id, 245, 100)  # 공 시작 위치 설정
        starts = [-3, -2, -1, 1, 2, 3]  # x축 방향 설정
        random.shuffle(starts)

        self.x = starts[0]
        self.y = -3
        self.canvas_height = self.canvas.winfo_height()
        self.canvas_width = self.canvas.winfo_width()
        self.hit_bottom = False  # 바닥에 닿으면 게임 끝나는 코드를 구현하기 위해서 쓰는 변수
        self.canvas.bind_all('<Key>', self.event_method)

    def draw(self):
        self.canvas.move(self.id, self.x, self.y)

        pos = self.canvas.coords(self.id)
        if pos[1] <= 0:
            self.y = 3
        if pos[3] >= self.canvas_height:
            self.hit_bottom = True
        if pos[0] <= 0:
            self.x = 3
        if pos[2] >= self.canvas_width:
            self.x = -3
        if self.hit_paddle(pos) == True:
            self.x = self.paddle.x
            self.y = 0

    def event_method(self, evt):
        print(evt)
        x_starts = [-3, -2, -1, 1, 2, 3]
        y_starts = [-1, -2, -3]
        random.shuffle(x_starts)
        random.shuffle(y_starts)
        self.x = x_starts[0]
        self.y = y_starts[0]

    def hit_paddle(self, pos):
        paddle_pos = self.canvas.coords(self.paddle.id)
        if pos[2] >= paddle_pos[0] and pos[0] <= paddle_pos[2]:
            if pos[3] >= paddle_pos[1] and pos[1] <= paddle_pos[3]:
                return True
        return False


class Paddle:
    def __init__(self, canvas, color):
        self.canvas = canvas
        self.id = canvas.create_rectangle(0, 0, 100, 10, fill=color)
        self.canvas.move(self.id, 200, 400)
        self.x = 0
        self.canvas_width = self.canvas.winfo_width()
        self.canvas.bind_all('<KeyPress-Left>', self.turn_left)
        self.canvas.bind_all('<KeyPress-Right>', self.turn_right)

    def draw(self):
        self.canvas.move(self.id, self.x, 0)
        pos = self.canvas.coords(self.id)
        if pos[0] <= 0:
            self.x = 0
        elif pos[2] >= self.canvas_width:
            self.x = 0

    def turn_left(self, evt):
        self.x = -9

    def turn_right(self, evt):
        self.x = 9


tk = Tk()
tk.title("Game")
tk.resizable(0, 0)
tk.wm_attributes("-topmost", 1)
canvas = Canvas(tk, width=600, height=500, bd=0, highlightthickness=0)
canvas.pack()
tk.update()
paddle = Paddle(canvas, 'white')
ball = Ball(canvas, paddle, 'white')

while 1:
    ball.draw()
    paddle.draw()
    tk.update_idletasks()
    tk.update()
    time.sleep(0.02)


print('')
print('====================================================================================================')
print('== 문제 239. 이 상황에서 스페이스 바를 눌렀을 때 공이 위로 올라가게 하시오.')
print('====================================================================================================')
from tkinter import *
import random
import time


class Ball:
    def __init__(self, canvas, paddle, color):
        self.canvas = canvas
        self.paddle = paddle
        self.id = canvas.create_oval(10, 10, 25, 25, fill=color)
        canvas.configure(background='black')
        self.canvas.move(self.id, 245, 100)  # 공 시작 위치 설정
        starts = [-3, -2, -1, 1, 2, 3]  # x축 방향 설정
        random.shuffle(starts)

        self.x = starts[0]
        self.y = -3
        self.canvas_height = self.canvas.winfo_height()
        self.canvas_width = self.canvas.winfo_width()
        self.hit_bottom = False  # 바닥에 닿으면 게임 끝나는 코드를 구현하기 위해서 쓰는 변수
        self.canvas.bind_all('<Key>', self.event_method)

    def draw(self):
        self.canvas.move(self.id, self.x, self.y)

        pos = self.canvas.coords(self.id)
        if pos[1] <= 0:
            self.y = 3
        if pos[3] >= self.canvas_height:
            self.hit_bottom = True
        if pos[0] <= 0:
            self.x = 3
        if pos[2] >= self.canvas_width:
            self.x = -3
        if self.hit_paddle(pos) == True:
            self.x = self.paddle.x
            self.y = 0

    def event_method(self, evt):
        print(evt)
        x_starts = [-3, -2, -1, 1, 2, 3]
        y_starts = [-1, -2, -3]
        random.shuffle(x_starts)
        random.shuffle(y_starts)
        self.x = x_starts[0]
        self.y = y_starts[0]

    def hit_paddle(self, pos):
        paddle_pos = self.canvas.coords(self.paddle.id)
        if pos[2] >= paddle_pos[0] and pos[0] <= paddle_pos[2]:
            if pos[3] >= paddle_pos[1] and pos[1] <= paddle_pos[3]:
                return True
        return False


class Paddle:
    def __init__(self, canvas, color):
        self.canvas = canvas
        self.id = canvas.create_rectangle(0, 0, 100, 10, fill=color)
        self.canvas.move(self.id, 200, 400)
        self.x = 0
        self.canvas_width = self.canvas.winfo_width()
        self.canvas.bind_all('<KeyPress-Left>', self.turn_left)
        self.canvas.bind_all('<KeyPress-Right>', self.turn_right)

    def draw(self):
        self.canvas.move(self.id, self.x, 0)
        pos = self.canvas.coords(self.id)
        if pos[0] <= 0:
            self.x = 0
        elif pos[2] >= self.canvas_width:
            self.x = 0

    def turn_left(self, evt):
        self.x = -9

    def turn_right(self, evt):
        self.x = 9


tk = Tk()
tk.title("Game")
tk.resizable(0, 0)
tk.wm_attributes("-topmost", 1)
canvas = Canvas(tk, width=600, height=500, bd=0, highlightthickness=0)
canvas.pack()
tk.update()
paddle = Paddle(canvas, 'white')
ball = Ball(canvas, paddle, 'white')

while 1:
    ball.draw()
    paddle.draw()
    tk.update_idletasks()
    tk.update()
    time.sleep(0.02)


# Paddle 클래스를 이해하기 위한 문제들
print('')
print('====================================================================================================')
print('== 문제 240. 패들이 위 아래로도 움직이게 하시오!')
print('====================================================================================================')
from tkinter import *
import random
import time


class Ball:
    def __init__(self, canvas, paddle, color):
        self.canvas = canvas
        self.paddle = paddle
        self.id = canvas.create_oval(10, 10, 25, 25, fill=color)
        canvas.configure(background='black')
        self.canvas.move(self.id, 245, 100)  # 공 시작 위치 설정
        starts = [-3, -2, -1, 1, 2, 3]  # x축 방향 설정
        random.shuffle(starts)

        self.x = starts[0]
        self.y = -3
        self.canvas_height = self.canvas.winfo_height()
        self.canvas_width = self.canvas.winfo_width()
        self.hit_bottom = False  # 바닥에 닿으면 게임 끝나는 코드를 구현하기 위해서 쓰는 변수
        self.canvas.bind_all('<Key>', self.event_method)

    def draw(self):
        self.canvas.move(self.id, self.x, self.y)

        pos = self.canvas.coords(self.id)
        if pos[1] <= 0:
            self.y = 3
        if pos[3] >= self.canvas_height:
            self.hit_bottom = True
        if pos[0] <= 0:
            self.x = 3
        if pos[2] >= self.canvas_width:
            self.x = -3
        if self.hit_paddle(pos) == True:
            self.x = self.paddle.x
            self.y = 0

    def event_method(self, evt):
        print(evt)
        x_starts = [-3, -2, -1, 1, 2, 3]
        y_starts = [-1, -2, -3]
        random.shuffle(x_starts)
        random.shuffle(y_starts)
        self.x = x_starts[0]
        self.y = y_starts[0]

    def hit_paddle(self, pos):
        paddle_pos = self.canvas.coords(self.paddle.id)
        if pos[2] >= paddle_pos[0] and pos[0] <= paddle_pos[2]:
            if pos[3] >= paddle_pos[1] and pos[1] <= paddle_pos[3]:
                return True
        return False


class Paddle:
    def __init__(self, canvas, color):
        self.canvas = canvas
        self.id = canvas.create_rectangle(0, 0, 100, 10, fill=color)
        self.canvas.move(self.id, 200, 400)
        self.x = 0
        self.canvas_width = self.canvas.winfo_width()
        self.canvas.bind_all('<KeyPress-Left>', self.turn_left)
        self.canvas.bind_all('<KeyPress-Right>', self.turn_right)
        self.canvas.bind_all('<KeyPress-Up>', self.turn_right)
        self.canvas.bind_all('<KeyPress-Down>', self.turn_right)

    def draw(self):
        self.canvas.move(self.id, self.x, 0)
        pos = self.canvas.coords(self.id)
        if pos[0] <= 0:
            self.x = 0
        elif pos[2] >= self.canvas_width:
            self.x = 0

    def turn_left(self, evt):
        self.x = -9

    def turn_right(self, evt):
        print(evt)
        self.x = 9


tk = Tk()
tk.title("Game")
tk.resizable(0, 0)
tk.wm_attributes("-topmost", 1)
canvas = Canvas(tk, width=600, height=500, bd=0, highlightthickness=0)
canvas.pack()
tk.update()
paddle = Paddle(canvas, 'white')
ball = Ball(canvas, paddle, 'white')

while 1:
    ball.draw()
    paddle.draw()
    tk.update_idletasks()
    tk.update()
    time.sleep(0.02)


print('')
print('====================================================================================================')
print('== 문제 241. 패들이 밖으로 안나가게 하시오!')
print('====================================================================================================')
from tkinter import *
import random
import time


class Ball:
    def __init__(self, canvas, paddle, color):
        self.canvas = canvas
        self.paddle = paddle
        self.id = canvas.create_oval(10, 10, 25, 25, fill=color)
        canvas.configure(background='black')
        self.canvas.move(self.id, 245, 100)  # 공 시작 위치 설정
        starts = [-3, -2, -1, 1, 2, 3]  # x축 방향 설정
        random.shuffle(starts)

        self.x = starts[0]
        self.y = -3
        self.canvas_height = self.canvas.winfo_height()
        self.canvas_width = self.canvas.winfo_width()
        self.hit_bottom = False  # 바닥에 닿으면 게임 끝나는 코드를 구현하기 위해서 쓰는 변수
        self.canvas.bind_all('<Key>', self.key_event)

    def draw(self):
        self.canvas.move(self.id, self.x, self.y)

        pos = self.canvas.coords(self.id)
        if pos[1] <= 0:
            self.y = 3
        if pos[3] >= self.canvas_height:
            self.hit_bottom = True
        if pos[0] <= 0:
            self.x = 3
        if pos[2] >= self.canvas_width:
            self.x = -3
        if self.hit_paddle(pos) == True:
            self.x = self.paddle.x
            self.y = self.paddle.y

    def key_event(self, evt):
        if evt.keysym == 'space':
            x_starts = [-3, -2, -1, 1, 2, 3]
            y_starts = [-1, -2, -3]
            random.shuffle(x_starts)
            random.shuffle(y_starts)
            self.x = x_starts[0]
            self.y = y_starts[0]

    def hit_paddle(self, pos):
        paddle_pos = self.canvas.coords(self.paddle.id)
        if pos[2] >= paddle_pos[0] and pos[0] <= paddle_pos[2]:
            if pos[3] >= paddle_pos[1] and pos[1] <= paddle_pos[3]:
                return True
        return False


class Paddle:
    def __init__(self, canvas, color):
        self.canvas = canvas
        self.id = canvas.create_rectangle(0, 0, 100, 10, fill=color)
        self.canvas.move(self.id, 200, 400)
        self.x = 0
        self.y = 0
        self.canvas_width = self.canvas.winfo_width()
        self.canvas.bind_all('<KeyPress-Left>', self.key_handler)
        self.canvas.bind_all('<KeyPress-Right>', self.key_handler)
        self.canvas.bind_all('<KeyPress-Up>', self.key_handler)
        self.canvas.bind_all('<KeyPress-Down>', self.key_handler)

    def draw(self):

        pos = self.canvas.coords(self.id)

        if pos[0] <= 0 and self.x < 0:
            self.x = 0
        elif pos[2] >= self.canvas_width and self.x > 0:
            self.x = 0

        self.canvas.move(self.id, self.x, self.y)

    def key_handler(self, event):
        if event.keysym == 'Left':
            self.x = -9
        elif event.keysym == 'Right':
            self.x = 9
        elif event.keysym == 'Up':
            self.y = -5
        elif event.keysym == 'Down':
            self.y = 5


tk = Tk()
tk.title("Game")
tk.resizable(0, 0)
tk.wm_attributes("-topmost", 1)
canvas = Canvas(tk, width=600, height=500, bd=0, highlightthickness=0)
canvas.pack()
tk.update()
paddle = Paddle(canvas, 'white')
ball = Ball(canvas, paddle, 'white')

while 1:
    ball.draw()
    paddle.draw()
    tk.update_idletasks()
    tk.update()
    time.sleep(0.02)


print('')
print('====================================================================================================')
print('== 문제 241. 공이 패들에 닿으면 hit, y축 400 좌표를 지나치면 miss 가 출력되게 하시오.')
print('====================================================================================================')
from tkinter import *
import random
import time


class Ball:
    def __init__(self, canvas, paddle, color):
        self.canvas = canvas
        self.paddle = paddle
        self.id = canvas.create_oval(10, 10, 25, 25, fill=color)
        canvas.configure(background='black')
        self.canvas.move(self.id, 245, 100)  # 공 시작 위치 설정
        starts = [-3, -2, -1, 1, 2, 3]  # x축 방향 설정
        random.shuffle(starts)

        self.x = starts[0]
        self.y = -3
        self.canvas_height = self.canvas.winfo_height()
        self.canvas_width = self.canvas.winfo_width()
        self.hit_bottom = False  # 바닥에 닿으면 게임 끝나는 코드를 구현하기 위해서 쓰는 변수
        self.canvas.bind_all('<Key>', self.key_event)
        self.ismiss = False

    def draw(self):
        self.canvas.move(self.id, self.x, self.y)

        pos = self.canvas.coords(self.id)
        self.gameover(pos)

        if pos[1] <= 0:
            self.y = 3
        if pos[3] >= self.canvas_height:
            self.y = -3
            self.hit_bottom = True
        if pos[0] <= 0:
            self.x = 3
        if pos[2] >= self.canvas_width:
            self.x = -3
        if self.hit_paddle(pos) == True:
            self.x = self.paddle.x
            self.y = self.paddle.y
            print('hit!!')

    def gameover(self, pos):
        if (pos[3] >= 400) and (self.ismiss == False):
            print('miss')
            self.ismiss = True
        elif pos[3] < 400:
            self.ismiss = False

    def key_event(self, evt):
        if evt.keysym == 'space':
            x_starts = [-3, -2, -1, 1, 2, 3]
            y_starts = [-1, -2, -3]
            random.shuffle(x_starts)
            random.shuffle(y_starts)
            self.x = x_starts[0]
            self.y = y_starts[0]

    def hit_paddle(self, pos):
        paddle_pos = self.canvas.coords(self.paddle.id)
        if pos[2] >= paddle_pos[0] and pos[0] <= paddle_pos[2]:
            if pos[3] >= paddle_pos[1] and pos[1] <= paddle_pos[3]:
                return True
        return False


class Paddle:
    def __init__(self, canvas, color):
        self.canvas = canvas
        self.id = canvas.create_rectangle(0, 0, 100, 10, fill=color)
        self.canvas.move(self.id, 200, 400)
        self.x = 0
        self.y = 0
        self.canvas_width = self.canvas.winfo_width()
        self.canvas.bind_all('<KeyPress-Left>', self.key_handler)
        self.canvas.bind_all('<KeyPress-Right>', self.key_handler)
        self.canvas.bind_all('<KeyPress-Up>', self.key_handler)
        self.canvas.bind_all('<KeyPress-Down>', self.key_handler)

    def draw(self):

        pos = self.canvas.coords(self.id)

        if pos[0] <= 0 and self.x < 0:
            self.x = 0
        elif pos[2] >= self.canvas_width and self.x > 0:
            self.x = 0

        self.canvas.move(self.id, self.x, self.y)

    def key_handler(self, event):
        if event.keysym == 'Left':
            self.x = -9
        elif event.keysym == 'Right':
            self.x = 9
        elif event.keysym == 'Up':
            self.y = -5
        elif event.keysym == 'Down':
            self.y = 5


tk = Tk()
tk.title("Game")
tk.resizable(0, 0)
tk.wm_attributes("-topmost", 1)
canvas = Canvas(tk, width=600, height=500, bd=0, highlightthickness=0)
canvas.pack()
tk.update()
paddle = Paddle(canvas, 'white')
ball = Ball(canvas, paddle, 'white')

while 1:
    ball.draw()
    paddle.draw()
    tk.update_idletasks()
    tk.update()
    time.sleep(0.02)


print('')
print('====================================================================================================')
print('== 문제 244. 게임을 시작할때 화면이 멈춰있다가 스페이스바를 눌러야 공이 움직이면서 게임이 시작될 수 있도록 하시오!')
print('====================================================================================================')
from tkinter import *
import random
import time


class Ball:
    def __init__(self, canvas, paddle, color):
        self.canvas = canvas
        self.paddle = paddle
        self.id = canvas.create_oval(10, 10, 25, 25, fill=color)
        canvas.configure(background='black')
        self.canvas.move(self.id, 245, 100)  # 공 시작 위치 설정
        self.isstart = False
        self.x = 0
        self.y = 0
        self.canvas_height = self.canvas.winfo_height()
        self.canvas_width = self.canvas.winfo_width()
        self.hit_bottom = False  # 바닥에 닿으면 게임 끝나는 코드를 구현하기 위해서 쓰는 변수
        self.canvas.bind_all('<Key>', self.key_event)
        self.ismiss = False

    def draw(self):
        self.canvas.move(self.id, self.x, self.y)

        pos = self.canvas.coords(self.id)
        self.gameover(pos)

        if pos[1] <= 0:
            self.y = 3
        if pos[3] >= self.canvas_height:
            self.y = -3
            self.hit_bottom = True
        if pos[0] <= 0:
            self.x = 3
        if pos[2] >= self.canvas_width:
            self.x = -3
        if self.hit_paddle(pos) == True:
            self.x = self.paddle.x
            self.y = self.paddle.y
            print('hit!!')

    def gameover(self, pos):
        if (pos[3] >= 400) and (self.ismiss == False):
            print('miss')
            self.ismiss = True
        elif pos[3] < 400:
            self.ismiss = False

    def key_event(self, evt):
        if self.isstart == False:
            starts = [-3, -2, -1, 1, 2, 3]  # x축 방향 설정
            random.shuffle(starts)

            self.x = starts[0]
            self.y = -3
            self.isstart = True
        else:
            x_starts = [-3, -2, -1, 1, 2, 3]
            y_starts = [-1, -2, -3]
            random.shuffle(x_starts)
            random.shuffle(y_starts)
            self.x = x_starts[0]
            self.y = y_starts[0]

    def hit_paddle(self, pos):
        paddle_pos = self.canvas.coords(self.paddle.id)
        if pos[2] >= paddle_pos[0] and pos[0] <= paddle_pos[2]:
            if pos[3] >= paddle_pos[1] and pos[1] <= paddle_pos[3]:
                return True
        return False


class Paddle:
    def __init__(self, canvas, color):
        self.canvas = canvas
        self.id = canvas.create_rectangle(0, 0, 100, 10, fill=color)
        self.canvas.move(self.id, 200, 400)
        self.x = 0
        self.y = 0
        self.canvas_width = self.canvas.winfo_width()
        self.canvas.bind_all('<KeyPress-Left>', self.key_handler)
        self.canvas.bind_all('<KeyPress-Right>', self.key_handler)
        self.canvas.bind_all('<KeyPress-Up>', self.key_handler)
        self.canvas.bind_all('<KeyPress-Down>', self.key_handler)

    def draw(self):

        pos = self.canvas.coords(self.id)

        if pos[0] <= 0 and self.x < 0:
            self.x = 0
        elif pos[2] >= self.canvas_width and self.x > 0:
            self.x = 0

        self.canvas.move(self.id, self.x, self.y)

    def key_handler(self, event):
        if event.keysym == 'Left':
            self.x = -9
        elif event.keysym == 'Right':
            self.x = 9
        elif event.keysym == 'Up':
            self.y = -5
        elif event.keysym == 'Down':
            self.y = 5


tk = Tk()
tk.title("Game")
tk.resizable(0, 0)
tk.wm_attributes("-topmost", 1)
canvas = Canvas(tk, width=600, height=500, bd=0, highlightthickness=0)
canvas.pack()
tk.update()
paddle = Paddle(canvas, 'white')
ball = Ball(canvas, paddle, 'white')

while 1:
    ball.draw()
    paddle.draw()
    tk.update_idletasks()
    tk.update()
    time.sleep(0.02)


# Ping Pong 게임 머신러닝 코드 구현
#  1. ping pong 게임이 되게 하는 코드
#   1) canvas class
#   2) ball class
#   3) paddle class
#   4) 메인함수(무한 루프)

#  2. ping pong 데이터를 학습 시키기 위한 코드
#   1) greedy 함수
#   2) lookup 함수
#   3) add 함수
#   4) statetuple 함수
#   5) winnerval 함수
#   6) emptystate 함수
#   7) action 함수
#   8) backup 함수

# ■ winnerval 함수 : 보상을 위한 데이터를 출력하는 함수 (1, -1, 0)


print('')
print('====================================================================================================')
print('== 문제 245. ping pong 게임을 위한 winnerval 함수를 생성하는데 hit 일때는 1이 리턴되고, miss 일때는 -1이 리턴되고')
print('==  그 외는 (게임이 끝나지 않았을 때는) 0을 리턴되게 생성하시오.')
print('====================================================================================================')
from tkinter import *
import random
import time


class Ball:
    def __init__(self, canvas, paddle, color):
        self.canvas = canvas
        self.paddle = paddle
        self.id = canvas.create_oval(10, 10, 25, 25, fill=color)
        canvas.configure(background='black')
        self.canvas.move(self.id, 245, 100)  # 공 시작 위치 설정
        self.isstart = False
        self.ismiss = False
        self.ishit = False
        self.x = 0
        self.y = 0
        self.canvas_height = self.canvas.winfo_height()
        self.canvas_width = self.canvas.winfo_width()
        self.hit_bottom = False  # 바닥에 닿으면 게임 끝나는 코드를 구현하기 위해서 쓰는 변수
        self.canvas.bind_all('<Key>', self.key_event)

    def draw(self):
        self.canvas.move(self.id, self.x, self.y)

        pos = self.canvas.coords(self.id)
        self.gameover(pos)

        if pos[1] <= 0:
            self.y = 3
        if pos[3] >= self.canvas_height:
            self.y = -3
            self.hit_bottom = True
        if pos[0] <= 0:
            self.x = 3
        if pos[2] >= self.canvas_width:
            self.x = -3
        if self.hit_paddle(pos) == True:
            self.x = self.paddle.x
            self.y = self.paddle.y
            self.ishit = True
            print('hit!!')

        print(self.winnerval())

    def gameover(self, pos):
        if (pos[3] >= 400) and (self.ismiss == False):
            print('miss')
            self.ismiss = True
        elif pos[3] < 400:
            self.ismiss = False

    def key_event(self, evt):
        if self.isstart == False:
            starts = [-3, -2, -1, 1, 2, 3]  # x축 방향 설정
            random.shuffle(starts)

            self.x = starts[0]
            self.y = -3
            self.isstart = True
        else:
            x_starts = [-3, -2, -1, 1, 2, 3]
            y_starts = [-1, -2, -3]
            random.shuffle(x_starts)
            random.shuffle(y_starts)
            self.x = x_starts[0]
            self.y = y_starts[0]
            self.ishit = False

    def hit_paddle(self, pos):
        paddle_pos = self.canvas.coords(self.paddle.id)
        if pos[2] >= paddle_pos[0] and pos[0] <= paddle_pos[2]:
            if pos[3] >= paddle_pos[1] and pos[1] <= paddle_pos[3]:
                return True
        return False

    def winnerval(self):
        if self.ishit == True:
            return 1
        if self.ismiss == True:
            return -1
        return 0


class Paddle:
    def __init__(self, canvas, color):
        self.canvas = canvas
        self.id = canvas.create_rectangle(0, 0, 100, 10, fill=color)
        self.canvas.move(self.id, 200, 400)
        self.x = 0
        self.y = 0
        self.canvas_width = self.canvas.winfo_width()
        self.canvas.bind_all('<KeyPress-Left>', self.key_handler)
        self.canvas.bind_all('<KeyPress-Right>', self.key_handler)
        self.canvas.bind_all('<KeyPress-Up>', self.key_handler)
        self.canvas.bind_all('<KeyPress-Down>', self.key_handler)

    def draw(self):

        pos = self.canvas.coords(self.id)

        if pos[0] <= 0 and self.x < 0:
            self.x = 0
        elif pos[2] >= self.canvas_width and self.x > 0:
            self.x = 0

        self.canvas.move(self.id, self.x, self.y)

    def key_handler(self, event):
        if event.keysym == 'Left':
            self.x = -9
        elif event.keysym == 'Right':
            self.x = 9
        elif event.keysym == 'Up':
            self.y = -5
        elif event.keysym == 'Down':
            self.y = 5


tk = Tk()
tk.title("Game")
tk.resizable(0, 0)
tk.wm_attributes("-topmost", 1)
canvas = Canvas(tk, width=600, height=500, bd=0, highlightthickness=0)
canvas.pack()
tk.update()
paddle = Paddle(canvas, 'white')
ball = Ball(canvas, paddle, 'white')

while 1:
    ball.draw()
    paddle.draw()
    tk.update_idletasks()
    tk.update()
    time.sleep(0.02)


print('')
print('====================================================================================================')
print('== 문제 246. 패들의 x 좌표, 공의 서쪽 좌표, 공의 남쪽 좌표를 아래와 같이 실시간 출력되게 하시오.')
print('====================================================================================================')
from tkinter import *
import random
import time


class Ball:
    def __init__(self, canvas, paddle, color):
        self.canvas = canvas
        self.paddle = paddle
        self.id = canvas.create_oval(10, 10, 25, 25, fill=color)
        canvas.configure(background='black')
        self.canvas.move(self.id, 245, 100)  # 공 시작 위치 설정
        self.isstart = False
        self.ismiss = False
        self.ishit = False
        self.x = 0
        self.y = 0
        self.canvas_height = self.canvas.winfo_height()
        self.canvas_width = self.canvas.winfo_width()
        self.hit_bottom = False  # 바닥에 닿으면 게임 끝나는 코드를 구현하기 위해서 쓰는 변수
        self.canvas.bind_all('<Key>', self.key_event)

    def draw(self):
        self.canvas.move(self.id, self.x, self.y)

        pos = self.canvas.coords(self.id)
        paddle_pos = self.canvas.coords(self.paddle.id)
        print('공의 위치 : (' + str(pos[0]) + ', ' + str(pos[3]) + '), 패들의 x 좌표 : ' + str(paddle_pos[0]))
        self.gameover(pos)

        if pos[1] <= 0:
            self.y = 3
        if pos[3] >= self.canvas_height:
            self.y = -3
            self.hit_bottom = True
        if pos[0] <= 0:
            self.x = 3
        if pos[2] >= self.canvas_width:
            self.x = -3
        if self.hit_paddle(pos) == True:
            self.x = self.paddle.x
            self.y = self.paddle.y
            self.ishit = True
            print('hit!!')

        print(self.winnerval())

    def gameover(self, pos):
        if (pos[3] >= 400) and (self.ismiss == False):
            print('miss')
            self.ismiss = True
        elif pos[3] < 400:
            self.ismiss = False

    def key_event(self, evt):
        if self.isstart == False:
            starts = [-3, -2, -1, 1, 2, 3]  # x축 방향 설정
            random.shuffle(starts)

            self.x = starts[0]
            self.y = -3
            self.isstart = True
        else:
            x_starts = [-3, -2, -1, 1, 2, 3]
            y_starts = [-1, -2, -3]
            random.shuffle(x_starts)
            random.shuffle(y_starts)
            self.x = x_starts[0]
            self.y = y_starts[0]
            self.ishit = False

    def hit_paddle(self, pos):
        paddle_pos = self.canvas.coords(self.paddle.id)
        if pos[2] >= paddle_pos[0] and pos[0] <= paddle_pos[2]:
            if pos[3] >= paddle_pos[1] and pos[1] <= paddle_pos[3]:
                return True
        return False

    def winnerval(self):
        if self.ishit == True:
            return 1
        if self.ismiss == True:
            return -1
        return 0


class Paddle:
    def __init__(self, canvas, color):
        self.canvas = canvas
        self.id = canvas.create_rectangle(0, 0, 100, 10, fill=color)
        self.canvas.move(self.id, 200, 400)
        self.x = 0
        self.y = 0
        self.canvas_width = self.canvas.winfo_width()
        self.canvas.bind_all('<KeyPress-Left>', self.key_handler)
        self.canvas.bind_all('<KeyPress-Right>', self.key_handler)
        self.canvas.bind_all('<KeyPress-Up>', self.key_handler)
        self.canvas.bind_all('<KeyPress-Down>', self.key_handler)

    def draw(self):

        pos = self.canvas.coords(self.id)
        if pos[0] <= 0 and self.x < 0:
            self.x = 0
        elif pos[2] >= self.canvas_width and self.x > 0:
            self.x = 0

        self.canvas.move(self.id, self.x, self.y)

    def key_handler(self, event):
        if event.keysym == 'Left':
            self.x = -9
        elif event.keysym == 'Right':
            self.x = 9
        elif event.keysym == 'Up':
            self.y = -5
        elif event.keysym == 'Down':
            self.y = 5


tk = Tk()
tk.title("Game")
tk.resizable(0, 0)
tk.wm_attributes("-topmost", 1)
canvas = Canvas(tk, width=600, height=500, bd=0, highlightthickness=0)
canvas.pack()
tk.update()
paddle = Paddle(canvas, 'white')
ball = Ball(canvas, paddle, 'white')

while 1:
    ball.draw()
    paddle.draw()
    tk.update_idletasks()
    tk.update()
    time.sleep(0.02)


print('')
print('====================================================================================================')
print('== 문제 247. 아래의 학습 데이터가 전부 출력되게 하시오.')
print('====================================================================================================')
from tkinter import *
import random
import time


class Ball:
    def __init__(self, canvas, paddle, color):
        self.canvas = canvas
        self.paddle = paddle
        self.id = canvas.create_oval(10, 10, 25, 25, fill=color)
        canvas.configure(background='black')
        self.canvas.move(self.id, 245, 100)  # 공 시작 위치 설정
        self.isstart = False
        self.ismiss = False
        self.ishit = False
        self.x = 0
        self.y = 0
        self.canvas_height = self.canvas.winfo_height()
        self.canvas_width = self.canvas.winfo_width()
        self.hit_bottom = False  # 바닥에 닿으면 게임 끝나는 코드를 구현하기 위해서 쓰는 변수
        self.canvas.bind_all('<Key>', self.key_event)
        self.pos = self.canvas.coords(self.id)
        self.paddle_pos = self.canvas.coords(self.paddle.id)

    def keystate(self):
        return self.paddle_pos[0], (self.pos[0], self.pos[3]), (self.x, self.y), self.paddle.x

    def draw(self):
        self.canvas.move(self.id, self.x, self.y)

        print(self.keystate())
        self.gameover(self.pos)

        if self.pos[1] <= 0:
            self.y = 3
        if self.pos[3] >= self.canvas_height:
            self.y = -3
            self.hit_bottom = True
        if self.pos[0] <= 0:
            self.x = 3
        if self.pos[2] >= self.canvas_width:
            self.x = -3
        if self.hit_paddle(self.pos) == True:
            self.x = self.paddle.x
            self.y = self.paddle.y
            self.ishit = True
            print('hit!!')

        print(self.winnerval())

    def gameover(self, pos):
        if (pos[3] >= 400) and (self.ismiss == False):
            print('miss')
            self.ismiss = True
        elif pos[3] < 400:
            self.ismiss = False

    def key_event(self, evt):
        if self.isstart == False:
            starts = [-3, -2, -1, 1, 2, 3]  # x축 방향 설정
            random.shuffle(starts)

            self.x = starts[0]
            self.y = -3
            self.isstart = True
        else:
            x_starts = [-3, -2, -1, 1, 2, 3]
            y_starts = [-1, -2, -3]
            random.shuffle(x_starts)
            random.shuffle(y_starts)
            self.x = x_starts[0]
            self.y = y_starts[0]
            self.ishit = False

    def hit_paddle(self, pos):
        paddle_pos = self.canvas.coords(self.paddle.id)
        if pos[2] >= paddle_pos[0] and pos[0] <= paddle_pos[2]:
            if pos[3] >= paddle_pos[1] and pos[1] <= paddle_pos[3]:
                return True
        return False

    def winnerval(self):
        if self.ishit == True:
            return 1
        if self.ismiss == True:
            return -1
        return 0


class Paddle:
    def __init__(self, canvas, color):
        self.canvas = canvas
        self.id = canvas.create_rectangle(0, 0, 100, 10, fill=color)
        self.canvas.move(self.id, 200, 400)
        self.x = 0
        self.y = 0
        self.canvas_width = self.canvas.winfo_width()
        self.canvas.bind_all('<KeyPress-Left>', self.key_handler)
        self.canvas.bind_all('<KeyPress-Right>', self.key_handler)
        self.canvas.bind_all('<KeyPress-Up>', self.key_handler)
        self.canvas.bind_all('<KeyPress-Down>', self.key_handler)

    def draw(self):

        pos = self.canvas.coords(self.id)
        if pos[0] <= 0 and self.x < 0:
            self.x = 0
        elif pos[2] >= self.canvas_width and self.x > 0:
            self.x = 0

        self.canvas.move(self.id, self.x, self.y)

    def key_handler(self, event):
        if event.keysym == 'Left':
            self.x = -5
        elif event.keysym == 'Right':
            self.x = 5
        elif event.keysym == 'Up':
            self.y = -5
        elif event.keysym == 'Down':
            self.y = 5


tk = Tk()
tk.title("Game")
tk.resizable(0, 0)
tk.wm_attributes("-topmost", 1)
canvas = Canvas(tk, width=600, height=500, bd=0, highlightthickness=0)
canvas.pack()
tk.update()
paddle = Paddle(canvas, 'white')
ball = Ball(canvas, paddle, 'white')

while 1:
    ball.draw()
    paddle.draw()
    tk.update_idletasks()
    tk.update()
    time.sleep(0.02)


print('')
print('====================================================================================================')
print('== 문제 247. 아래와 같이 keystate 함수를 수행하면 패들의 방향도 같이 출력될수 있게 하시오!')
print('====================================================================================================')
from tkinter import *
import random
import time


class Ball:
    def __init__(self, canvas, paddle, color):
        self.canvas = canvas
        self.paddle = paddle
        self.id = canvas.create_oval(10, 10, 25, 25, fill=color)
        canvas.configure(background='black')
        self.canvas.move(self.id, 245, 100)  # 공 시작 위치 설정
        self.isstart = False
        self.ismiss = False
        self.ishit = False
        self.x = 0
        self.y = 0
        self.canvas_height = self.canvas.winfo_height()
        self.canvas_width = self.canvas.winfo_width()
        self.hit_bottom = False  # 바닥에 닿으면 게임 끝나는 코드를 구현하기 위해서 쓰는 변수
        self.canvas.bind_all('<Key>', self.key_event)

    def keystate(self):
        pos = self.canvas.coords(self.id)
        paddle_pos = self.canvas.coords(self.paddle.id)
        return paddle_pos[0], (pos[0], pos[3]), (self.x, self.y), self.paddle.x

    def draw(self):
        self.canvas.move(self.id, self.x, self.y)

        pos = self.canvas.coords(self.id)
        print(self.keystate())
        self.gameover(pos)

        if pos[1] <= 0:
            self.y = 3
        if pos[3] >= self.canvas_height:
            self.y = -3
            self.hit_bottom = True
        if pos[0] <= 0:
            self.x = 3
        if pos[2] >= self.canvas_width:
            self.x = -3
        if self.hit_paddle(pos) == True:
            self.x = self.paddle.x
            self.y = self.paddle.y
            self.ishit = True
            print('hit!!')

        print(self.winnerval())

    def gameover(self, pos):
        if (pos[3] >= 400) and (self.ismiss == False):
            print('miss')
            self.ismiss = True
        elif pos[3] < 400:
            self.ismiss = False

    def key_event(self, evt):
        if self.isstart == False:
            starts = [-3, -2, -1, 1, 2, 3]  # x축 방향 설정
            random.shuffle(starts)

            self.x = starts[0]
            self.y = -3
            self.isstart = True
        else:
            x_starts = [-3, -2, -1, 1, 2, 3]
            y_starts = [-1, -2, -3]
            random.shuffle(x_starts)
            random.shuffle(y_starts)
            self.x = x_starts[0]
            self.y = y_starts[0]
            self.ishit = False

    def hit_paddle(self, pos):
        paddle_pos = self.canvas.coords(self.paddle.id)
        if pos[2] >= paddle_pos[0] and pos[0] <= paddle_pos[2]:
            if pos[3] >= paddle_pos[1] and pos[1] <= paddle_pos[3]:
                return True
        return False

    def winnerval(self):
        if self.ishit == True:
            return 1
        if self.ismiss == True:
            return -1
        return 0


class Paddle:
    def __init__(self, canvas, color):
        self.canvas = canvas
        self.id = canvas.create_rectangle(0, 0, 100, 10, fill=color)
        self.canvas.move(self.id, 200, 400)
        self.x = 0
        self.y = 0
        self.canvas_width = self.canvas.winfo_width()
        self.canvas.bind_all('<KeyPress-Left>', self.key_handler)
        self.canvas.bind_all('<KeyPress-Right>', self.key_handler)
        self.canvas.bind_all('<KeyPress-Up>', self.key_handler)
        self.canvas.bind_all('<KeyPress-Down>', self.key_handler)

    def draw(self):

        pos = self.canvas.coords(self.id)
        if pos[0] <= 0 and self.x < 0:
            self.x = 0
        elif pos[2] >= self.canvas_width and self.x > 0:
            self.x = 0

        self.canvas.move(self.id, self.x, self.y)

    def key_handler(self, event):
        if event.keysym == 'Left':
            self.x = -5
        elif event.keysym == 'Right':
            self.x = 5
        elif event.keysym == 'Up':
            self.y = -5
        elif event.keysym == 'Down':
            self.y = 5


tk = Tk()
tk.title("Game")
tk.resizable(0, 0)
tk.wm_attributes("-topmost", 1)
canvas = Canvas(tk, width=600, height=500, bd=0, highlightthickness=0)
canvas.pack()
tk.update()
paddle = Paddle(canvas, 'white')
ball = Ball(canvas, paddle, 'white')

while 1:
    ball.draw()
    paddle.draw()
    tk.update_idletasks()
    tk.update()
    time.sleep(0.02)


print('')
print('====================================================================================================')
print('== 문제 250. add 함수를 추가하시오.')
print('====================================================================================================')
from tkinter import *
import random
import time


class Ball:
    def __init__(self, canvas, paddle, color):
        self.canvas = canvas
        self.paddle = paddle
        self.id = canvas.create_oval(10, 10, 25, 25, fill=color)
        canvas.configure(background='black')
        self.canvas.move(self.id, 245, 100)  # 공 시작 위치 설정
        self.isstart = False
        self.ismiss = False
        self.ishit = False
        self.x = 0
        self.y = 0
        self.canvas_height = self.canvas.winfo_height()
        self.canvas_width = self.canvas.winfo_width()
        self.hit_bottom = False  # 바닥에 닿으면 게임 끝나는 코드를 구현하기 위해서 쓰는 변수
        self.canvas.bind_all('<Key>', self.key_event)
        self.values = {}

    def keystate(self):
        pos = self.canvas.coords(self.id)
        paddle_pos = self.canvas.coords(self.paddle.id)
        return paddle_pos[0], (pos[0], pos[3]), (self.x, self.y), self.paddle.x

    def add(self):
        self.values[self.keystate()] = 0

    def draw(self):
        self.canvas.move(self.id, self.x, self.y)

        pos = self.canvas.coords(self.id)
        self.add()
        print(self.values)
        self.gameover(pos)

        if pos[1] <= 0:
            self.y = 3
        if pos[3] >= self.canvas_height:
            self.y = -3
            self.hit_bottom = True
        if pos[0] <= 0:
            self.x = 3
        if pos[2] >= self.canvas_width:
            self.x = -3
        if self.hit_paddle(pos) == True:
            self.x = self.paddle.x
            self.y = self.paddle.y
            self.ishit = True
            print('hit!!')

        print(self.winnerval())

    def gameover(self, pos):
        if (pos[3] >= 400) and (self.ismiss == False):
            print('miss')
            self.ismiss = True
        elif pos[3] < 400:
            self.ismiss = False

    def key_event(self, evt):
        if self.isstart == False:
            starts = [-3, -2, -1, 1, 2, 3]  # x축 방향 설정
            random.shuffle(starts)

            self.x = starts[0]
            self.y = -3
            self.isstart = True
        else:
            x_starts = [-3, -2, -1, 1, 2, 3]
            y_starts = [-1, -2, -3]
            random.shuffle(x_starts)
            random.shuffle(y_starts)
            self.x = x_starts[0]
            self.y = y_starts[0]
            self.ishit = False

    def hit_paddle(self, pos):
        paddle_pos = self.canvas.coords(self.paddle.id)
        if pos[2] >= paddle_pos[0] and pos[0] <= paddle_pos[2]:
            if pos[3] >= paddle_pos[1] and pos[1] <= paddle_pos[3]:
                return True
        return False

    def winnerval(self):
        if self.ishit == True:
            return 1
        if self.ismiss == True:
            return -1
        return 0


class Paddle:
    def __init__(self, canvas, color):
        self.canvas = canvas
        self.id = canvas.create_rectangle(0, 0, 100, 10, fill=color)
        self.canvas.move(self.id, 200, 400)
        self.x = 0
        self.y = 0
        self.canvas_width = self.canvas.winfo_width()
        self.canvas.bind_all('<KeyPress-Left>', self.key_handler)
        self.canvas.bind_all('<KeyPress-Right>', self.key_handler)
        self.canvas.bind_all('<KeyPress-Up>', self.key_handler)
        self.canvas.bind_all('<KeyPress-Down>', self.key_handler)

    def draw(self):

        pos = self.canvas.coords(self.id)
        if pos[0] <= 0 and self.x < 0:
            self.x = 0
        elif pos[2] >= self.canvas_width and self.x > 0:
            self.x = 0

        self.canvas.move(self.id, self.x, self.y)

    def key_handler(self, event):
        if event.keysym == 'Left':
            self.x = -5
        elif event.keysym == 'Right':
            self.x = 5
        elif event.keysym == 'Up':
            self.y = -5
        elif event.keysym == 'Down':
            self.y = 5


tk = Tk()
tk.title("Game")
tk.resizable(0, 0)
tk.wm_attributes("-topmost", 1)
canvas = Canvas(tk, width=600, height=500, bd=0, highlightthickness=0)
canvas.pack()
tk.update()
paddle = Paddle(canvas, 'white')
ball = Ball(canvas, paddle, 'white')

while 1:
    ball.draw()
    paddle.draw()
    tk.update_idletasks()
    tk.update()
    time.sleep(0.02)


print('')
print('====================================================================================================')
print('== 문제 251. lookup 함수를 추가하시오.')
print('====================================================================================================')
from tkinter import *
import random
import time


class Ball:
    def __init__(self, canvas, paddle, color):
        self.canvas = canvas
        self.paddle = paddle
        self.id = canvas.create_oval(10, 10, 25, 25, fill=color)
        canvas.configure(background='black')
        self.canvas.move(self.id, 245, 100)  # 공 시작 위치 설정
        self.isstart = False
        self.ismiss = False
        self.ishit = False
        self.x = 0
        self.y = 0
        self.canvas_height = self.canvas.winfo_height()
        self.canvas_width = self.canvas.winfo_width()
        self.hit_bottom = False  # 바닥에 닿으면 게임 끝나는 코드를 구현하기 위해서 쓰는 변수
        self.canvas.bind_all('<Key>', self.key_event)
        self.values = {}

    def keystate(self, pos, paddle_pos):
        return paddle_pos[0], (pos[0], pos[3]), (self.x, self.y), self.paddle.x

    def add(self, key):
        self.values[key] = 0

    def lookup(self, key):
        if key not in self.values:
            self.add(key)
        return self.values[key]

    def draw(self):
        self.canvas.move(self.id, self.x, self.y)

        pos = self.canvas.coords(self.id)
        paddle_pos = self.canvas.coords(self.paddle.id)
        key = self.keystate(pos, paddle_pos)
        print(key, self.lookup(key))

        self.gameover(pos)

        if pos[1] <= 0:
            self.y = 3
        if pos[3] >= self.canvas_height:
            self.y = -3
            self.hit_bottom = True
        if pos[0] <= 0:
            self.x = 3
        if pos[2] >= self.canvas_width:
            self.x = -3
        if self.hit_paddle(pos) == True:
            self.x = self.paddle.x
            self.y = self.paddle.y
            self.ishit = True
            print('hit!!')

        print(self.winnerval())

    def gameover(self, pos):
        if (pos[3] >= 400) and (self.ismiss == False):
            print('miss')
            self.ismiss = True
        elif pos[3] < 400:
            self.ismiss = False

    def key_event(self, evt):
        if self.isstart == False:
            starts = [-3, -2, -1, 1, 2, 3]  # x축 방향 설정
            random.shuffle(starts)

            self.x = starts[0]
            self.y = -3
            self.isstart = True
        else:
            x_starts = [-3, 3]
            random.shuffle(x_starts)
            self.x = x_starts[0]
            self.y = -3
            self.ishit = False

    def hit_paddle(self, pos):
        paddle_pos = self.canvas.coords(self.paddle.id)
        if pos[2] >= paddle_pos[0] and pos[0] <= paddle_pos[2]:
            if pos[3] >= paddle_pos[1] and pos[1] <= paddle_pos[3]:
                return True
        return False

    def winnerval(self):
        if self.ishit == True:
            return 1
        if self.ismiss == True:
            return -1
        return 0


class Paddle:
    def __init__(self, canvas, color):
        self.canvas = canvas
        self.id = canvas.create_rectangle(0, 0, 100, 10, fill=color)
        self.canvas.move(self.id, 200, 400)
        self.x = 0
        self.y = 0
        self.canvas_width = self.canvas.winfo_width()
        self.canvas.bind_all('<KeyPress-Left>', self.key_handler)
        self.canvas.bind_all('<KeyPress-Right>', self.key_handler)
        self.canvas.bind_all('<KeyPress-Up>', self.key_handler)
        self.canvas.bind_all('<KeyPress-Down>', self.key_handler)

    def draw(self):

        pos = self.canvas.coords(self.id)
        if pos[0] <= 0 and self.x < 0:
            self.x = 0
        elif pos[2] >= self.canvas_width and self.x > 0:
            self.x = 0

        self.canvas.move(self.id, self.x, self.y)

    def key_handler(self, event):
        if event.keysym == 'Left':
            self.x = -5
        elif event.keysym == 'Right':
            self.x = 5
        elif event.keysym == 'Up':
            self.y = -5
        elif event.keysym == 'Down':
            self.y = 5


tk = Tk()
tk.title("Game")
tk.resizable(0, 0)
tk.wm_attributes("-topmost", 1)
canvas = Canvas(tk, width=600, height=500, bd=0, highlightthickness=0)
canvas.pack()
tk.update()
paddle = Paddle(canvas, 'white')
ball = Ball(canvas, paddle, 'white')

while 1:
    ball.draw()
    paddle.draw()
    tk.update_idletasks()
    tk.update()
    time.sleep(0.02)


print('')
print('====================================================================================================')
print('== 문제 252. 위에서 설명한 randomChoice 함수를 생성하시오!')
print('====================================================================================================')
from tkinter import *
import random
import time
import pygame

class Ball:
    def __init__(self, canvas, paddle, color):
        self.canvas = canvas
        self.paddle = paddle
        self.id = canvas.create_oval(10, 10, 25, 25, fill=color)
        canvas.configure(background='black')
        self.canvas.move(self.id, 245, 100)  # 공 시작 위치 설정
        self.isstart = False
        self.ismiss = False
        self.ishit = False
        self.x = 0
        self.y = 0
        self.canvas_height = self.canvas.winfo_height()
        self.canvas_width = self.canvas.winfo_width()
        self.hit_bottom = False  # 바닥에 닿으면 게임 끝나는 코드를 구현하기 위해서 쓰는 변수
        self.canvas.bind_all('<Key>', self.key_event)
        self.values = {}
        self.sound_setting()

    def sound_setting(self):
        pygame.init()
        self.hit_sound = pygame.mixer.Sound("hit.wav")

    def draw(self):
        self.canvas.move(self.id, self.x, self.y)

        pos = self.canvas.coords(self.id)
        paddle_pos = self.canvas.coords(self.paddle.id)
        key = self.keystate(pos, paddle_pos)
        print(key, self.lookup(key))

        self.gameover(pos)

        if pos[1] <= 0:
            self.y = 3
        if pos[3] >= self.canvas_height:
            self.y = -3
            self.hit_bottom = True
        if pos[0] <= 0:
            self.x = 3
        if pos[2] >= self.canvas_width:
            self.x = -3
        if self.hit_paddle(pos) == True:
            self.hit_sound.play()
            self.y = -3
            print('hit!!')

        print(self.winnerval())

    def keystate(self, pos, paddle_pos):
        return paddle_pos[0], (pos[0], pos[3]), (self.x, self.y), self.paddle.x

    def add(self, key):
        self.values[key] = 0

    def lookup(self, key):
        if key not in self.values:
            self.add(key)
        return self.values[key]

    def randomChoice(self):
        rand = random.choice([0, 1])
        key = self.keystate(rand)
        if key not in self.values:
            self.add(key)
        return rand

    def gameover(self, pos):
        if (pos[3] >= 400) and (self.ismiss == False):
            print('miss')
            self.ismiss = True
        elif pos[3] < 400:
            self.ismiss = False

    def key_event(self, evt):
        if self.isstart == False:
            starts = [-3, -2, -1, 1, 2, 3]  # x축 방향 설정
            random.shuffle(starts)

            self.x = starts[0]
            self.y = -3
            self.isstart = True
        else:
            x_starts = [-3, 3]
            random.shuffle(x_starts)
            self.x = x_starts[0]
            self.y = -3
            self.ishit = False

    def hit_paddle(self, pos):
        paddle_pos = self.canvas.coords(self.paddle.id)
        if pos[2] >= paddle_pos[0] and pos[0] <= paddle_pos[2]:
            if pos[3] >= paddle_pos[1] and pos[1] <= paddle_pos[3]:
                return True
        return False

    def winnerval(self):
        if self.ishit == True:
            return 1
        if self.ismiss == True:
            return -1
        return 0


class Paddle:
    def __init__(self, canvas, color):
        self.canvas = canvas
        self.id = canvas.create_rectangle(0, 0, 100, 10, fill=color)
        self.canvas.move(self.id, 200, 400)
        self.x = 0
        self.y = 0
        self.canvas_width = self.canvas.winfo_width()
        self.canvas.bind_all('<KeyPress-Left>', self.key_handler)
        self.canvas.bind_all('<KeyPress-Right>', self.key_handler)
        self.canvas.bind_all('<KeyPress-Up>', self.key_handler)
        self.canvas.bind_all('<KeyPress-Down>', self.key_handler)

    def draw(self):

        pos = self.canvas.coords(self.id)
        if pos[0] <= 0 and self.x < 0:
            self.x = 0
        elif pos[2] >= self.canvas_width and self.x > 0:
            self.x = 0

        self.canvas.move(self.id, self.x, self.y)

    def key_handler(self, event):
        if event.keysym == 'Left':
            self.x = -5
        elif event.keysym == 'Right':
            self.x = 5
        elif event.keysym == 'Up':
            self.y = -5
        elif event.keysym == 'Down':
            self.y = 5


tk = Tk()
tk.title("Game")
tk.resizable(0, 0)
tk.wm_attributes("-topmost", 1)
canvas = Canvas(tk, width=600, height=500, bd=0, highlightthickness=0)
canvas.pack()
tk.update()
paddle = Paddle(canvas, 'white')
ball = Ball(canvas, paddle, 'white')

while 1:
    ball.draw()
    paddle.draw()
    tk.update_idletasks()
    tk.update()
    time.sleep(0.02)


print('')
print('====================================================================================================')
print('== 문제 253. 학습되는 동안 공의 기울기는 어떻게 되는가?')
print('====================================================================================================')
from tkinter import *

import random
import time
import csv

EMPTY = 0
PADDLE_HEIGHT = 360.0
PADDLE_MOVE = [-10, 10]

START = 3
END = 4

class Ball:
    def __init__(self, canvas, paddle, color, announceterm, saveterm, winval=10, loseval=-1):
        self.canvas = canvas
        self.paddle = paddle
        self.id = canvas.create_oval(10, 10, 25, 25, fill=color)  # 공 크기 및 색깔
        self.canvas.move(self.id, 245, 200)  # 공을 캔버스 중앙으로 이동
        starts = [-3, 3]  # 공의 속도를 랜덤으로 구성하기 위해 준비한 리스트
        random.shuffle(starts)  # starts 리스트 중에 숫자를 랜덤으로 골라서
        self.x = starts[0]  # 처음 공이 패들에서 움직일때 왼쪽으로 올라갈지 오른쪽으로 올라갈지 랜덤으로 결정되는 부분
        self.y = -3  # 처음 공이 패들에서 움직일때 위로 올라가는 속도
        self.canvas_height = self.canvas.winfo_height()  # 캔버스의 현재 높이를 반환한다.(공이 화면에서 사라지지 않기위해)
        self.canvas_width = self.canvas.winfo_width()  # 캔버스의 현재 넓이를 반환한다.(공이 화면에서 사라지지 않기위해)
        self.hit_bottom = False
        self.values = {}
        self.epsilon = 0.1  # 랜덤율
        self.alpha = 0.99  # 망각계수
        self.learning = True
        self.cycle_data = []
        self.wincount = 0
        self.losecount = 0
        self.gamecount = 0
        self.winval = winval  # 성공 보상치
        self.loseval = loseval
        self.announceterm = announceterm  # 게임 횟수 프린트 텀
        self.saveterm = saveterm  # csv 저장 텀

        # csv 파일에서 gamecount / values 불러옴
        self.loadcsv()


    def action(self):
        r = random.random()
        if r < self.epsilon:
            direction = self.randomChoice()
        else:
            direction = self.greedyChoice()
        x = PADDLE_MOVE[direction]
        # 머신러닝을 위한 한 사이클 내 이동 데이터 저장
        self.cycle_data.append(self.keystate(direction))
        # 이동/그리기
        self.paddle.move(x)
        self.paddle.draw()


    # 이동 방향 랜덤으로 지정
    def randomChoice(self):
        rand = random.choice([0, 1])
        key = self.keystate(rand)
        if key not in self.values:
            self.add(key)
        return rand


    # 이동 방향 Greedy로 지정
    def greedyChoice(self):
        val_left = self.keystate(0)
        val_right = self.keystate(1)

        if self.lookup(val_left) > self.lookup(val_right):
            return 0
        elif self.lookup(val_left) < self.lookup(val_right):
            return 1
        else:
            return random.choice([0, 1])


    def add(self, key):
        self.values[key] = 0


    def lookup(self, key):
        if key not in self.values:
            # print(key)
            self.add(key)
        return self.values[key]


    def hit_paddle(self, pos):  # 패들에 공이 튀기게 하는 함수
        paddle_pos = self.canvas.coords(self.paddle.id)
        # 공의 x좌표가 패들의 너비 안에 있는지 / 공 바닥이 패들 윗면에 닿아 있는지
        if pos[2] >= paddle_pos[0] and pos[0] <= paddle_pos[2] \
                and pos[3] == PADDLE_HEIGHT:
            return True
        return False


    # 공을 패들로 받았는지 여부 출력(True/False)
    def is_paddle_hit(self):
        return self.hit_paddle(self.canvas.coords(self.id))


    def draw(self):
        # 볼의 현재 좌표를 출력해준다. 공 좌표( 좌상단 x,y좌표 / 우하단 x,y좌표 )
        pos = self.canvas.coords(self.id)
        # [ 255,29,270,44]
        print(self.y/self.x)
        if pos[1] <= 0:
            self.y = 3
        if pos[3] >= self.canvas_height:
            self.y = -3
        if pos[0] <= 0:
            self.x = 3
        if pos[2] >= self.canvas_width:
            self.x = -3
        if self.hit_paddle(pos) == True:
            self.y = -3

        self.canvas.move(self.id, self.x, self.y)  # 공을 움직이게 하는 부분
        # 공이 화면 밖으로 나가지 않게 해준다


    def keystate(self, movement):
        paddle_pos = self.canvas.coords(self.paddle.id)
        ball_pos = self.canvas.coords(self.id)
        # paddle 위치(좌측 x좌표), 공의 좌상단 x/y좌표, 공의 좌우/상하 속도(방향), paddle을 좌/우 중 어느 쪽으로 움직이는지
        return (paddle_pos[0], (ball_pos[0], ball_pos[1]), (self.x, self.y), movement)


    # 사이클 시작 : 1, 사이클 종료 : -1, 해당 없음 : 0
    def cyclestate(self):
        pos = self.canvas.coords(self.id)
        if pos[3] == PADDLE_HEIGHT:
            if self.y == -3:
                return START
            elif self.y == 3:
                return END
        return 0


    # 결과 학습
    def backup(self, newVal, idx):
        if idx >= 0 and self.learning:
            prevVal = self.values[self.cycle_data[idx]]
            self.values[self.cycle_data[idx]] += self.alpha * (newVal - prevVal)
            # print("key : {0}, val : {1}".format(self.cycle_data[idx],self.values[self.cycle_data[idx]]))
            self.backup(newVal * self.alpha, idx - 1)


    # 게임 끝났을 시 학습 및 cycle_data 초기화
    def gameover(self):
        if self.learning:
            # paddle로 받았으면 1, 못 받았으면 -1
            if self.is_paddle_hit():
                result_value = self.winval
                self.wincount += 1
            else:
                result_value = self.loseval
                self.losecount += 1
            self.backup(result_value, len(self.cycle_data) - 1)
            self.gamecount += 1

            # saveterm마다 csv 저장
            if self.gamecount % self.saveterm == 0:
                self.writecsv()
            if self.gamecount % self.announceterm == 0:
                print("cycle count : {0}".format(ball.gamecount))
        self.cycle_data.clear()


    def winnerval(self, winner):
        if winner == 'hit':
            return 1
        elif winner == 'miss':
            return -1
        else:
            return 0


    # 게임 결과 csv로 저장
    def writecsv(self):
        try:
            # Values 저장
            Fn = open("D:\\KYH\\02.PYTHON\\data\\pong_value.csv", 'w', newline='')
            writer = csv.writer(Fn, delimiter=',')
            writer.writerow([self.gamecount])  # 첫줄에 학습 게임 횟수 저장
            keys = self.values.keys()
            for key in keys:
                writer.writerow([key[0],
                                 key[1][0],
                                 key[1][1],
                                 key[2][0],
                                 key[2][1],
                                 key[3],
                                 ball.values[key]
                                 ])
            Fn.close()

            # 성공/실패 횟수 저장
            Fn = open("D:\\KYH\\02.PYTHON\\data\\pong_score.csv", 'a', newline='')
            writer = csv.writer(Fn, delimiter=',')
            writer.writerow([self.wincount, self.losecount, self.gamecount])
            Fn.close()
            # 승률의 변화를 확인하기 위해 일정 판수마다 카운트 리셋
            self.wincount = 0
            self.losecount = 0

            print("save data in cycle {0}.".format(self.gamecount))
        except Exception as e:
            print('save data failed in cycle {0}.\nError Type : {1}'.format(self.gamecount, type(e).__name__))


    def loadcsv(self):
        try:
            Fn = open("D:\\KYH\\02.PYTHON\\data\\pong_value.csv", 'r')
            self.gamecount = int(Fn.readline().split(',')[0])  # 첫 줄의 학습 게임 횟수 불러오기
            reader = csv.reader(Fn, delimiter=',')
            for key in reader:
                self.values[(
                int(float(key[0])), (int(float(key[1])), int(float(key[2]))), (int(float(key[3])), int(float(key[4]))),
                int(float(key[5])))] = float(key[6])
            print('Load Success! Start at cycle {0}'.format(self.gamecount))
        except Exception:
            print('Load Failed!')


class Paddle:
    def __init__(self, canvas, y_loc, color):
        self.canvas = canvas
        self.id = canvas.create_rectangle(0, 0, 100, 10, fill=color)  # 패들의 높이와 넓이 그리고 색깔

        self.canvas.move(self.id, 200, y_loc)  # 패들 사각형을 200,300 에 위치
        self.x = 0  # 패들이 처음 시작할때 움직이지 않게 0으로 설정
        self.canvas_width = self.canvas.winfo_width()  # 캔버스의 넓이를 반환한다. 캔버스 밖으로 패들이 나가지 않도록
        self.canvas.bind_all('<KeyPress-Left>', self.turn_left)  # 왼쪽 화살표 키를 '<KeyPress-Left>'  라는 이름로 바인딩
        self.canvas.bind_all('<KeyPress-Right>', self.turn_right)  # 오른쪽도 마찬가지로 바인딩한다.

    def draw(self):
        pos = self.canvas.coords(self.id)
        # print(pos)
        if pos[0] <= 0 and self.x < 0:  # 패들의 위치가 왼쪽 끝이고, 이동하려는 방향이 왼쪽이면 함수 종료(이동 안 함)
            return
        elif pos[2] >= self.canvas_width and self.x > 0:  # 패들의 위치가 오른쪽 끝이고, 이동하려는 방향이 오른쪽이면 함수 종료
            return
        self.canvas.move(self.id, self.x, 0)


        # 패들이 화면의 끝에 부딪히면 공처럼 튕기는게 아니라 움직임이 멈춰야한다.
        # 그래서 왼쪽 x 좌표(pos[0]) 가 0 과 같거나 작으면 self.x = 0 처럼 x 변수에 0 을
        # 설정한다.  같은 방법으로 오른쪽 x 좌표(pos[2]) 가 캔버스의 폭과 같거나 크면
        # self.x = 0 처럼 변수에 0 을 설정한다.

    def turn_left(self, evt):  # 패들의 방향을 전환하는 함수
        self.x = -3

    def turn_right(self, evt):
        self.x = 3

    def move(self, x):
        self.x = x


'''
LYE
1. cyclestart 함수 추가(공이 딱 패들의 높이를 지나 위로 출발하는 시점에 True)
2. 캔버스 높이 및 공/패들 시작점 조정(y 좌표가 3의 배수로 떨어지게)
'''

if __name__ == '__main__':
    tk = Tk()  # tk 를 인스턴스화 한다.
    tk.title("Game")  # tk 객체의 title 메소드(함수)로 게임창에 제목을 부여한다.
    tk.resizable(0, 0)  # 게임창의 크기는 가로나 세로로 변경될수 없다라고 말하는것이다.
    tk.wm_attributes("-topmost", 1)  # 다른 모든 창들 앞에 캔버스를 가진 창이 위치할것을 tkinter 에게 알려준다.

    canvas = Canvas(tk, width=500, height=450, bd=0, highlightthickness=0)
    # bg=0,highlightthickness=0 은 캔버스 외곽에 둘러싼
    # 외곽선이 없도록 하는것이다. (게임화면이 좀더 좋게)
    canvas.pack()  # 앞의 코드에서 전달된 폭과 높이는 매개변수에 따라 크기를 맞추라고 캔버스에에 말해준다.
    tk.update()  # tkinter 에게 게임에서의 애니메이션을 위해 자신을 초기화하라고 알려주는것이다.
    paddle = Paddle(canvas, PADDLE_HEIGHT, 'blue')

    # announceterm : 현재 count 출력 term, saveterm : csv에 저장하는 term
    ball = Ball(canvas, paddle, 'red', announceterm=500, saveterm=10000)
    start = False
    # 공을 약간 움직이고 새로운 위치로 화면을 다시 그리며, 잠깐 잠들었다가 다시 시작해 ! "
    is_cycling = False

    while 1:
        ball.draw()

        c_state = ball.cyclestate()
        if c_state == END:
            # print('END')
            ball.gameover()
            is_cycling = False
        if c_state == START or ball.is_paddle_hit():
            # print('START')
            is_cycling = True

        if is_cycling:
            ball.action()

        tk.update_idletasks()  # 우리가 창을 닫으라고 할때까지 계속해서 tkinter 에게 화면을 그려라 !
        tk.update()  # tkinter 에게 게임에서의 애니메이션을 위해 자신을 초기화하라고 알려주는것이다.

        #10만번 학습 후에 정상 속도로 플레이 시작(학습 결과 반영됨)
        if ball.gamecount > 10000:
            time.sleep(0.005)  # 무한 루프중에 100분의 1초마다 잠들어라 !


print('')
print('====================================================================================================')
print('== 문제 255. 패들의 위치를 위로 올리고 게임과 학습이 되게 하시오!')
print('====================================================================================================')
from tkinter import *

import random
import time
import csv

EMPTY = 0
PADDLE_HEIGHT = 360.0
PADDLE_MOVE = [-10, 10]

START = 3
END = 4

class Ball:
    def __init__(self, canvas, paddle, color, announceterm, saveterm, winval=1, loseval=-1):
        self.canvas = canvas
        self.paddle = paddle
        self.id = canvas.create_oval(10, 10, 25, 25, fill=color)  # 공 크기 및 색깔
        self.canvas.move(self.id, 245, 200)  # 공을 캔버스 중앙으로 이동
        starts = [-3, 3]  # 공의 속도를 랜덤으로 구성하기 위해 준비한 리스트
        random.shuffle(starts)  # starts 리스트 중에 숫자를 랜덤으로 골라서
        self.x = starts[0]  # 처음 공이 패들에서 움직일때 왼쪽으로 올라갈지 오른쪽으로 올라갈지 랜덤으로 결정되는 부분
        self.y = -3  # 처음 공이 패들에서 움직일때 위로 올라가는 속도
        self.canvas_height = self.canvas.winfo_height()  # 캔버스의 현재 높이를 반환한다.(공이 화면에서 사라지지 않기위해)
        self.canvas_width = self.canvas.winfo_width()  # 캔버스의 현재 넓이를 반환한다.(공이 화면에서 사라지지 않기위해)
        self.hit_bottom = False
        self.values = {}
        self.epsilon = 0.1  # 랜덤율
        self.alpha = 0.99  # 망각계수
        self.learning = True
        self.cycle_data = []
        self.wincount = 0
        self.losecount = 0
        self.gamecount = 0
        self.winval = winval  # 성공 보상치
        self.loseval = loseval
        self.announceterm = announceterm  # 게임 횟수 프린트 텀
        self.saveterm = saveterm  # csv 저장 텀

        # csv 파일에서 gamecount / values 불러옴
        self.loadcsv()


    def action(self):
        r = random.random()
        if r < self.epsilon:
            direction = self.randomChoice()
        else:
            direction = self.greedyChoice()
        x = PADDLE_MOVE[direction]
        # 머신러닝을 위한 한 사이클 내 이동 데이터 저장
        self.cycle_data.append(self.keystate(direction))
        # 이동/그리기
        self.paddle.move(x)
        self.paddle.draw()


    # 이동 방향 랜덤으로 지정
    def randomChoice(self):
        rand = random.choice([0, 1])
        key = self.keystate(rand)
        if key not in self.values:
            self.add(key)
        return rand


    # 이동 방향 Greedy로 지정
    def greedyChoice(self):
        val_left = self.keystate(0)
        val_right = self.keystate(1)

        if self.lookup(val_left) > self.lookup(val_right):
            return 0
        elif self.lookup(val_left) < self.lookup(val_right):
            return 1
        else:
            return random.choice([0, 1])


    def add(self, key):
        self.values[key] = 0


    def lookup(self, key):
        if key not in self.values:
            # print(key)
            self.add(key)
        return self.values[key]


    def hit_paddle(self, pos):  # 패들에 공이 튀기게 하는 함수
        paddle_pos = self.canvas.coords(self.paddle.id)
        # 공의 x좌표가 패들의 너비 안에 있는지 / 공 바닥이 패들 윗면에 닿아 있는지
        if pos[2] >= paddle_pos[0] and pos[0] <= paddle_pos[2] and pos[3] == PADDLE_HEIGHT:
            return True
        return False


    # 공을 패들로 받았는지 여부 출력(True/False)
    def is_paddle_hit(self):
        return self.hit_paddle(self.canvas.coords(self.id))


    def draw(self):
        # 볼의 현재 좌표를 출력해준다. 공 좌표( 좌상단 x,y좌표 / 우하단 x,y좌표 )
        pos = self.canvas.coords(self.id)

        if pos[1] <= 0:
            self.y = 3
        if pos[3] >= self.canvas_height:
            self.y = -3
        if pos[0] <= 0:
            self.x = 3
        if pos[2] >= self.canvas_width:
            self.x = -3
        if self.hit_paddle(pos) == True:
            self.y = -3

        self.canvas.move(self.id, self.x, self.y)  # 공을 움직이게 하는 부분
        # 공이 화면 밖으로 나가지 않게 해준다


    def keystate(self, movement):
        paddle_pos = self.canvas.coords(self.paddle.id)
        ball_pos = self.canvas.coords(self.id)
        # paddle 위치(좌측 x좌표), 공의 좌상단 x/y좌표, 공의 좌우/상하 속도(방향), paddle을 좌/우 중 어느 쪽으로 움직이는지
        return (paddle_pos[0], (ball_pos[0], ball_pos[1]), (self.x, self.y), movement)


    # 사이클 시작 : 1, 사이클 종료 : -1, 해당 없음 : 0
    def cyclestate(self):
        pos = self.canvas.coords(self.id)
        if pos[3] == PADDLE_HEIGHT:
            if self.y == -3:
                return START
            elif self.y == 3:
                return END
        return 0


    # 결과 학습
    def backup(self, newVal, idx):
        if idx >= 0 and self.learning:
            prevVal = self.values[self.cycle_data[idx]]
            self.values[self.cycle_data[idx]] += self.alpha * (newVal - prevVal)
            # print("key : {0}, val : {1}".format(self.cycle_data[idx],self.values[self.cycle_data[idx]]))
            self.backup(newVal * self.alpha, idx - 1)


    # 게임 끝났을 시 학습 및 cycle_data 초기화
    def gameover(self):
        if self.learning:
            # paddle로 받았으면 1, 못 받았으면 -1
            if self.is_paddle_hit():
                result_value = self.winval
                self.wincount += 1
            else:
                result_value = self.loseval
                self.losecount += 1
            self.backup(result_value, len(self.cycle_data) - 1)
            self.gamecount += 1

            # saveterm마다 csv 저장
            if self.gamecount % self.saveterm == 0:
                self.writecsv()
            if self.gamecount % self.announceterm == 0:
                print("cycle count : {0}".format(self.gamecount))
        self.cycle_data.clear()


    def winnerval(self, winner):
        if winner == 'hit':
            return 1
        elif winner == 'miss':
            return -1
        else:
            return 0


    # 게임 결과 csv로 저장
    def writecsv(self):
        try:
            # Values 저장
            Fn = open("D:\\KYH\\02.PYTHON\\data\\pong_value.csv", 'w', newline='')
            writer = csv.writer(Fn, delimiter=',')
            writer.writerow([self.gamecount])  # 첫줄에 학습 게임 횟수 저장
            keys = self.values.keys()
            for key in keys:
                writer.writerow([key[0],
                                 key[1][0],
                                 key[1][1],
                                 key[2][0],
                                 key[2][1],
                                 key[3],
                                 ball.values[key]
                                 ])
            Fn.close()

            # 성공/실패 횟수 저장
            Fn = open("D:\pong_score.csv", 'a', newline='')
            writer = csv.writer(Fn, delimiter=',')
            writer.writerow([self.wincount, self.losecount, self.gamecount])
            Fn.close()
            # 승률의 변화를 확인하기 위해 일정 판수마다 카운트 리셋
            self.wincount = 0
            self.losecount = 0

            print("save data in cycle {0}.".format(self.gamecount))
        except Exception as e:
            print('save data failed in cycle {0}.\nError Type : {1}'.format(self.gamecount, type(e).__name__))


    def loadcsv(self):
        try:
            Fn = open("D:\\KYH\\02.PYTHON\\data\\pong_value.csv", 'r')
            self.gamecount = int(Fn.readline().split(',')[0])  # 첫 줄의 학습 게임 횟수 불러오기
            reader = csv.reader(Fn, delimiter=',')
            for key in reader:
                self.values[(
                int(float(key[0])), (int(float(key[1])), int(float(key[2]))), (int(float(key[3])), int(float(key[4]))),
                int(float(key[5])))] = float(key[6])
            print('Load Success! Start at cycle {0}'.format(self.gamecount))
        except Exception:
            print('Load Failed!')


class Paddle:
    def __init__(self, canvas, y_loc, color):
        self.canvas = canvas
        self.id = canvas.create_rectangle(0, 0, 100, 10, fill=color)  # 패들의 높이와 넓이 그리고 색깔

        self.canvas.move(self.id, 200, y_loc)  # 패들 사각형을 200,300 에 위치
        self.x = 0  # 패들이 처음 시작할때 움직이지 않게 0으로 설정
        self.canvas_width = self.canvas.winfo_width()  # 캔버스의 넓이를 반환한다. 캔버스 밖으로 패들이 나가지 않도록
        self.canvas.bind_all('<KeyPress-Left>', self.turn_left)  # 왼쪽 화살표 키를 '<KeyPress-Left>'  라는 이름로 바인딩
        self.canvas.bind_all('<KeyPress-Right>', self.turn_right)  # 오른쪽도 마찬가지로 바인딩한다.

    def draw(self):
        pos = self.canvas.coords(self.id)
        # print(pos)
        if pos[0] <= 0 and self.x < 0:  # 패들의 위치가 왼쪽 끝이고, 이동하려는 방향이 왼쪽이면 함수 종료(이동 안 함)
            return
        elif pos[2] >= self.canvas_width and self.x > 0:  # 패들의 위치가 오른쪽 끝이고, 이동하려는 방향이 오른쪽이면 함수 종료
            return
        self.canvas.move(self.id, self.x, 0)

        # 패들이 화면의 끝에 부딪히면 공처럼 튕기는게 아니라 움직임이 멈춰야한다.
        # 그래서 왼쪽 x 좌표(pos[0]) 가 0 과 같거나 작으면 self.x = 0 처럼 x 변수에 0 을
        # 설정한다.  같은 방법으로 오른쪽 x 좌표(pos[2]) 가 캔버스의 폭과 같거나 크면
        # self.x = 0 처럼 변수에 0 을 설정한다.

    def turn_left(self, evt):  # 패들의 방향을 전환하는 함수
        self.x = -3

    def turn_right(self, evt):
        self.x = 3

    def move(self, x):
        self.x = x


'''
LYE
1. cyclestart 함수 추가(공이 딱 패들의 높이를 지나 위로 출발하는 시점에 True)
2. 캔버스 높이 및 공/패들 시작점 조정(y 좌표가 3의 배수로 떨어지게)
'''

if __name__ == '__main__':
    tk = Tk()  # tk 를 인스턴스화 한다.
    tk.title("Game")  # tk 객체의 title 메소드(함수)로 게임창에 제목을 부여한다.
    tk.resizable(0, 0)  # 게임창의 크기는 가로나 세로로 변경될수 없다라고 말하는것이다.
    tk.wm_attributes("-topmost", 1)  # 다른 모든 창들 앞에 캔버스를 가진 창이 위치할것을 tkinter 에게 알려준다.

    canvas = Canvas(tk, width=500, height=450, bd=0, highlightthickness=0)
    # bg=0,highlightthickness=0 은 캔버스 외곽에 둘러싼 외곽선이 없도록 하는것이다. (게임화면이 좀더 좋게)
    canvas.pack()  # 앞의 코드에서 전달된 폭과 높이는 매개변수에 따라 크기를 맞추라고 캔버스에 말해준다.
    tk.update()    # tkinter 에게 게임에서의 애니메이션을 위해 자신을 초기화하라고 알려주는것이다.
    paddle = Paddle(canvas, PADDLE_HEIGHT, 'blue')

    # announceterm : 현재 count 출력 term, saveterm : csv에 저장하는 term
    ball1 = Ball(canvas, paddle, 'red', announceterm=500, saveterm=10000)
    ball2 = Ball(canvas, paddle, 'blue', announceterm=500, saveterm=10000)
    start = False
    # 공을 약간 움직이고 새로운 위치로 화면을 다시 그리며, 잠깐 잠들었다가 다시 시작해 ! "
    is_cycling = False

    while 1:
        ball1.draw()

        c_state = ball1.cyclestate()

        if c_state == END:
            # print('END')
            ball1.gameover()
            is_cycling = False
        if c_state == START or ball1.is_paddle_hit():
            # print('START')
            is_cycling = True

        if is_cycling:
            ball1.action()

        ball2.draw()

        c_state = ball2.cyclestate()

        if c_state == END:
            # print('END')
            ball2.gameover()
            is_cycling = False
        if c_state == START or ball2.is_paddle_hit():
            # print('START')
            is_cycling = True

        if is_cycling:
            ball2.action()

        tk.update_idletasks()  # 우리가 창을 닫으라고 할때까지 계속해서 tkinter 에게 화면을 그려라 !
        tk.update()  # tkinter 에게 게임에서의 애니메이션을 위해 자신을 초기화하라고 알려주는것이다.

        time.sleep(0.01)

        #10만번 학습 후에 정상 속도로 플레이 시작(학습 결과 반영됨)
        if ball1.gamecount > 10000:
            time.sleep(0.005)  # 무한 루프중에 100분의 1초마다 잠들어라 !


print('')
print('====================================================================================================')
print('== 문제 258. epsilon 을 0.1 로 했을때와 0.01 로 했을때의 학습 상태를 확인하시오!')
print('====================================================================================================')
from tkinter import *

import random
import time
import csv

EMPTY = 0
PADDLE_HEIGHT = 50.0
PADDLE_MOVE = [-10, 10]

START = 3
END = 4

class Ball:
    def __init__(self, canvas, paddle, color, announceterm, saveterm, winval=1, loseval=-1):
        self.canvas = canvas
        self.paddle = paddle
        self.id = canvas.create_oval(10, 10, 25, 25, fill=color)  # 공 크기 및 색깔
        self.canvas.move(self.id, 245, 200)  # 공을 캔버스 중앙으로 이동
        starts = [-3, 3]  # 공의 속도를 랜덤으로 구성하기 위해 준비한 리스트
        random.shuffle(starts)  # starts 리스트 중에 숫자를 랜덤으로 골라서
        self.x = starts[0]  # 처음 공이 패들에서 움직일때 왼쪽으로 올라갈지 오른쪽으로 올라갈지 랜덤으로 결정되는 부분
        self.y = -3  # 처음 공이 패들에서 움직일때 위로 올라가는 속도
        self.canvas_height = self.canvas.winfo_height()  # 캔버스의 현재 높이를 반환한다.(공이 화면에서 사라지지 않기위해)
        self.canvas_width = self.canvas.winfo_width()  # 캔버스의 현재 넓이를 반환한다.(공이 화면에서 사라지지 않기위해)
        self.hit_bottom = False
        self.values = {}
        self.epsilon = 0.1  # 랜덤율
        self.alpha = 0.99  # 망각계수
        self.learning = True
        self.cycle_data = []
        self.wincount = 0
        self.losecount = 0
        self.gamecount = 0
        self.winval = winval  # 성공 보상치
        self.loseval = loseval
        self.announceterm = announceterm  # 게임 횟수 프린트 텀
        self.saveterm = saveterm  # csv 저장 텀

        # csv 파일에서 gamecount / values 불러옴
        self.loadcsv()


    def action(self):
        r = random.random()
        if r < self.epsilon:
            direction = self.randomChoice()
        else:
            direction = self.greedyChoice()
        x = PADDLE_MOVE[direction]
        # 머신러닝을 위한 한 사이클 내 이동 데이터 저장
        self.cycle_data.append(self.keystate(direction))
        # 이동/그리기
        self.paddle.move(x)
        self.paddle.draw()


    # 이동 방향 랜덤으로 지정
    def randomChoice(self):
        rand = random.choice([0, 1])
        key = self.keystate(rand)
        if key not in self.values:
            self.add(key)
        return rand


    # 이동 방향 Greedy로 지정
    def greedyChoice(self):
        val_left = self.keystate(0)
        val_right = self.keystate(1)

        if self.lookup(val_left) > self.lookup(val_right):
            return 0
        elif self.lookup(val_left) < self.lookup(val_right):
            return 1
        else:
            return random.choice([0, 1])


    def add(self, key):
        self.values[key] = 0


    def lookup(self, key):
        if key not in self.values:
            # print(key)
            self.add(key)
        return self.values[key]


    def hit_paddle(self, pos):  # 패들에 공이 튀기게 하는 함수
        paddle_pos = self.canvas.coords(self.paddle.id)
        # 공의 x좌표가 패들의 너비 안에 있는지 / 공 바닥이 패들 윗면에 닿아 있는지
        if pos[2] >= paddle_pos[0] and pos[0] <= paddle_pos[2] and pos[1] == PADDLE_HEIGHT:
            return True
        return False


    # 공을 패들로 받았는지 여부 출력(True/False)
    def is_paddle_hit(self):
        return self.hit_paddle(self.canvas.coords(self.id))


    def draw(self):
        # 볼의 현재 좌표를 출력해준다. 공 좌표( 좌상단 x,y좌표 / 우하단 x,y좌표 )
        pos = self.canvas.coords(self.id)

        if pos[1] <= 0:
            self.y = 3
        if pos[3] >= self.canvas_height:
            self.y = -3
        if pos[0] <= 0:
            self.x = 3
        if pos[2] >= self.canvas_width:
            self.x = -3
        if self.hit_paddle(pos) == True:
            self.y = 3

        self.canvas.move(self.id, self.x, self.y)  # 공을 움직이게 하는 부분
        # 공이 화면 밖으로 나가지 않게 해준다


    def keystate(self, movement):
        paddle_pos = self.canvas.coords(self.paddle.id)
        ball_pos = self.canvas.coords(self.id)
        # paddle 위치(좌측 x좌표), 공의 좌상단 x/y좌표, 공의 좌우/상하 속도(방향), paddle을 좌/우 중 어느 쪽으로 움직이는지
        return (paddle_pos[0], (ball_pos[0], ball_pos[1]), (self.x, self.y), movement)


    # 사이클 시작 : 1, 사이클 종료 : -1, 해당 없음 : 0
    def cyclestate(self):
        pos = self.canvas.coords(self.id)
        if pos[3] == PADDLE_HEIGHT:
            if self.y == 3:
                return START
            elif self.y == -3:
                return END
        return 0


    # 결과 학습
    def backup(self, newVal, idx):
        if idx >= 0 and self.learning:
            prevVal = self.values[self.cycle_data[idx]]
            self.values[self.cycle_data[idx]] += self.alpha * (newVal - prevVal)
            # print("key : {0}, val : {1}".format(self.cycle_data[idx],self.values[self.cycle_data[idx]]))
            self.backup(newVal * self.alpha, idx - 1)


    # 게임 끝났을 시 학습 및 cycle_data 초기화
    def gameover(self):
        if self.learning:
            # paddle로 받았으면 1, 못 받았으면 -1
            if self.is_paddle_hit():
                result_value = self.winval
                self.wincount += 1
            else:
                result_value = self.loseval
                self.losecount += 1
            self.backup(result_value, len(self.cycle_data) - 1)
            self.gamecount += 1

            # saveterm마다 csv 저장
            if self.gamecount % self.saveterm == 0:
                self.writecsv()
            if self.gamecount % self.announceterm == 0:
                print("cycle count : {0}".format(ball.gamecount))
        self.cycle_data.clear()


    def winnerval(self, winner):
        if winner == 'hit':
            return 1
        elif winner == 'miss':
            return -1
        else:
            return 0


    # 게임 결과 csv로 저장
    def writecsv(self):
        try:
            # Values 저장
            Fn = open("D:\\KYH\\02.PYTHON\\data\\pong_value.csv", 'w', newline='')
            writer = csv.writer(Fn, delimiter=',')
            writer.writerow([self.gamecount])  # 첫줄에 학습 게임 횟수 저장
            keys = self.values.keys()
            for key in keys:
                writer.writerow([key[0],
                                 key[1][0],
                                 key[1][1],
                                 key[2][0],
                                 key[2][1],
                                 key[3],
                                 ball.values[key]
                                 ])
            Fn.close()

            # 성공/실패 횟수 저장
            Fn = open("D:\pong_score.csv", 'a', newline='')
            writer = csv.writer(Fn, delimiter=',')
            writer.writerow([self.wincount, self.losecount, self.gamecount])
            Fn.close()
            # 승률의 변화를 확인하기 위해 일정 판수마다 카운트 리셋
            self.wincount = 0
            self.losecount = 0

            print("save data in cycle {0}.".format(self.gamecount))
        except Exception as e:
            print('save data failed in cycle {0}.\nError Type : {1}'.format(self.gamecount, type(e).__name__))


    def loadcsv(self):
        try:
            Fn = open("D:\\KYH\\02.PYTHON\\data\\pong_value.csv", 'r')
            self.gamecount = int(Fn.readline().split(',')[0])  # 첫 줄의 학습 게임 횟수 불러오기
            reader = csv.reader(Fn, delimiter=',')
            for key in reader:
                self.values[(
                int(float(key[0])), (int(float(key[1])), int(float(key[2]))), (int(float(key[3])), int(float(key[4]))),
                int(float(key[5])))] = float(key[6])
            print('Load Success! Start at cycle {0}'.format(self.gamecount))
        except Exception:
            print('Load Failed!')


class Paddle:
    def __init__(self, canvas, y_loc, color):
        self.canvas = canvas
        self.id = canvas.create_rectangle(0, 0, 100, 10, fill=color)  # 패들의 높이와 넓이 그리고 색깔

        self.canvas.move(self.id, 200, y_loc)  # 패들 사각형을 200,300 에 위치
        self.x = 0  # 패들이 처음 시작할때 움직이지 않게 0으로 설정
        self.canvas_width = self.canvas.winfo_width()  # 캔버스의 넓이를 반환한다. 캔버스 밖으로 패들이 나가지 않도록
        self.canvas.bind_all('<KeyPress-Left>', self.turn_left)  # 왼쪽 화살표 키를 '<KeyPress-Left>'  라는 이름로 바인딩
        self.canvas.bind_all('<KeyPress-Right>', self.turn_right)  # 오른쪽도 마찬가지로 바인딩한다.

    def draw(self):
        pos = self.canvas.coords(self.id)
        # print(pos)
        if pos[0] <= 0 and self.x < 0:  # 패들의 위치가 왼쪽 끝이고, 이동하려는 방향이 왼쪽이면 함수 종료(이동 안 함)
            return
        elif pos[2] >= self.canvas_width and self.x > 0:  # 패들의 위치가 오른쪽 끝이고, 이동하려는 방향이 오른쪽이면 함수 종료
            return
        self.canvas.move(self.id, self.x, 0)

        # 패들이 화면의 끝에 부딪히면 공처럼 튕기는게 아니라 움직임이 멈춰야한다.
        # 그래서 왼쪽 x 좌표(pos[0]) 가 0 과 같거나 작으면 self.x = 0 처럼 x 변수에 0 을
        # 설정한다.  같은 방법으로 오른쪽 x 좌표(pos[2]) 가 캔버스의 폭과 같거나 크면
        # self.x = 0 처럼 변수에 0 을 설정한다.

    def turn_left(self, evt):  # 패들의 방향을 전환하는 함수
        self.x = -3

    def turn_right(self, evt):
        self.x = 3

    def move(self, x):
        self.x = x


'''
LYE
1. cyclestart 함수 추가(공이 딱 패들의 높이를 지나 위로 출발하는 시점에 True)
2. 캔버스 높이 및 공/패들 시작점 조정(y 좌표가 3의 배수로 떨어지게)
'''

if __name__ == '__main__':
    tk = Tk()  # tk 를 인스턴스화 한다.
    tk.title("Game")  # tk 객체의 title 메소드(함수)로 게임창에 제목을 부여한다.
    tk.resizable(0, 0)  # 게임창의 크기는 가로나 세로로 변경될수 없다라고 말하는것이다.
    tk.wm_attributes("-topmost", 1)  # 다른 모든 창들 앞에 캔버스를 가진 창이 위치할것을 tkinter 에게 알려준다.

    canvas = Canvas(tk, width=500, height=450, bd=0, highlightthickness=0)
    # bg=0,highlightthickness=0 은 캔버스 외곽에 둘러싼 외곽선이 없도록 하는것이다. (게임화면이 좀더 좋게)
    canvas.pack()  # 앞의 코드에서 전달된 폭과 높이는 매개변수에 따라 크기를 맞추라고 캔버스에 말해준다.
    tk.update()    # tkinter 에게 게임에서의 애니메이션을 위해 자신을 초기화하라고 알려주는것이다.
    paddle = Paddle(canvas, PADDLE_HEIGHT, 'blue')

    # announceterm : 현재 count 출력 term, saveterm : csv에 저장하는 term
    ball = Ball(canvas, paddle, 'red', announceterm=500, saveterm=5000)
    start = False
    # 공을 약간 움직이고 새로운 위치로 화면을 다시 그리며, 잠깐 잠들었다가 다시 시작해 ! "
    is_cycling = False

    while 1:
        ball.draw()

        c_state = ball.cyclestate()
        if c_state == END:
            # print('END')
            ball.gameover()
            is_cycling = False
        if c_state == START or ball.is_paddle_hit():
            # print('START')
            is_cycling = True

        if is_cycling:
            ball.action()

        tk.update_idletasks()  # 우리가 창을 닫으라고 할때까지 계속해서 tkinter 에게 화면을 그려라 !
        tk.update()  # tkinter 에게 게임에서의 애니메이션을 위해 자신을 초기화하라고 알려주는것이다.

        #10만번 학습 후에 정상 속도로 플레이 시작(학습 결과 반영됨)
        if ball.gamecount > 10000:
            time.sleep(0.005)  # 무한 루프중에 100분의 1초마다 잠들어라 !


print('')
print('====================================================================================================')
print('== 문제 259. R을 사용해 그래프를 그리시오.')
print('====================================================================================================')
# for(i in 1:10000){
# aa <- read.csv("D:\\KYH\\02.PYTHON\\data\\pong_score_001.csv",header=F)
# bb <- aa$V1
# cc <- aa$V2
# dd<- aa$V3
# ee <- max(bb,cc)
# plot(bb,type='l',col="blue", ylim=c(0,ee))
# par(new=T)
# plot(cc,type='l',col="red", ylim=c(0,ee))
# Sys.sleep(1)
# }