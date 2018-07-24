# ■ 8장. 모듈과 패키지
#  * 8장. 목차
#   1. 모듈이란?
#   2. import 사용법
#   3. 모듈 찾는 방법
#   4. 메인 모듈과 하위 모듈
#   5. 패키지
#   6. __init__.py

# ■ 8.1. 모듈이란?
#  모듈 ? 독자적인 기능을 갖는 구성요소
#   파이썬에서는 각각의 소스 파일을 일컬어 모듈이라고 한다.


print('')
print('====================================================================================================')
print('== 문제 159. 오전에 만들었던 준호의 최대 공약수 구하는 함수를 모듈화 하시오.')
print('====================================================================================================')
from PythonClass.module_test import gcd_list
gcd_list(1000, 500, 250, 100, 25, 25)


print('')
print('====================================================================================================')
print('== 문제 160. 표준편차를 출력하는 함수를 모듈화시켜서 다른 실행창에서 아래와 같이 실행하면 실행되게 하시오.')
print('====================================================================================================')
from PythonClass.module_test import stddev
print(stddev(2.3, 1.7, 1.4, 0.7, 1.9))


# ■ 파이썬은 import 를 만나면 아래와 같은 순서로 모듈 파일을 찾아 나선다.
#  1. 파이썬 인터프리터 내장 모듈
#  2. sys.path 에 정의되어있는 디렉토리
import sys
print(sys.builtin_module_names)
print(sys.path)


print('')
print('====================================================================================================')
print('== 문제 161. 혜승이가 sys 모듈의 random 함수를 이용해서 구현해 낸 원주율 구하는 코드를 실행해보시오.')
print('====================================================================================================')
import matplotlib.pyplot as plt
import math
import random

circle = plt.Circle((0, 0), radius=1.0, fc='w', ec='b')  # 원 만들기
# 원의 중심이(0,0) 이고 반지름이 1이다. 채워짐색: white 테두리색: black
ax = plt.axes(xlim=(-1, 1), ylim=(-1, 1))  # x축 범위 (-1.1) y축 범위(-1,1)
ax.set_aspect('equal')  # 가로 세로축이 같은 스케일이 되도록 크기 조정
plt.gca().add_patch(circle)  # 원 그리기
dot_cnt = 1000  # dot_cnt가  100000일 때와 1000000일 때를 비교
cnt_in = 0

for i in range(dot_cnt):
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    plt.plot([x], [y], 'ro')  # 랜덤 점 찍기
    if math.pow(x, 2) + math.pow(y, 2) <= 1:
        cnt_in += 1

plt.show()
print('원의 넓이 : ')
print((cnt_in / dot_cnt) * 4)


# 필요한 모듈
import pygame, sys
import random
from pygame.locals import *
import math

# 함수 생성(while 절에서 쓰임)
def Input(events):
    for event in events:
        if event.type == QUIT:
            sys.exit(0)
        else:
            print(event)

# 변수 설정
dot_cnt = int(input('찍을 점의 개수 : '))  # 점 찍을 총 횟수
i = 0  # 점 찍기 수행횟수
cnt_in = 0  # 원 안에 들어간 횟수

# 출력 사이즈 설정
width = 600  # 점이 찍힐 정사각형 가로길이
height = 600  # 점이 찍힐 정사각형 세로길이
info_height = 50  # total_cnt, dot_cnt, cnt_in, pi 정보 출력란 사이즈
radius = int(width / 2)  # 원의 반지름

# 색깔 설정
white = (255, 255, 255)
red = (255, 0, 0)
blue = (0, 0, 255)
white_red = (255, 200, 200)
white_blue = (200, 200, 255)
white_black = (200, 200, 200)

# 파이게임 설정
pygame.init()
pygame.display.set_caption('원주율(π) 구하기')
font = pygame.font.SysFont("Arial", 20, 0, 0)
window = pygame.display.set_mode((width, height + info_height))
screen = pygame.display.get_surface()  # 바탕화면을 변수 screen에 할당
screen.fill(white)  # 바탕화면 screen 의 색깔은 하얀색

# 배경화면 색깔 설정
pygame.draw.rect(screen, white_red, (0, 0, width, height), 0)  # 점이 찍힐 정사각형 부분은 연한 빨간색으로 설정
pygame.draw.circle(screen, white_blue, (radius, radius), radius, 0)  # 원 부분은 연한 파란색으로 설정
pygame.draw.rect(screen, white_black, (0, height, width, info_height), 0)  # 정보 출력란은 연한 검정색으로 설정

# 출력 위한 루프문
while True:
    Input(pygame.event.get())  # 위에서 생성한 input 함수

    if i >= dot_cnt:  # 점 찍은 횟수가 총 횟수보다 커지면 건너뜀
        continue
    else:  # 점 찍은 횟수가 총 횟수 미만이면 아래 작업 수행
        i += 1
        x = random.uniform(-radius, radius)  # 난수 생성. 원의 중심점을 (0,0)이라 했을 때 난수 생성 범위는 (-반지름 ~ 반지름) 사이
        y = random.uniform(-radius, radius)

        if math.pow(x, 2) + math.pow(y, 2) <= math.pow(radius, 2):  # 피타고라스 정리 이용하여 원 안에 점이 찍힌 경우를 판별
            cnt_in += 1
            pygame.draw.circle(screen, blue, (int(x + radius), int(y + radius)), 1, 0)  # 원 안에 점이 찍힌 경우 파란 점 찍기.
            # 이때 파이게임의 좌표는 좌측 상단의 좌표가 (0,0) 이므로
            # 범위가 (-반지름 ~ 반지름) 인 x, y 에 (+반지름) 해야함.
        else:
            pygame.draw.circle(screen, red, (int(x + radius), int(y + radius)), 1, 0)  # 원 안에 들어오지 않은 경우 빨간 점 찍기

    # while 루프 돌때마다 정보 출력란 리셋(안그러면 글자가 겹침)
    pygame.draw.rect(screen, white_black, (0, height + 1, width, height + info_height))

    # 정보 출력란 텍스트 생성
    text_total_cnt = font.render("Total_cnt: " + repr(dot_cnt), 1, (0, 0, 0))
    text_dot_cnt = font.render("Dot_cnt: " + repr(i), 1, (0, 0, 0))
    text_cnt_in = font.render("Cnt_in: " + repr(cnt_in), 1, (0, 0, 0))

    # 생성한 텍스르 출력
    screen.blit(text_total_cnt,
                (1 * width / 24, height + (info_height / 3)))  # (1*width/24,height + (info_height/3))부분은 출력 위치 좌표
    screen.blit(text_dot_cnt, (7 * width / 24, height + (info_height / 3)))
    screen.blit(text_cnt_in, (13 * width / 24, height + (info_height / 3)))

    # 파이 계산 및 텍스트 출력(try 와 except 절 안 써도 출력 가능)
    try:
        pi = repr(round((cnt_in / i) * 4, 6))
        text_pi = font.render("π : " + pi, 1, (0, 0, 0))
        screen.blit(text_pi, (19 * width / 24, height + (info_height / 3)))

    except ZeroDivisionError as message:
        print(message)

    pygame.display.flip()


# ■ 8.5. 패키지
#  패키지란 ? 모듈을 모아놓은 디렉토리를 말한다.


# ■ 8.6. __init__.py
#  __init__.py 는 보통 비워둡니다. 이 파일을 손대는 경우는 __all__ 이라는 변수를 조정할 때 이다.
#  __all__ 변수 ? 패키지로부터 반입할 모듈의 목록을 정의하기 위해 사용한다.
#  어떤 모듈들이 있는지 파이썬이 알려면 __init__.py 에 __all__ 변수에 명시해줘야한다.
#  예) __all__=['plus_module', 'minus_module', 'multiply_module']


# ■ 8.7. site-package 란 ? 파이썬의 기본 라이브러리 패키지 외에 추가적인 패키지를 설치하는 디렉토리.
#  * site-package 의 위치를 확인하는 방법
import sys
print(sys.path)


print('')
print('====================================================================================================')
print('== 문제 162. d:\\my_loc2 라는 디렉토리를 만들고 cal_test.py 스크립트를 가져다두고 실행해보시오!')
print('====================================================================================================')
