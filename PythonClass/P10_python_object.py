# ■ 9장. 객체와 클래스

# 9.1. 객체와 클래스
#  객체 = 속성 + 기능
class Car:
    def __init__(self):
        self.color = 0xFF0000
        self.wheel_size = 16
        self.displacement = 2000

    def forward(self):
        pass

    def backward(self):
        pass

    def turn_left(self):
        pass

    def turn_right(self):
        pass

if __name__ == '__main__':
    my_car = Car()

    print('0x{:02X}'.format(my_car.color))
    print(my_car.wheel_size)
    print(my_car.displacement)

    my_car.forward()
    my_car.backward()
    my_car.turn_left()
    my_car.turn_right()


# 9.2. 클래스의 정의
#  클래스(자료형) --> 객체(변수)
#  위의 클래스는 자료형이고 아직 객체가 되지 않았다.


# 9.3. __init__() 메소드를 이용한 초기화
#  클래스의 생성자가 호출이 되면 내부적으로 두 개의 메소드가 호출
#  1. __new__() : 클래스의 인스턴스를 만드는 역할 (내부적으로 자동으로 호출)
#  2. __init__() : 객체가 생성될 때 객체를 초기화하는 역할

# 1) __init__() 메소드를 지정안했을 때


print('')
print('====================================================================================================')
print('== 문제 164. 초기화 코드를 구현해서 아래와 같이 출력하시오!')
print('====================================================================================================')
class ClassVar:
    def __init__(self):
        self.text_list = []

    def add(self, text):
        self.text_list.append(text)

    def print_list(self):
        print(self.text_list)

if __name__ == '__main__':
    a = ClassVar()
    a.add('a')
    a.print_list()

    b = ClassVar()
    b.add('b')
    b.print_list()


print('')
print('====================================================================================================')
print('== 문제 165. 머신러닝 코드 입히기 전인 핑퐁 게임을 파이썬으로 구현하시오.')
print('====================================================================================================')
# from tkinter import *
# import random
# import time
#
# class Ball:
#     def __init__(self, canvas, paddle, color):
#         self.canvas = canvas
#         self.paddle = paddle
#         self.id = canvas.create_oval(10, 10, 25, 25, fill=color) #공 좌표 및 색깔(oval : object 형태 타입)
#         self.canvas.move(self.id, 245, 100) #공을 캔버스 중앙으로 이동
#         starts = [-3, -2, -1, 1, 2, 3]
#         random.shuffle(starts)
#         #공의 속도
#         self.x = starts[0]
#         self.y = -3
#         self.canvas_height = self.canvas.winfo_height()
#         self.canvas_width = self.canvas.winfo_width()
#         self.hit_bottom = False
#
#     def hit_paddle(self,pos):
#         paddle_pos = self.canvas.coords(self.paddle.id)
#         if pos[2] >= paddle_pos[0] and pos[0] <= paddle_pos[2]:
#             if pos[3] >= paddle_pos[1] and pos[3] <= paddle_pos[3]:
#                 return True
#         return False
#
#     def draw(self):
#         self.canvas.move(self.id, self.x, self.y) #공을 움직이게 하는 부분
#         #공이 화면 밖으로 나가지 않게 해준다
#         pos = self.canvas.coords(self.id)
#         if pos[1] <= 0:
#             self.y = 3
#         if pos[3] >= self.canvas_height: #바닥에 부딪히면 게임오버
#             self.hit_bottom = True
#         if pos[0] <= 0:
#             self.x = 3
#         if pos[2] >= self.canvas_width:
#             self.x = -3
#         if self.hit_paddle(pos) == True: #판에 부딪히면 위로 튕겨올라가게
#             self.y = -3
#
# class Paddle:
#
#     def __init__(self,canvas,color):
#         self.canvas = canvas
#         self.id = canvas.create_rectangle(0,0,100,10,fill=color)
#         self.canvas.move(self.id, 200, 300)
#         self.x = 0
#         self.canvas_width = self.canvas.winfo_width()
#         self.canvas.bind_all('<KeyPress-Left>',self.turn_left)
#         self.canvas.bind_all('<KeyPress-Right>',self.turn_right)
#
#     def draw(self):
#         self.canvas.move(self.id, self.x, 0)
#         pos = self.canvas.coords(self.id)
#         if pos[0] <= 0:
#             self.x = 0
#         elif pos[2] >= self.canvas_width:
#             self.x = 0
#
#     def turn_left(self,evt):
#         self.x = -2
#
#     def turn_right(self,evt):
#         self.x = 2
#
# tk = Tk()
# tk.title("Game")
# tk.resizable(0, 0)
# tk.wm_attributes("-topmost", 1)
# canvas = Canvas(tk, width=500, height=400, bd=0, highlightthickness=0)
# canvas.pack()
# tk.update()
# paddle = Paddle(canvas,'blue')
# ball = Ball(canvas, paddle, 'red')
# start = False
# #공을 약간 움직이고 새로운 위치로 화면을 다시 그리며, 잠깐 잠들었다가 다시 시작해라!
# while 1:
#     if ball.hit_bottom == False:
#         ball.draw()
#         paddle.draw()
#     #그림을 다시 그려라! 라고 쉴새없이 명령
#     tk.update_idletasks()
#     tk.update()
#     #시작 전 2초간 sleep
#     if not start:
#         time.sleep(2)
#         start = True
#     time.sleep(0.01)


# 9.4. self 에 대한 이해
#  "self 는 클래스에서 사용하는 최초 매개변수인데 자기 자신을 가리킨다"


# 9.5. 정적 메소드와 클래스 메소드
#  1. 인스턴스 메소드
#  2. 클래스 메소드

# class Calculator:
#     @staticmethod
#     def plus(a, b):
#         return a + b
#
#     @staticmethod
#     def minus(a, b):
#         return a - b
#
#     @staticmethod
#     def multiply(a, b):
#         return a * b
#
#     @staticmethod
#     def divide(a, b):
#         return a / b

# if __name__ == '__main__':
#     print("{0} + {1} = {2}".format(7, 4, Calculator.plus(7, 4)))
#     print("{0} - {1} = {2}".format(7, 4, Calculator.minus(7, 4)))
#     print("{0} * {1} = {2}".format(7, 4, Calculator.multiply(7, 4)))
#     print("{0} / {1} = {2}".format(7, 4, Calculator.divide(7, 4)))


# 9.6. 클래스 내부에게만 열려있는 private 멤버
#  * 파이썬에서 사용하는 멤버 2가지
#   1. public number : 클래스 안에서든 밖에서는 접근 가능한 멤버
#    - __number__

#   2. private member : 클래스     안에서만 접근 가능한 멤버
#    - __number, __number_
class yourclass:
    pass

class myclass:
    def __init__(self):
        self.message = 'Hello'
        self.__private = 'private'  # 외부에서 접근 불가

    def some_method(self):
        print(self.message)
        print(self.__private)

obj = myclass()
obj.some_method()
print(obj.message)


# 9.7. 상속
class father:
    def base_method(self):
        print('hello~~')

class child(father):
    pass

father = father()
father.base_method()

child = child()
child.base_method()

# 예제 : __init__ 메소드를 가지고 실행하는데 부모와는 틀리게 자식에 message 라는 속성이 없어서 상속을 시키고 싶을 때?
class father:
    def __init__(self):
        print('hello~~')
        self.message = 'Good Morning'

class child(father):
    def __init__(self):
        super().__init__()
        print("hello ~~ I'm tired")

child = child()
print(child.message)

# 9.8. 다중 상속
#  두 개 이상의 클래스를 상속받는 것을 말함
#  이 경우에는 두 클래스의 모든 속성을 물려받게 됨
# class father1:
#     def func(self):
#         print('지식')
#
# class father2:
#     def func(self):
#         print('지혜')
#
# class child(father1, father2):
#     def childfunc(self):
#         father1.func(self)
#         father2.func(self)
#
# objectchild = child()
# objectchild.childfunc()
# objectchild.func()  # 상속받은 순서대로 출력
#
# class grandfather:
#     def __init__(self):
#         print('튼튼한 두팔')
#
# class father1(grandfather):
#     def __init__(self):
#         grandfather.__init__(self)g
#         print('지식')
#
# class father2(grandfather):
#     def __init__(self):
#         grandfather.__init__(self)
#         print('지혜')
#
# class grandchild(father1, father2):
#     def __init__(self):
#         father1.__init__(self)
#         father2.__init__(self)
#         print('자기 만족도가 높은 삶')
#
# grandchild = grandchild()


print('')
print('====================================================================================================')
print('== 문제 166. 다시 팔이 2개가 되게하시오!')
print('====================================================================================================')
# super() : 부모 클래스 또는 형제 클래스를 지칭(다이아몬드 형태일 경우 형제 노드가 존재하면 형제 노드를 지칭함)
# class.__mro__ : 해당 클래스가 슈퍼 클래스들에 대해 수행되는 순서를 나타냄
# class.__bases__ : 해당 클래스의 부모 클래스의 리스트를 출력
class grandfather:
    def __init__(self):
        print('튼튼한 두팔')

class father1(grandfather):
    def __init__(self):
        # super().__init__()
        grandfather.__init__(self)
        print('지식')

class father2(grandfather):
    def __init__(self):
        super().__init__()
        # grandfather.__init__(self)
        print('지혜')

class grandchild(father1, father2):
    def __init__(self):
        super().__init__()
        print('자기 만족도가 높은 삶')

child = grandchild()


# 9.6. 오버라이딩
#  부모로 부터 상속받은 메소드를 다시 override(재정의) 하겠다.
# class grandfather:
#     def __init__(self):
#         print('튼튼한 두팔')
#
# class father2(grandfather):
#     def __init__(self):
#         print('지혜')
#
# father2 = father2()


# 9.7. 데코레이터 사용법
#  1. 함수를 강력하게 해준다.
#  2. 공통적으로 사용하는 코드를 쉽게 관리.
def greet(name):
    return 'Hello {}'.format(name)

greet_someone = greet
print(greet_someone('scott'))
#################################################
def greeting(name):
    def greet_message():
        return 'Hello'
    return '{} {}'.format(greet_message(), name)

print(greeting('scott'))
#################################################
def greet(name):
    return 'Hello {}'.format(name)

def change_name_greet(func):
    name = 'King'
    return func(name)

print(change_name_greet(greet))
#################################################
def greet(name):
    return 'Hello {}'.format(name)

def uppercase(func):
    def wrapper(name):
        result = func(name)
        return result.upper()
    return wrapper

new_greet = uppercase(greet)
print(new_greet('scott'))
#################################################


print('')
print('====================================================================================================')
print('== 문제 167. 아래와 같이 이름을 입력하고 함수를 실행하면 해당하는 사원의 직업이 소문자로 출력되는 함수를 생성하시오.')
print('====================================================================================================')
from PythonClass.common_func import CommonFunc
emp_list = []

for empData in CommonFunc.returnCsvData('emp2.csv'):
    emp_list.append(empData)

def find_job(ename):
    for emp in emp_list:
        if ename.upper() == emp[1]:
            return emp[2]
    return None

def lowercase(func):
    def wrapper(name):
        result = func(name)
        return '해당 사원이 존재하지 않습니다.' if result is None else result.lower()
    return wrapper

new_find_job = lowercase(find_job)
print(new_find_job('scott'))


# ■ 데코레이터 표현법을 보기전에 먼저 데코레이터와 같은 역할을 하는 함수를 생성
class Greet(object):
    current_user = None

    def set_name(self, name):
        if name == 'admin':
            self.current_user = name
        else:
            raise Exception('권한이 없네요')

    def get_greeting(self, name):
        if name == 'admin':
            return 'Hello {}'.format(self.current_user)

greet = Greet()
greet.set_name('admin')
print(greet.get_greeting('admin'))


print('')
print('====================================================================================================')
print('== 문제 168. 위의 코드에서 중복적으로 사용되는 코드를 떼어내서 하나의 함수로 생성하시오.')
print('====================================================================================================')
class Greet(object):
    current_user = None

    def is_admin(self, user_name):
        if user_name != 'admin':
            raise Exception('권한이 없네요')

    def set_name(self, name):
        self.is_admin(name)
        self.current_user = name

    def get_greeting(self, name):
        self.is_admin(name)
        return 'Hello {}'.format(self.current_user)

greet = Greet()
greet.set_name('admin')
print(greet.get_greeting('admin'))


print('')
print('====================================================================================================')
print('== 문제 169. 이름을 넣어서 함수를 실행하면 해당 사원의 월급이 출력되게하는 함수를 생성하는데 KING 만 월급을 볼 수 있게 하고')
print('==  KING 이 아닌 다른 사원들은 권한이 없다면서 볼 수 없게 에러가 나게 하시오.')
print('====================================================================================================')
from PythonClass.common_func import CommonFunc
emp_list = []

for empData in CommonFunc.returnCsvData('emp2.csv'):
    emp_list.append(empData)

class FindSal:
    current_name = ''

    def is_admin(self, user_name):
        if user_name != 'KING':
            raise Exception('권한이 없습니다!')

    def set_name(self, name):
        self.is_admin(name.upper())
        FindSal.current_name = name.upper()

    def get_sal(self):
        self.is_admin(FindSal.current_name)
        for data in emp_list:
            if data[1] == FindSal.current_name:
                return data[5]
        return None

find_sal = FindSal()
find_sal.set_name('king')
print(find_sal.get_sal())


print('')
print('====================================================================================================')
print('== 문제 170. 위에 is_admin(name) 이라는 함수를 사용해서 코드가 더 좋아졌다. 하지만 데코레이터를 쓰면 더 좋은 코드가 될 수 있다.')
print('==  데코레이터를 써서 구현하시오!')
print('====================================================================================================')
class Greet:
    current_name = None

    def is_admin(self, func):
        def wrapper(*args, **kwargs):
            if kwargs.get('name') != 'KING':
                raise Exception('권한이 없다니까요.')
            return func(*args, **kwargs)
        return wrapper

    @is_admin
    def set_name(self, name):
        FindSal.current_name = name

    @is_admin
    def get_greeting(self):
        return 'Hello {}'.format(FindSal.current_name)

greet = Greet()
greet.set_name(name='admin')
print(greet.get_greeting())


print('')
print('====================================================================================================')
print('== 문제 171. 문제 169번 코드를 데코레이터 함수를 이용해서 더 좋게 개선시키시오.')
print('====================================================================================================')
from PythonClass.common_func import CommonFunc
emp_list = []

for empData in CommonFunc.returnCsvData('emp2.csv'):
    emp_list.append(empData)

class FindSal:
    current_name = ''

    def is_admin(self, user_name):
        if user_name != 'KING':
            raise Exception('권한이 없습니다!')

    @is_admin()
    def set_name(self, name):
        FindSal.current_name = name.upper()

    @is_admin(FindSal.current_name)
    def get_sal(self):
        for data in emp_list:
            if data[1] == FindSal.current_name:
                return data[5]
        return None

find_sal = FindSal()
find_sal.set_name(user_name='KING')
print(find_sal.get_sal())


print('')
print('====================================================================================================')
print('== 문제 172. gun 이라는 인스턴스를 생성하기 위해서 gun() 클래스 생성하시오.')
print('====================================================================================================')
class gun(object):
    def __init__(self):
        self.bullet = 0

    def charge(self):
        gun.bullet += 10

    def shoot(self, shoot_cnt):
        for i in shoot_cnt:
            print('탕!')
        gun.bullet -= shoot_cnt

    def print(self):
        print('현재 총알이 {}발 남았습니다.'.format(gun.bullet))


print('')
print('====================================================================================================')
print('== 문제 174. 자기 자신이 인스턴스 메소드에 인자로 전달된다는 것이 어떤것인지 인스턴스를 통하지 않고 클래스의 ')
print('==  introduce_myself 를 직접 호출해서 확인하시오.')
print('====================================================================================================')
class student(object):
    def __init__(self, name, year, class_num, student_id):
        self.name = name
        self.year = year
        self.class_num = class_num
        self.student_id = student_id

    def introduce_myself(self):
        return '{}, {}학년 {}반 {}번'.format(self.name, self.year, self.class_num, self.student_id)

stu = student('김인호', 2, 3, 35)
print(student.introduce_myself(stu))


print('')
print('====================================================================================================')
print('== 문제 175. gun class 의 메소드들을 static method 들로 변경해서 다시 총을 쏘시오')
print('====================================================================================================')
class gun(object):
    bullet = 0

    @staticmethod
    def charge():
        gun.bullet += 10

    @staticmethod
    def shoot(shoot_cnt):
        for i in range(shoot_cnt):
            print('탕!')
        gun.bullet -= shoot_cnt

    @staticmethod
    def print():
        print('현재 총알이 {}발 남았습니다.'.format(gun.bullet))

gun1 = gun()
gun1.charge()
gun1.shoot(1)
gun1.print()


print('')
print('====================================================================================================')
print('== 문제 176. static method 로 선언한 클래스를 이용해서 인스턴스화 한 두 개의 총을 쏘는 메소드가 서로 같은 메모리를')
print('==  쓰는지 다른 메모리를 쓰는지 확인하시오.')
print('====================================================================================================')
class gun(object):
    bullet = 0

    @staticmethod
    def charge():
        gun.bullet += 10

    @staticmethod
    def shoot(shoot_cnt):
        for i in range(shoot_cnt):
            print('탕!')
        gun.bullet -= shoot_cnt

    @staticmethod
    def print():
        print('현재 총알이 {}발 남았습니다.'.format(gun.bullet))

gun1 = gun()
gun1.charge()
gun1.shoot(1)
gun1.print()

gun2 = gun()
gun2.shoot(3)
gun2.print()


# 9.10. 마법의 __call__ 메소드
# 예제 1
class Sample(object):
    def __init__(self):
        print('전 생성하면 바로 실행되요')

sample = Sample()
# 설명 : __init__ 메소드는 인스턴스를 생성하면 바로 실행이 된다.

# 예제 2
class Sample2(object):
    def __call__(self):
        print('인스턴스에 괄호를 붙이면 제가 실행되요')

sample2 = Sample2()
sample2()


# ■ 클래스를 데코레이터로 구현하는 예제
class onlyadmin(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        name = kwargs.get('name').upper()
        self.func(name)

@onlyadmin
def greet(name):
    print('Hello {}'.format(name))

greet(name='Scott')


print('')
print('====================================================================================================')
print('== 문제 177. 위의 onlyadmin 데코레이터를 활용해서 find_job 이라는 함수를 강력하게 하시오.')
print('====================================================================================================')
from PythonClass.common_func import CommonFunc
class onlyadmin(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        name = kwargs.get('name').upper()
        self.func(name)

@onlyadmin
def find_job(name):
    for empData in CommonFunc.returnCsvData('emp2.csv'):
        if name == empData[1]:
            print('Job : {}'.format(empData[2]))

find_job(name='scott')


# 추상 기반 클래스
#  상속이란 클래스의 재 사용을 높임으로써 코드 반복을 줄이고, 유지보수 비용 낮추는데 그 목적이 있다.
#  추상 클래스는 추상 메소드가 있는 클래스를 말한다.
#  추상 메소드는 바디가 없는 메소드이다.
#  추상 클래스를 상속받는 자식 클래스에서 반드시 지켜줘야할 사항은?
#   추상 클래스내에는 추상 메서드와 일반 메서드들이 있을텐데 그 중에 추상 메서드는 반드시 가져와서 오버라이드 해야한다.
from abc import ABCMeta, abstractmethod  # 파이썬은 추상 클래스를 제공하지 않아서 외부 라이브러리를 받아서 구현
class Animal(object):
    __metaclass__ = ABCMeta  # 추상 클래스로 선언

    @abstractmethod  # 추상 메소드 선언
    def bark(self):
        pass  # 비어있는 메소드, 중요한 메소드

class Cat(Animal):
    def __init__(self):
        self.sound = '야옹'

    def bark(self):
        return self.sound

class Dog(Animal):
    def __init__(self):
        self.sound = '멍멍'

    def bark(self):
        return self.sound

cat = Cat()
dog = Dog()

print(cat.bark())
print(dog.bark())


print('')
print('====================================================================================================')
print('== 문제 178. 음료라는 추상클래스를 생성하고 아메리카노와 카페라떼 클래스를 자식 클래스로 생성하시오!')
print('====================================================================================================')
from abc import ABCMeta, abstractmethod

class Beverage(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def cost(self):
        pass

class Americano(Beverage):
    def __init__(self):
        self.price = 3.5
        self.name = 'americano'

    def cost(self):
        return (self.name, self.price)

class Caffelatte(Beverage):
    def __init__(self):
        self.price = 4.0
        self.name = 'caffelatte'

    def cost(self):
        return (self.name, self.price)

americano = Americano()
caffelatte = Caffelatte()

print(americano.cost()[0], americano.cost()[1])
print(caffelatte.cost()[0], caffelatte.cost()[1])