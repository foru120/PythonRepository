import sys
sys.path.append('D:/02.Python/ch11/')
from House import House
from House2 import House2

house_A = House(1999, 1000, 'Seoul', 777777777)
print(house_A.address)
house_A.country = 'Korea' #동적으로 객체에 속성 추가
house_A.address = 'Busan'
print(house_A.country, house_A.address)

house_B = House(2016, 125, 'Daegu', 100000000)
print(house_B.address, house_B.year, house_A.address, house_A.year) #각각의 객체마다 다른 namespace를 가진다

house2_A = House2(1990, 100, 'Jongrogu Pyung-Chang dong', 777777777)
house2_A.show_info()
house2_A.change_price(1.5)
house2_A.show_info()

def func1(self, a):
    self.a = a

def func2(self, b):
    self.b = b

class Test():
    f1 = func1
    f2 = func2

    def show_attr(self):
        print('(a:{}, b:{})'.format(self.a, self.b))

inst = Test()
inst.f1(1)
inst.f2(2)
inst.show_attr()

func1(inst, 77) #클래스 외부의 인스턴스 함수를 직접 호출할 경우 self에 인스턴스를 전달
func2(inst, 99)        
inst.show_attr()

class Test2():
    var = 777
    def method1(self):
        print(var) #print함수내의 var변수는 전역영역에 있는 var 변수를 가리킨다

var = 123456

inst = Test2()
inst.method1()

class Test():
    def __init__(self, a, b):
        self.a = a
        self.b = b

    @classmethod #클래스 메소드 정의
    def cls_method(cls, a, b, c):
        cls.a = a
        cls.b = b
        cls.c = c

test_A = Test(1,2)
test_A.cls_method(3,4,5)
print(test_A.a, test_A.b, test_A.c)
print(Test.a, Test.b)

class Container_x():
    @staticmethod #클래스나 인스턴스를 통해 접근 가능한 메소드
    def func_first():
        print('about Algorithm')

    @staticmethod
    def func_second():
        print('about English')

    @staticmethod
    def func_third():
        print('about Python')

Container_x.func_first()
a = Container_x()
a.func_first()

mylist = []
mylist_append = getattr(mylist, 'append') #.속성 연산을 함수로 바꾸는 기능
mylist_append(10)
print(mylist)

class Container_x():
    @staticmethod
    def func_first():
        print('about Algorithm')

    @staticmethod
    def func_second():
        print('about English')

    @staticmethod
    def func_third():
        print('about Python')

list1=['first']
list2=['second']
list3=['third']

def func_selector(data): #함수 선택기
    sel_func = getattr(Container_x, 'func_{}'.format(data[0]))
    return sel_func()

func_selector(list1)
func_selector(list2)
func_selector(list3)

import sys
sys.path.append('D:/02.Python/ch11/')
from House2 import House2

house_A = House2(1999, 200, 'Seoul', 77777777777)
house_A.price = 30
house_A.show_info()

from House3 import House3
house_A = House3(1999, 200, 'Seoul', 77777777777)
house_A.__price = 30
house_A.show_info()

print(dir(house_A))

class Ticket():
    def __init__(self, distance):
        self.__distance = distance

    def get_distance(self): #getter 메소드
        return '{} m(meter) '.format(self.__distance)

    def set_distance(self, distance): #setter 메소드
        self.__distance = distance

    def get_fare(self):
        return '{} \\(Won) '.format(self.__distance*13)

person_A = Ticket(15000)        
print(person_A.get_distance())
print(person_A.get_fare())
person_A.set_distance(30000)
print(person_A.get_distance())
print(person_A.get_fare())

class Ticket():
    def __init__(self, distance):
        self.__distance = distance

    @property #프로퍼티 장식자를 통해 distance 함수를 getter로 사용
    def distance(self):
        return '{} m(meter) '.format(self.__distance)

    def set_distance(self, distance):
        self.__distance = distance

    @property
    def fare(self):
        return '{} \\(Won) '.format(self.__distance*13)

person_A = Ticket(15000)
print(person_A.distance)
print(person_A.fare)    

class Ticket():
    def __init__(self, distance):
        self.__distance = distance

    @property
    def distance(self):
        return '{} m(meter) '.format(self.__distance)

    @distance.setter #property의 쓰기 속성 설정
    def distance(self, distance):
        self.__distance = distance

    @property
    def fare(self):
        return '{} \\(Won) '.format(self.__distance*13)

person_A = Ticket(15000)
print(person_A.distance)
person_A.distance = 20000
print(person_A.distance)

class Parent():
    def __init__(self, money):
        self.money = money

    def show_money(self):
        print(self.money)        

class Child(Parent):
    pass

son = Child(250)
son.show_money()

class Account():
    def __init__(self, money):
        self.balance = money

    def deposit(self, money):
        self.balance += money

    def withdraw(self, money):
        self.balance -= money

    def show_Account(self):
        print('balance : {}원'.format(self.balance))

my_account = Account(100)
my_account.show_Account()
my_account.deposit(200)    
my_account.show_Account()
my_account.withdraw(150)
my_account.show_Account()

class Yellow_Account(Account):
    def deposit(self, money): #오버라이딩
        self.balance += money*1.07

    def withdraw(self, money):
        self.balance -= money + 10

my_account = Yellow_Account(100)
my_account.deposit(200)
my_account.show_Account()
my_account.withdraw(150)
my_account.show_Account()

class Blue_Account(Account):
    def deposit(self, money):
        self.balance += money*1.17

    def withdraw(self, money):
        self.balance -= money+50

my_account = Blue_Account(100)        
my_account.deposit(200)
my_account.show_Account()
my_account.withdraw(150)
my_account.show_Account()

class Blue_Account(Account):
    def __init__(self, name, money):
        Account.__init__(self, money) #직접 클래스명을 사용해서 부모 클래스의 메소드를 호출
        self.name = name

    def deposit(self, money):
        self.balance += money*1.17

    def withdraw(self, money):
        self.balance -= money+50

    def show_Account(self):
        Account.show_Account(self)
        print('Account owner : {}'.format(self.name))

my_account = Blue_Account('Kisup Yun', 100)
my_account.show_Account()

class Blue_Account(Account):
    def __init__(self, name, money):
        super().__init__(money) #super 메소드를 통해 부모 클래스의 메소드 호출
        self.name = name

    def deposit(self, money):
        self.balance += money*1.17

    def withdraw(self, money):
        self.balance -= money+50

    def show_Account(self):
        super().show_Account()
        print('Account owner : {}'.format(self.name))

my_account = Blue_Account('my', 100)
my_account.deposit(200)
my_account.show_Account()

# 덕 타이핑
account1 = Yellow_Account(100)
account2 = Yellow_Account(100)
account3 = Blue_Account('Ji Kwang', 230)
account4 = Blue_Account('Chul Su', 170)
account_list = [account1, account2, account3, account4]

def check_account(account): #인수로 받는 참조변수의 타입에 상관없이 같은 이름을 가진 메소드는 호출 가능하다
    account.show_Account()

check_account(account1)
check_account(account2)
check_account(account3)
check_account(account4)

class Duck():
    def show_Account(self):
        print("It's Duck Account")

account100 = Duck()
check_account(account100)

class Dwarf():
    def __init__(self, name, item):
        self.__name = name
        self.__item = item

    def choose_item(self, item):
        print('{} Dwarf choose {}'.format(self.__name, item.name))
        self.__item = item

    def use_item(self):
        self.__item.use()

class Sword():
    def __init__(self, name, power):
        self.__name = name + ' Sword'
        self.__power = power

    def use(self):
        print('use {} **power : {}**'.format(self.__name, self.__power))

    @property
    def name(self):
        return self.__name

my_Dwarf = Dwarf('kisup', None)
silver_sword = Sword('Silver', 10)
my_Dwarf.choose_item(silver_sword)
my_Dwarf.use_item()

class Hammer():
    def __init__(self, name, power):
        self.__name = name + ' Hammer'
        self.__power = power

    def use(self):
        print('use {} **power : {}**'.format(self.__name, self.__power))

    @property
    def name(self):
        return self.__name

silver_hammer = Hammer('silver', 12)
my_Dwarf.choose_item(silver_hammer)
my_Dwarf.use_item()

class Food():
    def __init__(self, name, energy):
        self.__name = name
        self.__energy = energy

    def use(self):
        print('eat {} **energy : {}**'.format(self.__name, self.__energy))

    @property
    def name(self):
        return self.__name

hamburger = Food('Hamburger', 50)
my_Dwarf.choose_item(hamburger)
my_Dwarf.use_item()        

class Dwarf():
    def __init__(self, name, item):
        self.__name = name
        self.__item = item

    def choose_item(self, item):
        print('{} Dwarf choose {}'.format(self.__name, item.name))
        self.__item = item

    def use_item(self):
        self.__item.use()

    def __add__(self, item):
        print('{} Dwarf choose {}'.format(self.__name, item.name))
        self.__item = item

    def __pos__(self):
        self.__item.use()        

class Sword():
    def __init__(self, name, power):
        self.__name = name+' Sword'
        self.__power = power

    def use(self):
        print('use {} **power : {}**'.format(self.__name, self.__power))

    @property
    def name(self):
        return self.__name

silver_sword = Sword('Silver', 10)
my_Dwarf = Dwarf('Kisup', None)
my_Dwarf + silver_sword
my_Dwarf.use_item()
+my_Dwarf

class Dwarf():
    def __init__(self, name, item):
        self.__name = name
        self.__item = item

    def choose_item(self, item):
        print('{} Dwarf choose {}'.format(self.__name, item.name))
        self.__item = item

    def use_item(self):
        self.__item.use()

    def __add__(self, item):
        print('{} Dwarf choose {}'.format(self.__name, item.name))
        self.__item = item

    def __radd__(self, item):
        print('{} Dwarf choose {}'.format(self.__name, item.name))
        self.__item = item

class Sword():
    def __init__(self, name, power):
        self.__name = name
        self.__power = power

    def use(self):
        print('use {} **power : {}**'.format(self.__name, self.__power))

    @property
    def name(self):
        return self.__name

my_Dwarf = Dwarf('Kisup', None)
silver_sword = Sword('Silver', 10)
my_Dwarf + silver_sword
silver_sword + my_Dwarf

# 직접해 봅시다
# 1-1. 파이썬 고등학교에서는 학생에 대한 클래스를 만들어 학생들을 관리하려고 한다.
#      예를 들어 학생의 이름, 나이, 학년 등이 객체의 속성이 될 수 있을 것이다.
#      한 번 학생 클래스를 만들어 보자.
class Student():
    def __init__(self, name, age, grade):
        self.__name = name
        self.__age = age
        self.__grade = grade

# 1-2. 학생 클래스의 클래스 속성은 어떤 것으로 하면 될지 생각해보고 클래스 속성을 추가하도록 하자.
class Student():
    school_name = '서울 고등학교'

    def __init__(self, name, age, grade):
        self.__name = name
        self.__age = age
        self.__grade = grade

# 1-3. 교육청에서는 학교마다 클래스로 관리해오던 학생에 대한 정보를 통합하여 관리하려고 한다.
#      그런데 학교마다 만들어 놓은 클래스의 형태가 천차만별이기 때문에 관리하는 것이 쉽지 않을 듯 보인다.
#      이런 문제로 교육청에서는 학생 클래스가 다음에 제시되는 클래스를 반드시 상속하도록 지침을 내렸다.
class Student():
    def __init__(self, name, grade, s_class, number, score, etc):
        self.name = name
        self.grade = grade
        self.s_class = s_class
        self.number = number
        self.score = score

    def up_grade(self):
        pass

    def down_grade(self):
        pass

    def show_info(self):
        pass

class PythonStudent(Student):
    school_name = '서울 고등학교'

    def up_grade(self):
        if self.grade != 'graduation':
            self.grade += 1    
            if self.grade > 3:
                self.grade = 'graduation'

    def down_grade(self):
        if self.grade != 'graduation' and self.grade != 1:
            self.grade -= 1                        

    def show_info(self):
        print('{}, {}, {}, {}, {}, {}'.format(PythonStudent.school_name, self.grade, self.s_class, self.number, self.name, self.score))

# 1-4. Student 클래스를 상속받는 클래스에 학교 이름과 교장 이름에 대한 클래스 속성을 추가해보자.
#      그리고 교장이 바뀔 경우 교장 이름을 바꾸는 클래스 메소드도 정의해보자.
class PythonStudent(Student):
    school_name = '서울 고등학교'
    principal_name = '길용현'

    def up_grade(self):
        if self.grade != 'graduation':
            self.grade += 1    
            if self.grade > 3:
                self.grade = 'graduation'

    def down_grade(self):
        if self.grade != 'graduation' and self.grade != 1:
            self.grade -= 1

    @classmethod
    def change_principal(cls, principal_name):
        PythonStudent.principal_name = principal_name                                

    def show_info(self):
        print('{}, {}, {}, {}, {}, {}'.format(PythonStudent.school_name, self.grade, self.s_class, self.number, self.name, self.score))        

# 1-5. 프로퍼티를 사용하여 인스턴스 속성 name을 읽기전용으로 바꿔보자.
class Student():
    def __init__(self, name, grade, s_class, number, score):
        self.__name = name
        self.grade = grade
        self.s_class = s_class
        self.number = number
        self.score = score

    def up_grade(self):
        pass

    def down_grade(self):
        pass

    def show_info(self):
        pass

class PythonStudent(Student):
    school_name = '서울 고등학교'
    principal_name = '길용현'

    def __init__(self, name, grade, s_class, number, score):
        self.__name = name
        self.grade = grade
        self.s_class = s_class
        self.number = number
        self.score = score

    def up_grade(self):
        if self.grade != 'graduation':
            self.grade += 1    
            if self.grade > 3:
                self.grade = 'graduation'

    def down_grade(self):
        if self.grade != 'graduation' and self.grade != 1:
            self.grade -= 1

    @classmethod
    def change_principal(cls, principal_name):
        PythonStudent.principal_name = principal_name                                

    @property
    def name(self):
        return self.__name

    def show_info(self):
        print('{}, {}, {}, {}, {}, {}'.format(PythonStudent.school_name, self.grade, self.s_class, self.number, self.name, self.score))             


student1 = PythonStudent('홍길동', 3, 1, 32, 99)
student1.show_info()
print(student1.name)
student1.down_grade()
student1.show_info()
student1.up_grade()
student1.show_info()
PythonStudent.change_principal('안철수')
print(PythonStudent.principal_name)

# 2. 리스트를 상속받는 클래스를 만들고 연산자 오버로딩을 사용하여 리스트 간에 빼기 연산을 정의해보자.
#    빼기 연산의 결과는 set 타입의 빼기 연산(차집합)과 동일하도록 만들어 보자.
class Data(list):
    def __init__(self, item):
        self.item = item

    def __sub__(self, item):
        for x in item:
            if x in self.item:
                self.item.remove(x)

    def show_info(self):
        print(self.item)

oldData = [1,2,5,6]
newData = [4,5]
newList = Data(oldData)
newList - newData
newList.show_info()

# 3. 덕 타이핑에서 소개한 예제를 참고하여 Elf라는 클래스를 만든 후 Sword, Hammer 클래스의 객체를 사용을 해보자.
#    그 다음 Elf가 사용할 수 있도록 Car 라는 클래스도 정의한 후 사용해보자.
class Elf():
    def __init__(self, name, item):
        self.__name = name
        self.__item = item

    def choose_item(self, item):
        print('{} Elf choose {}'.format(self.__name, item.name))
        self.__item = item

    def use_item(self):
        self.__item.use()

class Sword():
    def __init__(self, name, power):
        self.__name = name + ' Sword'
        self.__power = power

    def use(self):
        print('use {} **power : {}**'.format(self.__name, self.__power))

    @property
    def name(self):
        return self.__name

class Hammer():
    def __init__(self, name, power):
        self.__name = name + ' Hammer'
        self.__power = power

    def use(self):
        print('use {} **power : {}**'.format(self.__name, self.__power))

    @property
    def name(self):
        return self.__name

class Car():
    def __init__(self, name):
        self.__name = name

    def use(self):
        print('use {}'.format(self.__name))                    

    @property
    def name(self):
        return self.__name

elf = Elf('엘프', None)        
car = Car('람보르기니')
elf.choose_item(car)
elf.use_item()