import pickle

class Member():
    def __init__(self, name, age, gender):
        self.__name = name
        self.__age = age
        self.__gender = gender

    @property
    def name(self):
        return self.__name

    @property
    def age(self):
        return self.__age

    @property
    def gender(self):
        return self.__gender

mem1 = Member('길용현', 30, '남')
mem2 = Member('홍길동', 24, '남')

f = open('D:/02.Python/ch12/direct/num4.txt', 'wb')        
pickle.dump(mem1, f) #순서대로 저장
pickle.dump(mem2, f)
f.close()

f = open('D:/02.Python/ch12/direct/num4.txt', 'rb')
mem1 = pickle.load(f) #순서대로 로드
mem2 = pickle.load(f)
print(mem1.name, mem1.age, mem1.gender)
print(mem2.name, mem2.age, mem2.gender)