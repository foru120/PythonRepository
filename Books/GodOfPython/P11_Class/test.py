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
        pass


student1 = PythonStudent('홍길동', 3, 1, 32, 99)
student1.show_info()
print(dir(student1))