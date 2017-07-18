class Myexcept(Exception):
    def __init__(self, num, data):
        self.args = (num, data)
        self.num = num
        self.data = data

    def __str__(self): #해당 클래스에 대한 객체 출력시 수행
        return '{} is greater than {}'.format(self.args[1], self.args[0])

if __name__=='__main__':
    while True:
        try:
            data = input('?>')
            if int(data) > 100:
                raise Myexcept(100, data)
            print(data)
        except Myexcept as e:
            print('exception', e)