class House(object): #괄호안은 상속받는 클래스
    def __init__(self, year, acreages, address, price): #생성자
        self.year = year
        self.acreages = acreages
        self.address = address
        self.price = price

    def change_price(self, rate):
        self.price = self.price*rate

    def show_info(self): #각 메소드에 self(객체 자신을 지칭) 키워드를 첫 번째 인수로 지정해야 함
        print("""This houes is built in {},
            acreages:{},
            address:{},
            price:{}""".format(self.year,self.acreages,self.address,self.price))

if __name__=='__main__':
    houes_A = House(1990, 100, 'seoul', 77777777)
    houes_A.show_info()