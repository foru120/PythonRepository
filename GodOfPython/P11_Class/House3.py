class House3(object): #object는 상속받을 클래스
    Company = 'Python Factory' #클래스 변수

    def __init__(self, year, acreages, address, price): #생성자
        self.__year = year #private 접근 지정자
        self.__acreages = year
        self.__address = address
        self.__price = price

    def show_Company(self): #인스턴스 메서드
        print(House3.Company)

    def change_price(self, rate):
        self.__price = self.__price*rate

    def show_info(self):
        print('''This houes was built by {} in {},
            acreages:{},
            address:{},
            price:{}'''.format(House3.Company, self.__year, self.__acreages, self.__address, self.__price))