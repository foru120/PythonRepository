# C:\gop\ch11\House.py
class House(object):  # House 클래스 정의

    def __init__(self, year, acreages, address, price):
        self.year = year
        self.acreages = acreages
        self.address = address
        self.price = price

    def change_price(self, rate):
        self.price = self.price * rate

    def show_info(self):
        print("This houes is built in {},   \
        acreages : {},  \
        address : {},   \
        price : {} "
              .format(self.year, self.acreages, self.address, self.price))
if __name__ == "__main__":
    house_A = House(1999, 100, "seoul", 777777777)  # 객체 house_A 생성
    house_A.show_info()  # 객체를 통한 메소드 사용


class House2(object):  # House2 클래스 정의
    Company = "Python Factory"  # 클래스 속성

    def __init__(self, year, acreages, address, price):
        self.year = year
        self.acreages = acreages
        self.address = address
        self.price = price

    def show_Company(self):
        print(House2.Company)

    def change_price(self, rate):
        self.price = self.price * rate

    def show_info(self):
        print("This houes was built by {} in {},    \
            acreages : {},  \
            address : {},   \
            price : {} "
              .format(House2.Company, self.year, self.acreages, self.address, self.price))


class House3(object):  # House2 클래스 정의
    Company = "Python Factory"  # 클래스 속성

    def __init__(self, year, acreages, address, price):
        self.__year = year
        self.__acreages = acreages
        self.__address = address
        self.__price = price

    def show_Company(self):
        print(House3.Company)

    def change_price(self, rate):
        self.__price = self.__price * rate

    def show_info(self):
        print("This houes was built by {} in {},    \
            acreages : {},  \
            address : {},   \
            price : {} "
              .format(House3.Company, self.__year, self.__acreages, self.__address, self.__price))
