class House(object):
    Company = 'Python Factory' #클래스 속성

    def __init__(self, year, acreages, address, price):
        self.year = year #인스턴스 속성
        self.acreages = acreages
        self.address = address
        self.price = price

    def show_Company(self):
        print(House.Company) #클래스 명을 앞에 기술하지 않으면 전역 영역의 namespace를 검사