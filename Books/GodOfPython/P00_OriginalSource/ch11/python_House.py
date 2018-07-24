# python_House.py

# Company = "Python Factory"     #전역변수 Company


class House(object):
    #Company = "Python Factory"     #클래스 속성

    def __init__(self, year, acreages, address, price):
        self.year = year
        self.acreages = acreages
        self.address = address
        self.price = price
        self.Company = "Python Factory"  # Company 인스턴스 속성 추가
