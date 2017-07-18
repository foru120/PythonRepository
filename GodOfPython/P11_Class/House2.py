class House2(object):
    Company = 'Python Factory'

    def __init__(self, year, acreages, address, price):
        self.year = year
        self.acreages = acreages
        self.address = address
        self.price = price

    def show_Company(self):
        print(House2.Company)

    def change_price(self, rate):
        self.price = self.price*rate

    def show_info(self):
        print("""This houes was built by {} in {},
            acreages:{},
            address:{},
            price:{}""".format(House2.Company, self.year, self.acreages, self.address, self.price))