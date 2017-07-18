# Error9.py
class Myexcept(Exception):

    def __init__(self, num, data):  # 두 개의 인수(num, data)를 받음
        self.args = (num, data)
        self.num = num
        self.data = data

    def __str__(self):
        # 예외 정보
        return "{} is greater than {}".format(self.args[1], self.args[0])

if __name__ == "__main__":
    while True:
        try:
            data = input("?>")
            if int(data) > 100:
                raise Myexcept(100, data)  # 예외 인스턴스 생성
            print(data)
        except Myexcept as e:
            print("exception", e)
