# main_from.py
from sample import sample_func  # (from) 모듈 이름으로부터 (import) 함수를 직접 불러들임
x = 222


def main_func():
    print('x is', x)

sample_func()  # 모듈 이름을 명시하지 않고 직접 사용할 수 있다.
main_func()
