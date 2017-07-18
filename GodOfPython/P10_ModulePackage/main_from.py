from GodOfPython.P10_ModulePackage.sample import sample_func #from 절에 모듈명, import 절에 함수

x=222
def main_func():
    print('x is', x)

sample_func() #from 절을 사용해 함수를 불러오면 모듈명을 기술하지 않고 사용 가능
main_func()