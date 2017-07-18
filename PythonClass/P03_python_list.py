from PythonClass.common_func import CommonFunc

a = ['김', '이', '최']
print(type(a))

# ■ 리스트 변수의 유용한 메소드 함수
#  1. append()  : 리스트 끝에 새 요소를 추가
#  2. extend()  : 기존 리스트에 다른 리스트를 이어 붙임
#  3. insert()  : 리스트의 특정 위치에 새로운 요소를 입력
#  4. remove()  : 리스트의 요소 삭제
#  5. pop()     : 리스트의 마지막 요소를 제거
#  6. index()   : 리스트의 특정 요소의 위치를 출력
#  7. count()   : 리스트내의 요소의 건수를 출력
#  8. sort()    : 리스트의 요소를 정렬
#  9. reverse() : 리스트의 요소 순서를 반대로 출력

print('')
print('====================================================================================================')
print('== 문제 71. emp_list 라는 비어있는 리스트 변수를 선언하고 input 명령어를 이용해서 리스트 변수에 추가할 요소를')
print('== 물어보게해서 요소가 추가되게 하는 코드를 구현하시오!')
print('====================================================================================================')
emp_list = []
emp_list.append(input('emp_list 에 추가할 요소를 입력하세요 : '))
print(emp_list)


print('')
print('====================================================================================================')
print('== 문제 72. emp_list 에 추가된 요소를 삭제하는 코드를 구현하시오!')
print('====================================================================================================')
try:
    emp_list.remove(input('emp_list 에 삭제할 요소를 입력하세요 : '))
    print(emp_list)
except ValueError:
    print('삭제할 요소가 존재하지 않습니다.')


print('')
print('====================================================================================================')
print('== 문제 73. emp_list 에 요소를 추가하고 삭제하고 갯수를 확인하는 코드를 구현하는데 아래와 같이 실행되게 하시오!')
print('====================================================================================================')
emp_list = []

while True:
    num = input('(추가 = 1, 삭제 = 2, 개수 확인 = 3) : ')

    if num == '1':
        emp_list.append(input('emp_list 에 추가할 요소를 입력하세요 : '))
        print(emp_list)
    elif num == '2':
        try:
            emp_list.remove(input('emp_list 에 삭제할 요소를 입력하세요 : '))
            print(emp_list)
        except ValueError:
            print('삭제할 요소가 존재하지 않습니다.')
    elif num == '3':
        print(emp_list.count(input('emp_list 개수를 구할 요소를 입력하세요 : ')))
    else:
        print('프로그램을 종료합니다.')
        break


print('')
print('====================================================================================================')
print('== 문제 74. (점심시간 문제) 리스트 메소드 중에 sort 를 이용해서 월급을 출력할때 높은 것부터 출력될 수 있도록 하시오!')
print('====================================================================================================')
emp_list = []
for empData in CommonFunc.returnCsvData('emp2.csv'):
    emp_list.append(empData)

for empData in sorted(emp_list, key=lambda x: int(x[5]), reverse=True):  # lambda 가 리턴하는 것은 해당 리스트의 값
    print(empData[1], empData[5])


print('')
print('====================================================================================================')
print('== 문제 76. 직업이 SALESMAN 인 사원들의 이름과 입사일과 직업을 출력하는데 가장 최근에 입사한 사원부터 출력하시오!')
print('====================================================================================================')
emp_list = []
for empData in CommonFunc.returnCsvData('emp2.csv'):
    emp_list.append(empData)

for empData in sorted(emp_list, key=lambda emp: emp[4]):
    print(empData[1], empData[2], empData[4])


print('')
print('====================================================================================================')
print('== 문제 77. emp list 에서 최대 월급을 출력하시오!')
print('====================================================================================================')
sal_list = []
max_sal = 0
for empData in CommonFunc.returnCsvData('emp2.csv'):
    sal_list.append(int(empData[5]))

print(max(sal_list), min(sal_list))


print('')
print('====================================================================================================')
print('== 문제 78. emp list 에서 평균값을 출력하시오!')
print('====================================================================================================')
sal_list = []
for empData in CommonFunc.returnCsvData('emp2.csv'):
    sal_list.append(int(empData[5]))

print(round(sum(sal_list)/len(sal_list), 2))


print('')
print('====================================================================================================')
print('== 문제 79. emp list 에서 직업이 SALEMAN 인 사원들 중에서의 최대 월급을 출력하시오!')
print('====================================================================================================')
sal_list = []
for empData in CommonFunc.returnCsvData('emp2.csv'):
    if empData[2] == 'SALESMAN':
        sal_list.append(int(empData[5]))

print(max(sal_list))