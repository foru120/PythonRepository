print('')
print('====================================================================================================')
print('== 문제 134. 아래와 같이 딕셔너리 형태의 데이터를 만들고 출력하시오!')
print('====================================================================================================')
emp_dic = {'mgr': '7788', 'sal': '1100', 'deptno': '20', 'comm': '0', 'job': 'CLERK', 'hiredate': '1983-01-15', 'empno': '7876', 'ename': 'ADAMS'}
print(emp_dic['mgr'])


print('')
print('====================================================================================================')
print('== 문제 135. 6장에서 배운 for loop 를 이용해서 emp2.csv 를 읽어와서 emp_dic 이라는 딕셔너리 데이터 유형을 만드시오.')
print('====================================================================================================')
from PythonClass.common_func import CommonFunc
emp_col = ['empno', 'ename', 'job', 'mgr', 'hiredate', 'sal', 'comm', 'deptno']
emp_dic = []

for empData in CommonFunc.returnCsvData('emp2.csv'):
    temp = {}
    for i in range(len(empData)):
        temp[emp_col[i]] = empData[i]
    emp_dic.append(temp)

print(emp_dic)


print('')
print('====================================================================================================')
print('== 문제 136. emp 딕셔너리 변수에서 이름만 출력하시오!')
print('====================================================================================================')
from PythonClass.common_func import CommonFunc
emp_col = ['empno', 'ename', 'job', 'mgr', 'hiredate', 'sal', 'comm', 'deptno']
emp_dic = []

for empData in CommonFunc.returnCsvData('emp2.csv'):
    temp = {}
    for i in range(len(empData)):
        temp[emp_col[i]] = empData[i]
    emp_dic.append(temp)

for emp in emp_dic:
    print(emp['ename'])


print('')
print('====================================================================================================')
print('== 문제 138. dept.csv 를 읽어서 딕셔너리로 저장하고 아래와 같이 수행하면 deptno, dname, loc 가 출력되게 하시오.')
print('====================================================================================================')
from PythonClass.common_func import CommonFunc
dept_col = ['deptno', 'dname', 'loc']
dept_dic = []

for deptData in CommonFunc.returnCsvData('dept.csv'):
    temp = {}
    for i in range(len(deptData)):
        temp[dept_col[i]] = deptData[i]
    dept_dic.append(temp)

for dept in dept_dic:
    print(dept['deptno'], dept['dname'], dept['loc'])


print('')
print('====================================================================================================')
print('== 문제 139. emp.csv 와 dept.csv 를 각각 읽어와서 emp_dic, dept_dic 딕셔너리 자료형으로 만드는 스크립트를 하나로 합치시오.')
print('====================================================================================================')
from PythonClass.common_func import CommonFunc

dept_col = ['deptno', 'dname', 'loc']
dept_list = []

emp_col = ['empno', 'ename', 'job', 'mgr', 'hiredate', 'sal', 'comm', 'deptno']
emp_list = []

for deptData in CommonFunc.returnCsvData('dept.csv'):
    temp = {}
    for i in range(len(deptData)):
        temp[dept_col[i]] = deptData[i]
    dept_list.append(temp)

for empData in CommonFunc.returnCsvData('emp2.csv'):
    temp = {}
    for i in range(len(empData)):
        temp[emp_col[i]] = empData[i]
    emp_list.append(temp)


print('')
print('====================================================================================================')
print('== 문제 140. emp 와 dept 라는 딕셔너리 자료구조를 만드는 스크립트와 중첩 for loop 문을 이용해서 emp 와 dept 를')
print('==  조인시켜서 ename 과 loc 를 출력하시오!')
print('====================================================================================================')
from PythonClass.common_func import CommonFunc

dept_col = ['deptno', 'dname', 'loc']
dept_list = []

emp_col = ['empno', 'ename', 'job', 'mgr', 'hiredate', 'sal', 'comm', 'deptno']
emp_list = []

for deptData in CommonFunc.returnCsvData('dept.csv'):
    temp = {}
    for i in range(len(deptData)):
        temp[dept_col[i]] = deptData[i]
    dept_list.append(temp)

for empData in CommonFunc.returnCsvData('emp2.csv'):
    temp = {}
    for i in range(len(empData)):
        temp[emp_col[i]] = empData[i]
    emp_list.append(temp)

for empData in emp_list:
    for deptData in dept_list:
        if empData['deptno'] == deptData['deptno']:
            print(empData['ename'], deptData['loc'])


print('')
print('====================================================================================================')
print('== 문제 141. 부서위치가 DALLAS 인 사원들의 이름과 부서위치를 출력하시오!')
print('====================================================================================================')
for deptData in dept_list:
    if deptData['loc'] == 'DALLAS':
        for empData in emp_list:
            if deptData['deptno'] == empData['deptno']:
                print(empData['ename'], deptData['loc'])


print('')
print('====================================================================================================')
print('== 문제 142. 위의 스크립트를 이용해서 조인 함수를 생성하시오.')
print('====================================================================================================')
from PythonClass.common_func import CommonFunc

dept_col = ['deptno', 'dname', 'loc']
dept_list = []

emp_col = ['empno', 'ename', 'job', 'mgr', 'hiredate', 'sal', 'comm', 'deptno']
emp_list = []

for deptData in CommonFunc.returnCsvData('dept.csv'):
    temp = {}
    for i in range(len(deptData)):
        temp[dept_col[i]] = deptData[i]
    dept_list.append(temp)

for empData in CommonFunc.returnCsvData('emp2.csv'):
    temp = {}
    for i in range(len(empData)):
        temp[emp_col[i]] = empData[i]
    emp_list.append(temp)

def tab_join(tab1, tab2, join_col):
    result_list = []

    for tabData1 in tab1:
        for tabData2 in tab2:
            if tabData1[join_col] == tabData2[join_col]:
                tabData1.update(tabData2)
                result_list.append(tabData1)

    return result_list

print(len(tab_join(emp_list, dept_list, 'deptno')))


print('')
print('====================================================================================================')
print('== 문제 143. pandas 를 이용해서 ename 과 loc 를 출력하시오!')
print('====================================================================================================')
import pandas as pd
empFrame = pd.read_csv('D:\\KYH\\02.PYTHON\\data\\emp.csv')
deptFrame = pd.read_csv('D:\\KYH\\02.PYTHON\\data\\deptPD.csv')
result = pd.merge(empFrame, deptFrame, on='deptno')
print(result[['ename', 'loc']])


print('')
print('====================================================================================================')
print('== 문제 144. 부서위치가 DALLAS 인 사원들의 이름과 부서위치를 출력하시오.')
print('====================================================================================================')
import pandas as pd
empFrame = pd.read_csv('D:\\KYH\\02.PYTHON\\data\\emp.csv')
deptFrame = pd.read_csv('D:\\KYH\\02.PYTHON\\data\\deptPD.csv')
result = pd.merge(empFrame, deptFrame, on='deptno')
print(result[['ename', 'loc']][result['loc'] == 'DALLAS'])


print('')
print('====================================================================================================')
print('== 문제 145. 이름과 부서위치를 출력하는데 아래와 같이 Outer Join 을 구현하시오.')
print('====================================================================================================')
import pandas as pd
empFrame = pd.read_csv('D:\\KYH\\02.PYTHON\\data\\emp.csv')
deptFrame = pd.read_csv('D:\\KYH\\02.PYTHON\\data\\deptPD.csv')
result = pd.merge(empFrame, deptFrame, on='deptno', how='right')
print(result[['ename', 'loc']])