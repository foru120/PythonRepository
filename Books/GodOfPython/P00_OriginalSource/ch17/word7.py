#word7.py
import sqlite3

conn = sqlite3.connect('mydb.db')
c = conn.cursor()
c.execute('''SELECT * FROM mytable''')		#테이블 전체를 선택.
print(c.fetchone())				#다음 행 출력
print('----------')
print(c.fetchone())				#다음 행 출력
print('----------')

c.close()
conn.close()
