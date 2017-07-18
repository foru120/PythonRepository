#word8.py
import sqlite3

conn = sqlite3.connect('mydb.db')
c = conn.cursor()
c.execute('''SELECT * FROM mytable''')	#테이블 전체를 선택
print(c.fetchone())			#다음 행 출력
print('----------')
print(c.fetchmany(3))		#다음 3개의 행 출력
print('----------')
print(c.fetchall())			#다음부터 마지막까지의 출력

c.close()
conn.close()
