#word1.py
import sqlite3

conn = sqlite3.connect('C:/gop/ch17/mydb.db')   #해당 경로의mydb.db를 읽는다. 없다면 새로 생성.
c = conn.cursor()
c.execute('''CREATE TABLE mytable(word TEXT, meaning TEXT, level INTEGER)''')
conn.close()
