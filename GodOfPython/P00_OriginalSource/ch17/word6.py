#word6.py
import sqlite3

conn = sqlite3.connect('C:/gop/ch17/yourdb.db')
c = conn.cursor()
c.execute('''SELECT * FROM yourtable''')
print(c.fetchall())
conn.close()
