import sqlite3

conn = sqlite3.connect('mydb.db')
c = conn.cursor()
c.execute('''select * from mytable''')
print(c.fetchone())
print('----------')
print(c.fetchmany(3))
print('----------')
print(c.fetchall())

c.close()
conn.close()