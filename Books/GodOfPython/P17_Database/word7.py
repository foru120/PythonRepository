import sqlite3

conn = sqlite3.connect('mydb.db')
c = conn.cursor()
c.execute('''select * from mytable''')
print(c.fetchone())
print('----------')
print(c.fetchone())
print('----------')

c.close()
conn.close()