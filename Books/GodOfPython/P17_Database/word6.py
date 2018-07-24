import sqlite3

conn = sqlite3.connect('yourdb.db')
c = conn.cursor()
c.execute('''select * from yourtable''')
print(c.fetchall())
conn.close()