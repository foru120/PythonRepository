import sqlite3

conn = sqlite3.connect('Score.db')
c = conn.cursor()
c.execute('select * from score order by score desc')

for data in c.fetchall():
    print(data)

c.close()
conn.close()
