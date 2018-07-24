import sqlite3

conn = sqlite3.connect('Score.db')
c = conn.cursor()

person1 = ('Jhon', 2, 15483, 'America')
person2 = ('Kim', 2, 12547, 'Korea')
person3 = ('Wang', 3, 21557, 'China')
person4 = ('Lee', 4, 35466, 'Korea')
person5 = ('Yamada', 1, 9531, 'Japan')

data = (person1, person2, person3, person4, person5)

c.execute('create table score(name varchar, last_stage int, score int, nation varchar)')
c.executemany('insert into score values(?,?,?,?)', (data))
c.close()

conn.commit()
conn.close()