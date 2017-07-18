#word4.py
import sqlite3

conn = sqlite3.connect('mydb.db')
c = conn.cursor()

record1 = ('test_word1', 'test_word1_meaning', 1)
record2 = ('test_word2', 'test_word2_meaning', 2)
record3 = ('test_word3', 'test_word3_meaning', 3)
record4 = ('test_word4', 'test_word4_meaning', 4)

all_record = (record1,record2,record3,record4)

c.executemany('''INSERT INTO mytable VALUES(?,?,?)''', all_record)
c.execute('''SELECT * FROM mytable''')
print(c.fetchall())
c.close()
conn.commit()
conn.close()
