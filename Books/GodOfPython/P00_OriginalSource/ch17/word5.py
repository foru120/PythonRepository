#word5.py
import sqlite3

conn = sqlite3.connect('C:/gop/ch17/yourdb.db')
c = conn.cursor()

#c.execute('''CREATE TABLE yourtable(word, meaning, level, time)''')
#c.execute('''INSERT INTO yourtable VALUES('test_workd0', 'test_meaning0', 1, datetime('now'))''')
#c.execute('''INSERT INTO yourtable VALUES('test_workd1', 'test_meaning1', 2, datetime('now'))''')

c.executescript('''CREATE TABLE yourtable(word, meaning, level, time);
    INSERT INTO yourtable VALUES('test_workd0', 'test_meaning0', 1, datetime('now'));
    INSERT INTO yourtable VALUES('test_workd1', 'test_meaning1', 2, datetime('now'));
    ''')

c.close()
conn.commit()
conn.close()
