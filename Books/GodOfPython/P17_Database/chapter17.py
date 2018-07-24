from sqlite3 import *
# mydb = connect('D:/02.Python/ch17/fruit.db')
# csr = mydb.cursor()
# csr.execute('create table test(fruit varchar(20), num int, price int)')
# csr.execute("insert into test(fruit, num, price) values('Apple', 10, 1000)")
# csr.execute('select * from test')
# row = csr.fetchone()
# print(row)
# mydb.commit()
# mydb.close()

# conn = connect('C:/sqlite/mydb.db')
# csr = conn.cursor()
# csr.execute('select * from mytable')
# print(csr.fetchone())

# conn = connect('D:/02.Python/ch17/mydb.db')
# c = conn.cursor()
# c.execute('''create table mytable(word text, meaning text, level integer)''')
# c.close()

# conn = connect('D:/02.Python/ch17/mydb.db')
# c = conn.cursor()
# c.execute('''insert into mytable(word, meaning, level)
#     values("python", "A python is a large snake that kills animals by squeezing them with its body.", 2)''')
# c.execute('''insert into mytable(word, meaning, level)
#     values("sql", "structured query language: a computer programing language used for database management.", 1)''')
# c.execute('''insert into mytable(word, meaning, level)
#     values("apple", "An apple is a round fruit with smooth green, yellow, or red skin and firm write flesh.", 1)''')
# c.execute('''select * from mytable''')
# print(c.fetchall())

# c.close()
# conn.commit()
# conn.close()

# conn = connect('D:/02.Python/ch17/mydb.db')
# c = conn.cursor()

# word = input('word : ')
# meaning = input('meaning : ')
# level = input('level : ')
# c.execute('insert into mytable values(?,?,?)', (word, meaning, level))
# c.execute('select * from mytable')
# print(c.fetchall())

# c.close()
# conn.commit()
# conn.close()

# conn = connect('D:/02.Python/ch17/mydb.db')
# c = conn.cursor()

# record1 = ('test_word1', 'test_word1_meaning', 1)
# record2 = ('test_word2', 'test_word2_meaning', 2)
# record3 = ('test_word3', 'test_word3_meaning', 3)
# record4 = ('test_word4', 'test_word4_meaning', 4)

# all_record = (record1, record2, record3, record4)

# c.executemany('''insert into mytable values(?,?,?)''', all_record)
# c.execute('''select * from mytable''')
# print(c.fetchall())

# c.close()
# conn.commit()
# conn.close()

conn = connect('yourdb.db')
c = conn.cursor()

c.executescript('''create table yourtable(word, meaning, level, time);
    insert into yourtable values('test_workd0', 'test_meaning0', 1, datetime('now'));
    insert into yourtable values('test_workd1', 'test_meaning1', 2, datetime('now'));''')

c.close()
conn.commit()
conn.close()