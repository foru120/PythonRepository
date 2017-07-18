import tkinter
import sqlite3

conn = sqlite3.connect('mydb.db')
c = conn.cursor()
c.execute('''select * from mytable''')

root = tkinter.Tk()
var_word = tkinter.StringVar(root, value='1')
var_meaning = tkinter.StringVar(root, value='2')
var_level = tkinter.StringVar(root, value=0)

def next_word():
    record = c.fetchone()
    if not record:
        return 0
    var_word.set(record[0])
    var_meaning.set(record[1])
    var_level.set(record[2])

frame = tkinter.Frame(root, height=100, width=300)
frame.pack()
label_title = tkinter.Label(frame, text='- 영어 단어장 -')
label_title.grid(row=0, column=0, columnspan=2)
label_word = tkinter.Label(frame, text='단어')
label_meaning = tkinter.Label(frame, text='의미')
label_level = tkinter.Label(frame, text='난이도')
label_word.grid(row=1, column=0, sticky=tkinter.W)
label_meaning.grid(row=2, column=0, sticky=tkinter.W)
label_level.grid(row=3, column=0, sticky=tkinter.W)

button = tkinter.Button(frame, text='next', command=next_word)
button.grid(row=4, column=0, columnspan=2)

word = tkinter.Label(frame, textvariable=var_word, width=50, anchor='w')
meaning = tkinter.Label(frame, textvariable=var_meaning, wraplength=200, justify='left', height=5)
level = tkinter.Label(frame, textvariable=var_level)

word.grid(row=1, column=1, sticky=tkinter.W)
meaning.grid(row=2, column=1, sticky=tkinter.W)
level.grid(row=3, column=1, sticky=tkinter.W)

next_word()
root.mainloop()