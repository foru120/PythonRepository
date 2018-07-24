# gui_05.py
import tkinter
import random


def randxy():
    a = random.randint(1, 250)
    return a


def click():
    cvs.create_line(randxy(), randxy(), randxy(), randxy())

root = tkinter.Tk()
cvs = tkinter.Canvas(root, width=250, height=250)
cvs.pack()

button = tkinter.Button(text='click', command=click)  # 방법 1
cvs.create_window(150, 200, window=button)

label = tkinter.Label(text="Hellow")  # 방법 2
label_w = cvs.create_window(100, 200)
cvs.itemconfigure(label_w, window=label)

root.mainloop()
