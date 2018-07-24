# gui_18.py
import tkinter
from tkinter import messagebox

root = tkinter.Tk()
menubar = tkinter.Menu(root)
root['menu'] = menubar

menu_file = tkinter.Menu(menubar, tearoff=0)
menu_edit = tkinter.Menu(menubar, tearoff=0)
menu_help = tkinter.Menu(menubar, tearoff=0)

menubar.add_cascade(menu=menu_file, label="File")
menubar.add_cascade(menu=menu_edit, label="Edit")
menubar.add_cascade(menu=menu_help, label="Help")

root.mainloop()
