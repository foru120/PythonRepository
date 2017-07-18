# gui_16.py
import tkinter  # python 3.x
from tkinter import messagebox  # python 3.x
# import tkMessageBox as messagebox      #python 2.x
# import Tkinter as tkinter               #python 2.x

root = tkinter.Tk()

if messagebox.askokcancel(title="Hello python", detail="Yes or No"):
    messagebox.showinfo(title="OK", detail="You have pressed 'Yes'")
else:
    messagebox.showwarning(title="Warning", detail="You have pressed 'No'")

root.mainloop()
