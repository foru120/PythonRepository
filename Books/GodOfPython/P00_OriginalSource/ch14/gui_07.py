# gui_07.py
import tkinter

root = tkinter.Tk()
cvs = tkinter.Canvas(root, width=100, height=100)
cvs.pack()
cvs.create_bitmap(50, 50, bitmap="@C:/gop/ch14/python.xbm")
root.mainloop()
