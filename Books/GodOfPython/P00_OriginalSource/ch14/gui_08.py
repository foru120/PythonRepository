# gui_08.py
import tkinter

root = tkinter.Tk()
cvs = tkinter.Canvas(root, width=100, height=100)
cvs.pack()
img = tkinter.BitmapImage(file="C:/gop/ch14/python.xbm")
cvs.create_image(50, 50, image=img)
root.mainloop()
