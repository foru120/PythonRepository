# gui_07_1.py
import tkinter

root = tkinter.Tk()
img = tkinter.BitmapImage(file='C:/gop/ch14/python.xbm')
button = tkinter.Button(root, image=img)  # 버튼에 그림을 넣음
button.pack()
root.mainloop()
