# gui_09.py
import tkinter

root = tkinter.Tk()
frame = tkinter.Frame(root, background='red')
frame.pack()
button1 = tkinter.Button(frame, text='--button1--')
button1.pack()      # button1.pack(side = 'left')
button2 = tkinter.Button(frame, text='button2')
button2.pack()      # button2.pack(fill='x')
button3 = tkinter.Button(frame, text='****button3****')
button3.pack()
root.mainloop()
