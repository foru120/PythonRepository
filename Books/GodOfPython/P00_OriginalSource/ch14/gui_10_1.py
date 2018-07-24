# gui_10_1.py
import tkinter

root = tkinter.Tk()
frame = tkinter.Frame(root)
frame.pack()
button1 = tkinter.Button(text='button1', padx=25, pady=10)
button1.grid(in_=frame, row=0, column=0, padx=25, pady=10)
button2 = tkinter.Button(frame, text='-----button2----')
button2.grid(row=1, column=1)
button3 = tkinter.Button(frame, text='button3')
button3.grid(row=2, column=1, sticky='w')
root.mainloop()
