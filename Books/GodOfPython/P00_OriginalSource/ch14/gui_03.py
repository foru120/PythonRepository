#gui_03.py
import tkinter
def increase():
    number.set(number.get()+1)

root = tkinter.Tk()
frame = tkinter.Frame(root)
frame.pack()
number = tkinter.IntVar(value = 0)
button = tkinter.Button(frame, text = 'increase', command = increase)
button.pack()
label = tkinter.Label(frame, text = 'start', textvariable = number)
label.pack()
root.mainloop()