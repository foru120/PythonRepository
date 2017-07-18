#gui_15.py
import tkinter

root = tkinter.Tk()
frame = tkinter.Frame(root)
frame.pack()

def result():
    var_result.set('I have checked ' + var.get()+ '.')

var = tkinter.StringVar()
var.set("Nothing")
var_result = tkinter.StringVar()
var_result.set('result')

r1 = tkinter.Radiobutton(frame, text="Sun" ,variable = var, value = "Sun", command=result)
r2 = tkinter.Radiobutton(frame, text="Mon" ,variable = var, value = "Mon", command=result)
r3 = tkinter.Radiobutton(frame, text="Tue" ,variable = var, value = "Tue", command=result)
r4 = tkinter.Radiobutton(frame, text="Wen" ,variable = var, value = "Wen", command=result)
r5 = tkinter.Radiobutton(frame, text="Thu" ,variable = var, value = "Thu", command=result)
r6 = tkinter.Radiobutton(frame, text="Fri" ,variable = var, value = "Fri", command=result)
r7 = tkinter.Radiobutton(frame, text="Sat" ,variable = var, value = "Sat", command=result)

label = tkinter.Label(frame, textvariable=var_result, background='white')
label.pack()

radiogroup = [r1,r2,r3,r4,r5,r6,r7]

for x in radiogroup:
    x.pack(side=tkinter.LEFT)

root.mainloop()