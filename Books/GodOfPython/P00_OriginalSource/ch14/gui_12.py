#gui_12.py
import tkinter
def test(e):
	a = "	\
	char : {}\n\
	delta : {}\n\
	height : {}\n\
	keycode: {}\n\
	keysym : {}\n\
	keysym_num : {}\n\
	num : {}\n\
	time : {}\n\
widget : {}\n\
	width : {}\n\
	x : {}\n\
	y : {}\n\
	x_root : {}\n\
	y_root : {}\n".format(e.char,
e.delta,
e.height,
e.keycode,
e.keysym,
e.keysym_num,
e.num,
e.time,
e.widget,
e.width,
e.x,
e.y,
e.x_root,
e.y_root
)
	info.set(a)


root = tkinter.Tk()
info = tkinter.StringVar()
frame = tkinter.Frame(root, width = 500, height=500, padx=100)
frame.grid()
button = tkinter.Button(root, text = 'Test')
button.grid()
label_title = tkinter.Label(frame, text = "------------test	\
Event-----------", justify = 'left')
label_title.grid()
label = tkinter.Label(frame, textvariable = info, justify = 'left')
label.grid()
print(id(label))
root.bind("<ButtonPress>", lambda e: test(e))
root.bind("<MouseWheel>", lambda e: test(e))
root.bind("<KeyPress>", test)
root.bind("<Motion>", test)
root.mainloop()