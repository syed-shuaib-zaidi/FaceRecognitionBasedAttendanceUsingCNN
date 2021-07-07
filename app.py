import tkinter as tk
from utils import *
from face_data_collect import *
from face_recognition import *
from training import *

window = tk.Tk()
window.bind('<Escape>', lambda event: window.destroy())
window.state('normal')

window.title("Attendance System")
window.configure(background='pink')
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)
x_cord = 75;
y_cord = 20;
checker=0;

message = tk.Label(window, text="JAMIA MILLIA ISLAMIA UNIVERSITY" ,bg="white"  ,fg="black"  ,width=35  ,height=2,font=('Times New Roman', 20, 'bold')) 
message.place(x=750, y=760, anchor=tk.CENTER)

message = tk.Label(window, text="ATTENDANCE MANAGEMENT PORTAL" ,bg="pink"  ,fg="black"  ,width=40  ,height=1,font=('Times New Roman', 35, 'bold underline')) 
message.place(x=200, y=20)

lbl = tk.Label(window, text="Enter Your College ID",width=20  ,height=2  ,fg="black"  ,bg="Pink" ,font=('Times New Roman', 25, ' bold ') ) 
lbl.place(x=110, y=200-y_cord)


txt = tk.Entry(window,width=30,bg="white" ,fg="blue",font=('Times New Roman', 15, ' bold '))
txt.place(x=120, y=300-y_cord)
# txt.pack(ipady=3)

lbl2 = tk.Label(window, text="Enter Your Name",width=20  ,fg="black"  ,bg="pink"    ,height=2 ,font=('Times New Roman', 25, ' bold ')) 
lbl2.place(x=740-x_cord, y=200-y_cord)

txt2 = tk.Entry(window,width=30  ,bg="white"  ,fg="blue",font=('Times New Roman', 15, ' bold ')  )
txt2.place(x=780-x_cord, y=300-y_cord)
# txt2.pack(ipady=3)

takeImg = tk.Button(window, text="REGISTER",command= lambda arg1=txt, arg2=txt2 : register(arg1,arg2)
,fg="white"  ,bg="blue"  ,width=30  ,height=2, activebackground = "pink" ,
font=('Times New Roman', 15, ' bold '), state="disabled")
takeImg.place(x=120, y=412-y_cord)

trackImg = tk.Button(window, text="MARK ATTENDANCE" , command=lambda : mark_attendance(),fg="white"  ,bg="red"  ,width=30  ,height=2, activebackground = "pink" ,font=('Times New Roman', 15, ' bold '))
trackImg.place(x=780-x_cord, y=412-y_cord)

lbl3 = tk.Button(window, text="TRAIN",command= lambda : train(),width=20  ,fg="white"  ,bg="lightgreen"  ,height=2 ,font=('Times New Roman', 30, ' bold ')) 
lbl3.place(x=120, y=570-y_cord)


## events
txt.bind('<Key>',lambda event, arg1=txt ,arg2=txt2, arg3=takeImg:switch(event,txt,txt2,takeImg))
txt2.bind('<Key>',lambda event, arg1=txt ,arg2=txt2, arg3=takeImg:switch(event,txt,txt2,takeImg))

window.mainloop()

