from tkinter import *
from tkinter import messagebox
import subprocess as sub
"""Goal: Make a GUI to connect to an excel file to log in math students.
   Start Date: 31st December 2019
   End Date: """

"Variables Needed"
#Last Name
#ID Number
#MTH course
#Time In
#Time Out
#Topics


#Base
root = Tk()
root.geometry('500x600')

#Title
root.title("Login Sheet")
label_0 = Label(root, text="Login Sheet", width = 20, font = ("bold", 20))
label_0.place(x = 90, y = 53)

#Last Name
label_1 = Label(root, text = "Last Name",width = 20, font = ("bold", 10))
label_1.place(x = 80, y = 130)
entry_1 = Entry(root)
entry_1.place(x = 240, y = 130)

#ID Number
label_2 = Label(root, text = "ID Number", width = 20, font = ("bold", 10))
label_2.place(x = 68, y = 180)
entry_2 = Entry(root)
entry_2.place(x=240,y=180)

#Gender
label_3 = Label(root, text = "Gender", width = 20, font = ("bold", 10))
label_3.place(x = 70, y = 230)
var = IntVar()
Radiobutton(root, text = "Male",padx = 5, variable = var, value = 1).place(x=235,y=230)
Radiobutton(root, text = "Female",padx = 20, variable = var, value = 2).place(x=290,y=230)

#MTH Course
label_4 = Label(root, text = "STA Course", width = 20, font = ("bold", 10))
label_4.place(x=70,y=280)

list1 = ["116", "147", "323", "347"];
c = StringVar()
droplist = OptionMenu(root,c, *list1)
droplist.config(width=15)
c.set("Select STA Course")
droplist.place(x=240,y=280)

#Time-In
label_5 = Label(root, text = "Time In", width = 20, font = ("bold", 10))
label_5.place(x = 80, y = 330)
entry_5 = Entry(root)
entry_5.place(x=240,y = 330)

#Time-Out
label6 = Label(root, text = "Time Out", width = 20, font = ("bold", 10))
label6.place(x = 80, y = 360)
entry6 = Entry(root)
entry6.place(x = 240, y = 360)

#Submission Button
Button(root, text = 'Submit', width = 20, bg = 'brown', fg = 'white').place(x = 180, y = 420)


root.mainloop()


