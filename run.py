import subprocess
from tkinter import *



def show_frame():
    value=var.get()
    if(value==0):
        subprocess.run(["python","webcam.py"]) # run the webcam program in background
    else:
        subprocess.run(["python","path.py"])  #  run the path finding program in background
    
    
 #Initializing tkinter window to root
root=Tk()                                                                 
root.title("Select by webcam or any Video")
root.geometry("400x100+50+50")
 #creating a variable to store the radiobutton selection
var=IntVar() 
 #Creating two radio buttons                                                              
webcam=Radiobutton(root,text="Open Webcam",variable=var,value=0).place(relx=0.2,rely=0.2)       
path=Radiobutton(root,text="Select any Video",variable=var,value=1).place(relx=0.5,rely=0.2) 
#creating a start button to start webcam or by video path
button =  Button(root, text='Start',width=10, command=show_frame).place(relx=0.4,rely=0.5)          

root.mainloop()


    
    