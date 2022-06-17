from tkinter import *
from tkinter.font import Font
from PIL import ImageTk, Image
from tkinter import filedialog
import cv2
import PySimpleGUI as sg
import tensorflow as tf

w_base=Tk()

class_names = ["Angry", "Happy", "Sad", "Surprise"]

def Op_cam():
    #model=tf.keras.models.load_model("C:\\Users\\SOFI\\Desktop\\project_end2022\\model.h5")
    layout = [
    [sg.Image(key = '-IMAGE-')],
    [sg.Text('People in picture: 0', key='-TEXT-', expand_x= True, justification= 'c')]
    ]
    window=sg.Window('Emo_Recog', layout)
    video=cv2.VideoCapture(0)
    while True:
        event, values = window.read(timeout=0)
        if event == sg.WIN_CLOSED:
            break
        _, frame=video.read()
        #create a grayscale image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #the face detection
        face_cascade=cv2.CascadeClassifier('C:\\Users\\SOFI\\Desktop\\project_end2022\\haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(48,48))
        #create a rectangular
        for(x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0), thickness=2)
        #if faces is not None:
            #num = max(emotion[0])
            #idx = list(emotion[0]).index(num)
            #class_name = class_names[idx]
        imgbytes=cv2.imencode('.png', frame)[1].tobytes()
        window['-IMAGE-'].update(data=imgbytes)
        window['-TEXT-'].update(f'People in picture: {len(faces)} and emotion is ')
    window.close()

resize=(360,360)

def Inp_img():
    path= filedialog.askopenfilename(title="Select an Image", filetype=(('image    files','*.jpg'),('image    files','*.png'),('all files','*.*')))
    img= Image.open(path)
    img.thumbnail(resize)
    greyscale=img.convert('1')
    new_img=ImageTk.PhotoImage(greyscale)
    label.configure(image=new_img)
    label.image=new_img

def Recog_img(label): 
    face_cascade=cv2.CascadeClassifier('C:\\Users\\SOFI\\Desktop\\project_end2022\\haarcascade_frontalface_default.xml')
    faces1 = face_cascade.detectMultiScale(label, scaleFactor=1.1, minNeighbors=4, minSize=(48,48))
    for(x,y,w,h) in faces1:
        cv2.rectangle(label,(x,y),(x+w,y+h),(255,0,0), thickness=2)

w_base.title('Emo_Recog')
w_base.iconbitmap('C:\\Users\\SOFI\\Desktop\\project_end2022\\face_recognition_icon_135652.ico')
w_base.configure(bg='#6699ff')


welc_lab=Label(w_base, text="Welcome!",bg='#6699ff',fg="black", font=('Times New Roman',60),padx=70,pady=70)
welc_lab.grid(row=0)

op_cam=Button(w_base, text="Open camera",command=Op_cam, padx=50,pady=10)
op_cam.grid(row=1)

frame1 = Frame(w_base, width=360, height=360)
frame1.grid(row=3, columnspan=2)

img1= ImageTk.PhotoImage(Image.open("C:\\Users\\SOFI\\Desktop\\project_end2022\\no_imgaval.jpg"))
label= Label(frame1,image= img1)
label.pack()

input_img=Button(w_base, text=" Input image  ",command=lambda: Inp_img(), padx=50,pady=10)
input_img.grid(row=2)

rec_img=Button(w_base, text=" Click for emotion  ",command=lambda: Recog_img(label), padx=50,pady=10)
rec_img.grid(row=4)

result=Label(w_base, text="emotion",bg='#6699ff',fg="black", font=('Times New Roman',20),padx=20,pady=20)
result.grid(row=5)

w_base.geometry("500x800")
w_base.mainloop()