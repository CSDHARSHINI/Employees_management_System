##package
import cv2
import numpy as np
import os
import pickle


##xml file
video=cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier('haarcascade_frontalface_default (1).xml')

face_data=[]

i = 0

name=input("enter name: ")


##webcam to detect the faces
while True:
    ret,frame=video.read()## reading a image
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)## convert a colorful image to black and white
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for(x,y,w,h) in faces:
        crop_img=frame[y:y+h, x:x+w, :]## crop the image
        resized_img=cv2.resize(crop_img, dsize=(50,50))## resize the image
        if len(face_data)<=100 and i%10==0:
            face_data.append(resized_img)
        i=i+1
        cv2.putText(frame, str(len(face_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (50,50,255), 1)

    cv2.imshow("frame", frame)
    k=cv2.waitKey(1)
    if len(face_data)==50:
        break


# Release the video capture object and close all OpenCV windows
video.release()
cv2.destroyAllWindows()



##save the pickle package in the pickle file

face_data=np.array(face_data)
face_data=face_data.reshape(100,-1)

## saving the two pickle file
if 'name.pkl' not in os.listdir('data/'):
    names=[name]*100
    with open('data/names.pkl','wb') as f:
        pickle.dump(names,f)
else:
    with open('data/names.pkl','rb') as f:
        names=pickle.load(f)
        names=names+[names]*100

    with open('data/names.pkl','wb') as f:
        pickle.dump(names,f)


##load the dataset

if 'face_data.pkl' not in os.listdir('data/'):
    with open('data/face_data.pkl','wb') as f: 
        pickle.dump(face_data,f) 
else:
    with open('data/face_data.pkl','rb') as f:
        faces=pickle.load(f)
    faces=np.append(faces,face_data,axis=0)
    with open('data/face_data.pkl','wb') as f:
        pickle.dump(face_data,f) 