import face_recognition
import cv2
import numpy as np 
import os

path = 'FaceRecognition/Known'
images = []
names = []

mylist=os.listdir(path)

for person in mylist:
    curImg=cv2.imread(f'{path}/{person}')
    images.append(curImg)
    names.append(os.path.splitext(person)[0])

def FindEncodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeKnown=FindEncodings(images)


video=cv2.VideoCapture('BreakingBad.mp4')

while True:
    success, img =video.read()
    imgSmall=cv2.resize(img,(0,0),None,1,1)
    imgSmall=cv2.cvtColor(imgSmall,cv2.COLOR_BGR2RGB)
    faceinframe=face_recognition.face_locations(imgSmall)
    encodinginframe=face_recognition.face_encodings(imgSmall,faceinframe)

    for encodeface,faceloc in zip(encodinginframe,faceinframe):
        matches=face_recognition.compare_faces(encodeKnown,encodeface)
        facedist=face_recognition.face_distance(encodeKnown,encodeface)
        match=np.argmin(facedist)
        
        if matches[match]:
            name=names[match].upper()
            #print(name)
            cv2.rectangle(imgSmall,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(0,255,0),2)
            cv2.putText(imgSmall,name,(faceloc[3]+20,faceloc[0]+50),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
            
            

    cv2.imshow('Output',imgSmall)
    cv2.waitKey(1)


