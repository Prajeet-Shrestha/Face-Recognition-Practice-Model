import cv2
import numpy as np
import pickle

face_cascade = cv2.CascadeClassifier('C:\\Users\\Toothlexx\\Desktop\\Project #1\\cascades\\data\\haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("C:\\Users\\Toothlexx\\Desktop\\Project #1\\img\\trainer.yml")
lables= {}
with open("C:\\Users\\Toothlexx\\Desktop\\Project #1\\img\\lables.pickle","rb") as f:
    old_lables = pickle.load(f) 
    lables = {v:k for k, v in old_lables.items()}
capture = cv2.VideoCapture(0)
count = 1
while True:
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5,minNeighbors=5)
    for (x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h,x:x+w]
        img_name = "image-1-" + str(count) + ".png"  
        count+=1
        cv2.imwrite(img_name,roi_gray)
        rect_color = (255,0,0) #BGR
        thickness = 2
        cv2.rectangle(frame,(x,y),(x+w,y+h),rect_color,thickness)
    cv2.imshow('frame',frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
capture.release()
cv2.destroyAllWindows()