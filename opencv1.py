import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("C:\\Users\\Toothlexx\\Desktop\\Project #1\\img\\trainer.yml")
capture = cv2.VideoCapture(0)
count = 0
while True:
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5,minNeighbors=5)
    for (x,y,w,h) in faces:
        print(x,y,w,h)
        roi_gray = gray[y:y+h,x:x+w]

        #img_name = "image-" + str(count) + ".png"  
        #cv2.imwrite(img_name,roi_gray)

        rect_color = (255,0,0) #BGR
        thickness = 2
        cv2.rectangle(gray,(x,y),(x+w,y+h),rect_color,thickness)
    cv2.imshow('gray',gray)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
capture.release()
cv2.destroyAllWindows()