# -*- coding: utf-8 -*-

import cv2
print(cv2.__version__)

cascade_src = 'cascade/Bus_front.xml'
video_src = 'dataset/video2.avi'
#video_src = 'dataset/video2.avi'
font = cv2.FONT_HERSHEY_SIMPLEX # font tulisan

#cap = cv2.VideoCapture(video_src)
cap = cv2.VideoCapture(0)
car_cascade = cv2.CascadeClassifier(cascade_src)

while True:
    ret, img = cap.read()
    if (type(img) == type(None)):
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cars = car_cascade.detectMultiScale(gray, 1.1, 1) # cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        
        # write text, size 0.5, black(0,0,0), tebal=1
        cv2.putText(img,'kereta',(x,y), font, 0.5,(0,0,0),1,cv2.LINE_AA)
    
    cv2.imshow('video', img)
    
    if cv2.waitKey(33) == 27: # Esc Key
        break

cv2.destroyAllWindows()