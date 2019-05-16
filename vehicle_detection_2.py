# -*- coding: utf-8 -*-

import cv2
print(cv2.__version__)


# Load file
cascade_src = 'cascade/cars.xml'
cascade_src2 = 'cascade/Bus_front.xml'
video_src = 'dataset/video6.MOV'

# Font
font = cv2.FONT_HERSHEY_SIMPLEX

# Type of video capture [0 for stream]
cap = cv2.VideoCapture(video_src)
#cap = cv2.VideoCapture(0)


# code start
car_cascade = cv2.CascadeClassifier(cascade_src)
bus_cascade = cv2.CascadeClassifier(cascade_src2)

while True:
    ret, img = cap.read()
    if (type(img) == type(None)):
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    cars = car_cascade.detectMultiScale(gray, 1.9, 1) #2 # cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    bus = bus_cascade.detectMultiScale(gray, 1.1, 1) #1.1

    # kotak for car
    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        
        # write text, size 0.5, black(0,0,0), tebal=1
        cv2.putText(img,'Kereta RM 1',(x,y-5), font, 0.5,(0,255,0),1,cv2.LINE_AA)

    # kotak for bus
    for (x,y,w,h) in bus:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        
        # write text, size 0.5, black(0,0,0), tebal=1
        cv2.putText(img,'Bas RM 2',(x,y-5), font, 0.5,(0,255,0),1,cv2.LINE_AA)


    cv2.imshow('video', img)
    
    if cv2.waitKey(33) == 27: # Esc Key
        break

cap.release()
cv2.destroyAllWindows()