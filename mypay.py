import cv2
import numpy as np
import pydoc
# img = cv2.imread('tel1.jpg')
# cv2.imshow('result',img)
# cv2.waitKey(0)
# def rotate(img_p,angle):
#     height,width=img_p.shape[:2]
#     point = (width//2,height//2)
#     mat=cv2.getRotationMatrix2D(point,angle,1)
#     return cv2.warpAffine(img_p,mat,(width,height))

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    new_img = np.zeros(img.shape,dtype='uint8') # создаем пустую картинку
    # img =cv2.GaussianBlur(img,(3,3),3) # размытие
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    circle= cv2.circle(img.copy(),(125,125),150,255,1)
    # img = cv2.Canny(img,30,130) # выделяет края
    faces=cv2.CascadeClassifier('Hand.Cascade.1.xml') # подгрузка HaarCascade уже натренированной нейронной сети  на определенные объекты

    results=faces.detectMultiScale(img,scaleFactor=1.5 ,minNeighbors=3)
    for (x,y,w,h) in results:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),thickness=1)

    # img_new = cv2.bitwise_and(circle, img) # объединение множеств точек
    # con,hir = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    # print(con)
    # cv2.drawContours(new_img,con,-1,(230,111,148),1) # рисуем выделенные контуры
    # img = rotate(img,-90)
    cv2.imshow('Result',img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
