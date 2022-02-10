import cv2
from preproces import *


cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

#Check de si funciona la camara web
if not cap.isOpened():
    raise IOError("No se pudo abrir la webcam")

while True:
    ret, frame = cap.read()
    frame_contorno  = contornos(frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    image = cv2.putText(frame, 'MateoCareverga', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)

    try:
        cv2.imshow('Input', frame_contorno)
    except:
    
        cv2.imshow('Input', frame)
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()