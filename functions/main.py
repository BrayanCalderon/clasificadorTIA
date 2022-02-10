import cv2
from preproces import *


cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

#Check de si funciona la camara web
if not cap.isOpened():
    raise IOError("No se pudo abrir la webcam")

while True:
    ret, frame = cap.read()
    frame_contorno  = contornos(frame)
    try:
        cv2.imshow('Input', frame_contorno)
    except:
        cv2.imshow('Input', frame)
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()