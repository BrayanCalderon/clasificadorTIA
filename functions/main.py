import cv2
from preproces import *
from red import *
import tensorflow as tf
import time


cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

#Check de si funciona la camara web
if not cap.isOpened():
    raise IOError("No se pudo abrir la webcam")
model = cargar_red()

while True:
    ret, frame = cap.read()
    
    frame_contorno,rects  = contornos(frame)
    imgs = recortes(frame,rects)
    #
    # print(len(imgs))


    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 0.7
    color = (0, 0, 0)
    thickness = 2
    
    switches = {'Switches 1' : 0,'Switches 2' : 0,'Switches 3' : 0,'Switches 4' : 0,'Switches 5' : 0,}


    def escribirpantalla(k, value, i):
        org = (50,50+(i*30))
        image = cv2.putText(frame, k+"-->"+str(value), org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)

    
    for img in imgs:
        img = img.reshape(1,100,100,1)
        result = prediccion(img,model)
        #print(result.shape)
        #print(type(result))
        if result[0,0] == np.amax(result[0]):
            switches["Switches 1"] = switches.get("Switches 1",0) +1
        elif result[0,1] == np.amax(result[0]):
            switches["Switches 2"] = switches.get("Switches 2",0) +1
        elif result[0,2] == np.amax(result[0]):
            switches["Switches 3"] = switches.get("Switches 3",0) +1
        elif result[0,3] == np.amax(result[0]):
            switches["Switches 4"] = switches.get("Switches 4",0) +1
        elif result[0,4] == np.amax(result[0]):
            switches["Switches 5"] = switches.get("Switches 5",0) +1
    i = 0
    for k in switches:
        print(i,k, switches[k])
        escribirpantalla(k,switches[k],i)
        i+=1

    try:
        cv2.imshow('Input', frame_contorno)
    except:
    
        cv2.imshow('Input', frame)
    c = cv2.waitKey(1)
    if c == 27:
        break
    #time.sleep(1) # Sleep for 3 seconds

cap.release()
cv2.destroyAllWindows()