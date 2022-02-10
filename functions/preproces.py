import cv2
import numpy as np
import matplotlib.pyplot as plt
def gaussian(img):
    img = cv2.medianBlur(img,7)
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
    return th2

#Funcion para calcular W y H
def area(box):
    x = box[0,1]
    y = box[0,0]
    dx1 = box[0,1]-box[3,1]
    dy1 = box[0,0]-box[3,0]
    dx2 = box[0,1]-box[1,1]
    dy2 = box[0,0]-box[1,0]
    w   = (dx1**2+dy1**2)**(1/2)
    h   = (dx2**2+dy2**2)**(1/2)
    return x,y,w,h


def plotimg(img):
    plt.figure(figsize=[8,8])
    plt.imshow(img, cmap = 'gray')
    
    plt.xticks([]),plt.yticks([])
    plt.show()

def contornos(frame):

    # Convert image to gray and blur it
    
    src_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #src_color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB )
    src_color = frame
    src_filter = gaussian(src_gray)

    #Closing
    kernel = np.ones((3,3),np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    closing = cv2.morphologyEx(src_filter, cv2.MORPH_CLOSE, kernel, iterations = 1)
    
    #Binary Treshold 

    ret,thresh1 = cv2.threshold(closing,190,255,cv2.THRESH_BINARY)
    for i in range(len(thresh1)):
        for j in range(5):
            thresh1[i,j] = 255
    for i in range(-1,-9,-1):
        for j in range(len(thresh1)):
            thresh1[i,j] = 255

    #erode
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(thresh1,kernel,iterations = 1)

    img_copy = thresh1.copy()

    # Encuentra los contornos 
    contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    outer_contours = []
    #src_color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB )
    src_color = frame
    color_copy = src_color.copy()

    # Seleccionar solo los contornos que no tienen padres por nivel de jerarquia
    for i in range(len(hierarchy[0])):
        if hierarchy[0,i,3] < 1:
         outer_contours.append(contours[i])
        else:
         continue
    rects = []
    h_max = 0
    w_max = 0

    for i in range(len(outer_contours)):
        
        # Obtengo el centro del contorno, largo y alto, y ángulo
        rect = cv2.minAreaRect(outer_contours[i])
        # obtengo representación en puntos
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        x ,y ,w, h = area(box)
        
        # Selecciona los contornos que cumplen ciertas condiciones
        if (w*h >= src_color.shape[0]*src_color.shape[1]) or w*h < 250 or w*h > 200000:
            continue
        else:
            #Dibuja los contornos de los elementos encontrados
            cv2.drawContours(src_color,[box],0,(0,0,255),2)
            rect = cv2.minAreaRect(outer_contours[i])
            rects.append(rect)
   
    return src_color, rects

def colorear(rect,img):
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img,[box],0,(255,0,0),2)
    return img



