import cv2
from preproces import *
from red import *
import tensorflow as tf


cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

#Check de si funciona la camara web
if not cap.isOpened():
    raise IOError("No se pudo abrir la webcam")

while True:
    ret, frame = cap.read()
    frame_contorno,rects  = contornos(frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    image = cv2.putText(frame, 'MateoCareverga', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    
    #new_model = keras.models.load_model('modelo.h5')



    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)




    #imgs = recortes(frame_contorno,rects)

    #for image in imgs:
    #    print(model.predict(image))

    try:
        cv2.imshow('Input', frame_contorno)
    except:
    
        cv2.imshow('Input', frame)
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()