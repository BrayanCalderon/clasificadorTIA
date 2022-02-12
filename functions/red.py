import tensorflow as tf

def cargar_red():
    model = tf.keras.models.load_model('C:/Users/Brayan_Calderon/Documents/Github/clasificadorTIA/functions/modelo.h5')
    model.summary()
    return model

def prediccion(img,model):
    prediccion = model.predict(img)
    return prediccion