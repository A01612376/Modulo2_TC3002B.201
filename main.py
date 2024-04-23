import tensorflow as tf
from tensorflow.keras.utils import load_img
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# Carga del modelo entrenado
model = load_model("birds.keras")

# Función para mostrar los resultados de la predicción
def results(filename, class_names):
    img = load_img(filename, target_size=(224, 224))
    imgconv = img_to_array(img)
    img_array = np.expand_dims(imgconv, axis=0)
    pred = np.argmax(model.predict(img_array))
    predimg = class_names[pred]
    predver = np.max(model.predict(img_array))
    print(pred)
    plt.figure()
    plt.imshow(img)
    plt.title("Eto: {}, veroyatnost : {}".format(predimg, predver))
    plt.show()

results('/content/gdrive/MyDrive/AI/Décimo/Reto Mod2/Birds/test/MASKED BOOBY/1.jpg', class_names)
