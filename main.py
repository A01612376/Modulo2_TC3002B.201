import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image

test_dir = 'test'
image_size = (224, 224)
batch_size = 32

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

class_names = list(test_generator.class_indices.keys())
print(np.shape(class_names))

# Cargar el modelo preentrenado
model = load_model('birds_model.h5')

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

# Llamada a la función results fuera de su definición
results('test/MERLIN/2.jpg', class_names)
#results('mandarin.jpeg', class_names)
