# Importar las librerías necesarias
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Establecer el directorio base donde se encuentran los datos de entrenamiento y prueba
base_dir = 'K:\\Josem\\Documents\\Tec Mty\\Décimo Semestre\\Desarrollo de aplicaciones avanzadas de ciencias computacionales\\M2\\Reto\\Modulo2_TC3002B.201'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Crear un generador de datos de entrenamiento con diferentes transformaciones de imágenes
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=60,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True)

# Cargar los datos de entrenamiento con el generador de datos
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=8,
    class_mode='binary',
)

# Obtener un lote de imágenes y etiquetas del generador de datos
images, labels = train_generator[0]
print(images.shape)  # Imprimir la forma de las imágenes
print(labels)  # Imprimir las etiquetas

# Visualizar las imágenes del lote
plt.figure()
f, axarr = plt.subplots(1, images.shape[0], figsize=(30, 4))
for i in range(images.shape[0]):
    axarr[i].imshow(images[i])

# Crear un nuevo generador de datos de entrenamiento y guardar las imágenes aumentadas
path = "K:\\Josem\\Documents\\Tec Mty\\Décimo Semestre\\Desarrollo de aplicaciones avanzadas de ciencias computacionales\\M2\\Reto\\Modulo2_TC3002B.201\\nuevos"
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=8,
    class_mode='binary',
    save_to_dir=path + '/augmented',
    save_prefix='aug',
    save_format='png'
)

# Importar clases y funciones necesarias de Keras
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import layers

# Definir el modelo de red neuronal convolucional
model = models.Sequential()
model.add(layers.Conv2D(10, (3, 3), activation="relu", input_shape=(150, 150, 3)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Imprimir un resumen del modelo
model.summary()

# Compilar el modelo con una función de pérdida, optimizador y métricas
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=2e-5),
              metrics=['acc'])

# Entrenar el modelo con los datos de entrenamiento
history = model.fit(
    train_generator,
    epochs=10)

# Obtener la precisión y pérdida del entrenamiento
acc = history.history['acc']
loss = history.history['loss']
epochs = range(1, len(acc) + 1)

# Graficar la precisión del entrenamiento
plt.plot(epochs, acc, 'bo', label='train accuracy')
plt.title('train acc')
plt.legend()

# Graficar la pérdida del entrenamiento
plt.figure()
plt.plot(epochs, loss, 'bo', label='training loss')
plt.title('train loss')
plt.legend()
plt.show()

# Crear un generador de datos de prueba
test_datagen = ImageDataGenerator(1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

# Evaluar el modelo con los datos de prueba
test_loss, test_acc = model.evaluate(test_generator, steps=25)
print('\ntest acc :\n', test_acc)