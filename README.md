# Modulo2_TC3002B.201

Josemaría Robledo Lara A01612376

## Dataset
El dataset utilizado es "BIRDS 525 SPECIES- IMAGE CLASSIFICATION" obtenido de la plataforma de Kaggle. Cuenta con 525 clases, las cuales están divididas en train, test y validation. Para este proyecto se redujo el número de clases a 20, esto debido al largo tiempo de entrenamiento al que se somete el modelo. Las clases son las siguientes:

- MANDRIN DUCK
- MANGROVE CUCKOO
- MARABOU STORK
- MASKED BOBWHITE
- MASKED BOOBY
- MASKED LAPWING
- MCKAYS BUNTING
- MERLIN
- MIKADO PHEASANT
- MILITARY MACAW
- MOURNING DOVE
- MYNA
- NICOBAR PIGEON
- NOISY FRIARBIRD
- NORTHERN BEARDLESS TYRANNULET
- NORTHERN CARDINAL
- NORTHERN FLICKER
- NORTHERN FULMAR
- NORTHERN GANNET
- NORTHERN GOSHAWK

Link al dataset original: https://www.kaggle.com/datasets/gpiosenka/100-bird-species

Link al dataset reducido: https://drive.google.com/drive/folders/1X3wNE4jmeC7B7UOqyVEw9U6AVVXe_8vu?usp=sharing

## birds_Josemaria.ipynb
Archivo demostrativo con el funcionamiento completo del proyecto.

Primero se cargan los datasets (train, valid y test) en sus respectivos directorios, para así definir el tamaño de batch (32) y de imagen (224x224), a su vez se obtienen los nombres de las clases para tener localizados cada uno de los labels. Para confirmar que todo esté correcto, se imprimen imágenes aleatorias de train

![imagen](https://github.com/A01612376/Modulo2_TC3002B.201/assets/83626334/a1b56edb-2a48-4fdc-87d7-c4e44c4707f7)


Para diversificar el conjunto de datos de entrenamiento, se aplica un data augmentation al conjunto de train. Las transformaciones aplicadas son rotaciones, acercamientos y volteos horizontales.

![imagen](https://github.com/A01612376/Modulo2_TC3002B.201/assets/83626334/337fb1ce-defd-4c4d-8591-3d951a51e8df)

Con los datos listos, se crea el modelo CNN con keras (explicado en la sección birds.keras)

Para la compilación del modelo se utilizó lo siguiente:
- Loss: sparse_categorical_crossentropy
- Optimizer: adam
- Metrics: accuracy

Se entrena el modelo con 20 épocas de 101 steps cada una.

**Resultados**

## birds.keras
Archivo con el modelo guardado.

El modelo es una CNN, ya que, las CNNs han demostrado un rendimiento superior en tareas como clasificación de imágenes, detección de objetos, segmentación semántica, entre otras. Esto demostrado en Deep residual learning for image recognition (He, K., Zhang, X., Ren, S., & Sun, J. 2016).

Al modelo se le aplica una capa de normalización con BatchNormalization. Cuenta con 5 capas convolucionales con diferentes filtros (32, 64, 128, 256, 256) y utilizan la función de activación ReLU. Al final de cada capa se utiliza MaxPooling para reducir la resolución espacial, además de definir el tamaño del padding.

Para evitar sobreajuste se utiliza la técnica dropout para reducir el sobreajuste. Finalmente para la capa de salida, se añade una capa densa con 20 neuronas, debido a la cantidad de clases existentes y su respectiva función de activación, que en este caso es softmax.

## main.py
Código para probar el modelo.
