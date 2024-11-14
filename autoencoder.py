import os
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split

# Directorios
full_images_dir = "dataset/full_images/"
segmented_images_dir = "dataset/segmented_images/"

# Parámetros
IMG_HEIGHT, IMG_WIDTH = 128, 128  # Tamaño de las imágenes
BATCH_SIZE = 16
EPOCHS = 50

# Función para cargar y procesar las imágenes
def load_images(image_dir):
    images = []
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = img_to_array(img) / 255.0  # Normalizar entre 0 y 1
        images.append(img_array)
    return np.array(images)

# Cargar imágenes segmentadas y originales
segmented_images = load_images(segmented_images_dir)
full_images = load_images(full_images_dir)

# Dividir en conjunto de entrenamiento y validación
x_train, x_val, y_train, y_val = train_test_split(segmented_images, full_images, test_size=0.2, random_state=42)

# Definir el modelo Autoencoder
input_img = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Encoder
x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)

# Decoder
x = UpSampling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

# Construir el modelo
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Entrenar el modelo
autoencoder.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_val, y_val))

# Guardar el modelo entrenado
autoencoder.save("autoencoder_model.h5")
