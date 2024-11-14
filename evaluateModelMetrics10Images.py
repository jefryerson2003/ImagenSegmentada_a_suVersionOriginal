import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import mean_squared_error 
import cv2
import matplotlib.pyplot as plt

test_images_dir = "dataset/segmentedTest_images/"  

# Parámetros
IMG_HEIGHT, IMG_WIDTH = 128, 128  # Tamaño de las imágenes

# Función para cargar y procesar las imágenes
def load_image(image_path):
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0  # Normalizar entre 0 y 1
    return np.expand_dims(img_array, axis=0)  # Añadir una dimensión extra para el batch size

# Cargar las 10 primeras imágenes de prueba
test_images = []
test_image_paths = [os.path.join(test_images_dir, img) for img in os.listdir(test_images_dir)[:10]]  # Cargar las primeras 10 imágenes

for img_path in test_image_paths:
    test_images.append(load_image(img_path))

test_images = np.vstack(test_images)  # Apilar las imágenes en un array

# Cargar el modelo entrenado desde el archivo .h5 (con la corrección para MSE)
autoencoder = load_model("autoencoder_model.h5", custom_objects={'mse': MeanSquaredError()})

# Realizar las predicciones en las 10 imágenes
predictions = autoencoder.predict(test_images)

# Calcular MSE y PSNR
mse_values = []
psnr_values = []

for i in range(10):
    # Cálculo de MSE
    mse = mean_squared_error(test_images[i].flatten(), predictions[i].flatten())
    mse_values.append(mse)
    
    # Cálculo de PSNR
    psnr = cv2.PSNR(test_images[i], predictions[i])
    psnr_values.append(psnr)

# Mostrar métricas
print("MSE de las 10 imágenes:", mse_values)
print("MSE promedio:", np.mean(mse_values))
print("PSNR de las 10 imágenes:", psnr_values)
print("PSNR promedio:", np.mean(psnr_values))

# Mostrar las imágenes originales y reconstruidas
plt.figure(figsize=(20, 4))
for i in range(10):
    # Imagen original
    plt.subplot(2, 10, i + 1)
    plt.imshow(test_images[i])  # Mostrar la imagen original
    plt.title("Original")
    plt.axis('off')
    
    # Reconstrucción
    plt.subplot(2, 10, i + 11)
    plt.imshow(predictions[i])  # Mostrar la imagen reconstruida
    plt.title("Reconstrucción")
    plt.axis('off')

plt.show()
