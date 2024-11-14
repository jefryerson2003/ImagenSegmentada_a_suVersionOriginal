import cv2
import numpy as np
import os
from PIL import Image

# Directorios
input_dir = "DataSet/full_images"  # Directorio de imágenes originales
segmented_dir = "DataSet/segmented_images"  # Directorio para guardar las segmentadas
os.makedirs(segmented_dir, exist_ok=True)

def segment_image_superpixels(image_path, output_path, n_segments=100):
    # Leer la imagen
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir a RGB

    # Aplicar el algoritmo de superpíxeles SLIC
    slic = cv2.ximgproc.createSuperpixelSLIC(image_rgb, region_size=30, ruler=10.0)
    slic.iterate(10)  # Más iteraciones para mejor segmentación

    # Obtener la máscara de superpíxeles
    mask_slic = slic.getLabelContourMask()
    segmented_image = image_rgb.copy()
    segmented_image[mask_slic == 255] = [0, 0, 0]  # Resaltar bordes en negro

    # Convertir a imagen de PIL y guardar
    segmented_img = Image.fromarray(segmented_image)
    segmented_img.save(output_path)

# Aplicar segmentación a todas las imágenes
for img_name in os.listdir(input_dir):
    if img_name.endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(segmented_dir, img_name)
        segment_image_superpixels(input_path, output_path)
