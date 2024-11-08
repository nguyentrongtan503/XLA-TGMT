import cv2
import numpy as np

def load_and_preprocess_image(filepath, size=(100, 100)):
    # Load image
    image = cv2.imread(filepath)
    # Resize to desired dimensions
    image = cv2.resize(image, size)
    # Normalize pixel values to range [0, 1]
    image = image.astype('float32') / 255.0
    return image

def flatten_image(image):
    # Convert 2D image to a 1D vector for clustering
    return image.reshape(-1, image.shape[-1])
