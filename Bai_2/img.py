import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image using the correct path
image = cv2.imread('data/m1.jpg')


    # Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Sobel edge detection
sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)  # Gx
sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)  # Gy

    # Calculate gradient magnitude
sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    # Apply Laplacian of Gaussian (LoG)
laplacian_gaussian = cv2.Laplacian(gray_image, cv2.CV_64F, ksize=3)

    # Normalize and convert to uint8 for displaying
sobel_combined = cv2.convertScaleAbs(sobel_combined)
laplacian_gaussian = cv2.convertScaleAbs(laplacian_gaussian)

    # Display images
plt.figure(figsize=(10,5))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 2)
plt.title("Sobel Edge Detection")
plt.imshow(sobel_combined, cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Laplacian of Gaussian")
plt.imshow(laplacian_gaussian, cmap='gray')

plt.tight_layout()
plt.show()