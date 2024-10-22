import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh X-quang
image = cv2.imread('data/xq.png', cv2.IMREAD_GRAYSCALE)

# 1. Ảnh âm tính
negative_image = 255 - image

# 2. Tăng cường độ tương phản
# Chuyển đổi sang kiểu float để tăng cường độ tương phản
contrast_image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)

# 3. Biến đổi logarit
c = 255 / np.log(1 + np.max(image))  
log_image = c * (np.log(image + 1))
log_image = np.array(log_image, dtype=np.uint8)

# 4. Cân bằng histogram
equalized_image = cv2.equalizeHist(image)

# Hiển thị các ảnh kết quả
titles = ['Ảnh gốc', 'Ảnh âm tính', 'Tăng tương phản', 'Biến đổi log', 'Ảnh cân bằng (histogram)']
images = [image, negative_image, contrast_image, log_image, equalized_image]

for i in range(5):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
