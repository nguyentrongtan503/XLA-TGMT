import cv2
import numpy as np

def preprocess(imgOriginal):
    # Chuyển sang ảnh xám
    imgGray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)

    # Làm mịn ảnh để giảm nhiễu
    imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)

    # Tăng cường tương phản bằng CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgContrast = clahe.apply(imgBlurred)

    # Tìm biên bằng Canny
    imgThresh = cv2.Canny(imgContrast, 50, 150)  # Thay đổi ngưỡng nếu cần

    return imgGray, imgThresh

# if __name__ == "__main__":
#     # Đọc ảnh mẫu để kiểm tra
#     imgPath = ""  # Đường dẫn tới ảnh
#     imgOriginal = cv2.imread(imgPath)
    
#     # Tiền xử lý ảnh
#     imgGray, imgThresh = preprocess(imgOriginal)
    
#     # Hiển thị các ảnh đã xử lý
#     cv2.imshow('Original Image', imgOriginal)
#     cv2.imshow('Gray Image', imgGray)
#     cv2.imshow('Threshold Image', imgThresh)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()