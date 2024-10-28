import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from preprocess import preprocess

LABELS_FILE = "labels.txt"
FEATURES_FILE = "features.txt"

def load_data():
    features = np.loadtxt(FEATURES_FILE)
    with open(LABELS_FILE, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return features, labels

def classify_image(imgPath):
    # Tải dữ liệu gán nhãn
    features, labels = load_data()
    
    # Khởi tạo mô hình KNN
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(features, labels)
    
    # Đọc và tiền xử lý ảnh đầu vào
    img = cv2.imread(imgPath)
    if img is None:
        print("Không thể tải ảnh.")
        return
    
    _, imgThresh = preprocess(img)
    imgResized = cv2.resize(imgThresh, (20, 30))
    imgFlattened = imgResized.flatten().reshape(1, -1)

    # Dự đoán và in kết quả
    prediction = knn.predict(imgFlattened)
    print(f"Dự đoán: {prediction[0]}")
    
    # Hiển thị hình ảnh cùng với tên nhãn
    # cv2.imshow('Hình ảnh', img)
    cv2.putText(img, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('ANH', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    imgPath = "D:/XLA_TGMT/Xu ly anh/Bai_3/data/dog.jpg"  # Đường dẫn ảnh đầu vào
    classify_image(imgPath)
