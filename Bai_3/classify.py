import os
import cv2
import numpy as np
from preprocess import preprocess
from sklearn.preprocessing import StandardScaler

DATASET_DIR = "D:/XLA_TGMT/Xu ly anh/Bai_3/datatrain"  # Thư mục chứa ảnh gán nhãn
LABELS_FILE = "labels.txt"  # Lưu nhãn vào file txt
FEATURES_FILE = "features.txt"  # Lưu đặc trưng vào file txt

def classify():
    labels = []
    features = []

    for filename in os.listdir(DATASET_DIR):
        filepath = os.path.join(DATASET_DIR, filename)
        label = filename.split("_")[0]  # Tách nhãn từ tên file
        
        img = cv2.imread(filepath)
        if img is None:
            print(f"Cannot read file: {filepath}")
            continue
        
        # Tiền xử lý ảnh
        _, imgThresh = preprocess(img)
        
        # Resize ảnh để trích xuất đặc trưng
        imgResized = cv2.resize(imgThresh, (20, 30))
        imgFlattened = imgResized.flatten()
        
        features.append(imgFlattened)
        labels.append(label)
        print(f"Processed {filename} - Label: {label}, Features size: {imgFlattened.size}")

    # Chuẩn hóa các vectơ đặc trưng
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Lưu nhãn và đặc trưng vào file
    np.savetxt(LABELS_FILE, labels, fmt='%s')
    np.savetxt(FEATURES_FILE, features)

    print("Successfully labeled and saved data.")
    
if __name__ == "__main__":
    classify()
