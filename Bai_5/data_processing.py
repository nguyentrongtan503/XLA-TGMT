
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

def load_and_process_data():
    # Đọc dữ liệu Iris
    data = pd.read_csv('iris.csv', header=0)  # Nếu tệp có tiêu đề

    X = data.drop(columns=['Species'])
    y = data['Species']
    
    # Chuẩn hóa dữ liệu (tùy chọn)
    X = preprocessing.StandardScaler().fit_transform(X)
    
    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test
