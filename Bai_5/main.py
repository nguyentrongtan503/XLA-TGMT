# main.py

from data_processing import load_and_process_data
from svm_classifier import svm_classifier
from cart_classifier import cart_classifier
from id3_classifier import id3_classifier

# Nạp và xử lý dữ liệu
X_train, X_test, y_train, y_test = load_and_process_data()

print("=== SVM Classifier ===")
svm_classifier(X_train, X_test, y_train, y_test)

print("\n=== CART Classifier (Gini Index) ===")
cart_classifier(X_train, X_test, y_train, y_test)

print("\n=== ID3 Classifier (Information Gain) ===")
id3_classifier(X_train, X_test, y_train, y_test)
