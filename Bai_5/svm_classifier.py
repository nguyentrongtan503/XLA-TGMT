# svm_classifier.py

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def svm_classifier(X_train, X_test, y_train, y_test):
    # SVM Classifier
    svm_model = SVC(kernel='linear')  # Hoặc 'rbf', 'poly' tùy chọn
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    
    print("SVM Accuracy:", accuracy_score(y_test, y_pred))
    print("SVM Classification Report:\n", classification_report(y_test, y_pred))
