import time
from preprocess import load_iris_data
from knn_classifier import KNN
from svm_classifier import SVM
from ann_classifier import ANN

# Load data
X_train, X_test, y_train, y_test = load_iris_data()

# Initialize models
knn_model = KNN(k=3)
svm_model = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
ann_model = ANN(n_features=X_train.shape[1], n_hidden=10, n_outputs=3, learning_rate=0.01, n_iters=1000)

# Dictionary to store results
results = {}

# Evaluate KNN
start_time = time.time()
knn_model.train(X_train, y_train)
knn_predictions = knn_model.predict(X_test)
knn_accuracy = knn_model.accuracy(knn_predictions, y_test)
results['KNN'] = {'accuracy': knn_accuracy, 'time': time.time() - start_time}

# Evaluate SVM
start_time = time.time()
svm_model.train(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
svm_accuracy = svm_model.accuracy(svm_predictions, y_test)
results['SVM'] = {'accuracy': svm_accuracy, 'time': time.time() - start_time}

# Evaluate ANN
start_time = time.time()
ann_model.train(X_train, y_train)
ann_predictions = ann_model.predict(X_test)
ann_accuracy = ann_model.accuracy(ann_predictions, y_test)
results['ANN'] = {'accuracy': ann_accuracy, 'time': time.time() - start_time}

# Print results
for model, metrics in results.items():
    print(f"{model} Results:")
    print(f"  Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"  Time Taken: {metrics['time']:.4f} seconds")
