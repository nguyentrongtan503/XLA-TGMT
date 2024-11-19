from time import time
from sklearn.metrics import accuracy_score

def evaluate(model, X_test, y_test):
    start = time()
    predictions = model.predict(X_test)
    elapsed_time = time() - start
    accuracy = accuracy_score(y_test, predictions)
    return accuracy, elapsed_time
