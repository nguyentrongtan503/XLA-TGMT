# id3_classifier.py

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

def id3_classifier(X_train, X_test, y_train, y_test):
    # ID3 Classifier với Information Gain
    id3_model = DecisionTreeClassifier(criterion='entropy')
    id3_model.fit(X_train, y_train)
    y_pred = id3_model.predict(X_test)
    
    print("ID3 (Information Gain) Accuracy:", accuracy_score(y_test, y_pred))
    print("ID3 (Information Gain) Classification Report:\n", classification_report(y_test, y_pred))
