# cart_classifier.py

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

def cart_classifier(X_train, X_test, y_train, y_test):
    # CART Classifier với Gini Index
    cart_model = DecisionTreeClassifier(criterion='gini')
    cart_model.fit(X_train, y_train)
    y_pred = cart_model.predict(X_test)
    
    print("CART (Gini Index) Accuracy:", accuracy_score(y_test, y_pred))
    print("CART (Gini Index) Classification Report:\n", classification_report(y_test, y_pred))
