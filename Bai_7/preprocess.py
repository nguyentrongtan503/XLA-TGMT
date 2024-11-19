import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_iris_data():
    # Load the dataset
    data = pd.read_csv("Iriss.csv")
    
    # Separate features and target labels (use 'Species' as the correct column name)
    X = data.drop('Species', axis=1).values
    y = data['Species'].values
    
    # Encode labels as integers if they are strings
    labels, y = np.unique(y, return_inverse=True)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test
