import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split

X, y = sklearn.datasets.make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X = np.array(X)
y = np.array(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)



class MyLogisticRegression:
    def __init__(self, learning_rate, iterations, b):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.b = b
    
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def loss(self, y, y_pred):
        return -np.mean(y*np.log(y_pred + 1e-15) + (1-y)*np.log(1-y_pred + 1e-15))
    
    def fit(self, X, y):
        w = np.zeros(X.shape[1])
        b = self.b
        for _ in range(self.iterations):
            y_pred = self.sigmoid(X@w + b)
            error = self.loss(y, y_pred)
            dw = (1/X.shape[0])*(X.T @ (y_pred - y))
            db = (1/X.shape[0])*np.sum(y_pred - y)
            w = w - self.learning_rate*dw
            b = b - self.learning_rate*db
        self.w, self.b = w, b

    def predict_probabillity(self, X):
        return self.sigmoid(X@self.w + self.b)
    
    def predict(self, X):
        group = self.predict_probabillity(X)
        return (group >= 0.5).astype(int)
    

model = MyLogisticRegression(learning_rate=0.01, iterations=1000, b=0.01)
model.fit(X_train,y_train)


y_pred = model.predict(X_test)

accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)

