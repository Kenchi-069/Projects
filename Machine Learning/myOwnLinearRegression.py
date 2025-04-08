import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MyLinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.m = 0
        self.b = 0

    def fit(self, X, Y):
        for _ in range(self.iterations):
            Y_pred = self.m * X + self.b
            D_m = -2 * np.mean(X * (Y - Y_pred))
            D_b = -2 * np.mean(Y - Y_pred)
            self.m -= self.learning_rate * D_m
            self.b -= self.learning_rate * D_b
    
    def predict(self, X):
        return self.m * X + self.b
    
    def r2_score(self, Y, Y_pred):
        ss_total = np.sum((Y - np.mean(Y)) ** 2)
        ss_residual = np.sum((Y - Y_pred) ** 2)
        return 1 - (ss_residual / ss_total)


x = np.array([1, 2, 3, 4, 5])
y = 2 * x + np.random.randn(5) 
x = np.linspace(0, 10, 100)
y = 2 * x + np.random.randn(100) * 2  

X = x
Y = y

model = MyLinearRegression(learning_rate=0.01, iterations=2000)
model.fit(X, Y) 

print("Slope (m):", model.m)
print("Intercept (b):", model.b)
print("R^2 score:", model.r2_score(Y, model.predict(X)))
plt.scatter(X, Y, color='blue', label='Data points')
plt.plot(X, model.predict(X), color='red', label='Regression line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression Fit')
plt.legend()
plt.show()

y_pred = model.predict(X)

plt.scatter(Y, y_pred, color='green')
plt.plot([min(Y), max(Y)], [min(Y), max(Y)], color='red', linestyle='--')  # Ideal line
plt.xlabel("Actual Y")
plt.ylabel("Predicted Y")
plt.title("Predicted vs Actual")
plt.grid(True)
plt.show()


residuals = Y - y_pred

plt.scatter(X, residuals, color='purple')
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel("X")
plt.ylabel("Residuals")
plt.title("Residuals vs X")
plt.grid(True)
plt.show()
