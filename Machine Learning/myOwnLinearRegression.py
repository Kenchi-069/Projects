import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


x = np.array([1, 2, 3, 4, 5])
y = 2 * x + np.random.randn(5) 


x = np.linspace(0, 10, 100)
y = 2 * x + np.random.randn(100) * 2  # noise with std dev 2

X = x
Y = y


m = 0
b = 0

def hypothesis(X, m, b):
    return m * X + b

def cost_function(X, Y, m, b):
    return np.mean((hypothesis(X, m, b) - Y) ** 2)

def gradient_descent(X, Y, m, b, iterations = 1000, learning_rate = 0.01):
    for i in range(iterations):
        m_gradient = -2 * np.mean(X * (Y - hypothesis(X, m, b)))
        b_gradient = -2 * np.mean(Y - hypothesis(X, m, b))
        m -= learning_rate * m_gradient
        b -= learning_rate * b_gradient
    return m, b

m,b = gradient_descent(X, Y, m, b, iterations = 10000, learning_rate = 0.001)
print("Slope (m):", m)
print("Intercept (b):", b)
plt.scatter(X, Y, color='blue', label='Data points')
plt.plot(X, hypothesis(X, m, b), color='red', label='Regression line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression using Gradient Descent')
plt.legend()
plt.show()
