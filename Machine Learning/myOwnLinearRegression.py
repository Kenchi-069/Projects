import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
X= np.array(x)
Y = np.array(y)

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

m,b = gradient_descent(X, Y, m, b, iterations = 10000, learning_rate = 0.05)
print("Slope (m):", m)
print("Intercept (b):", b)
plt.scatter(X, Y, color='blue', label='Data points')
plt.plot(X, hypothesis(X, m, b), color='red', label='Regression line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression using Gradient Descent')
plt.legend()
plt.show()
