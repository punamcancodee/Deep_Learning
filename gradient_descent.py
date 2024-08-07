# -*- coding: utf-8 -*-
"""Gradient_descent.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ShCXERhdstARnCQqvc4wNRCiz_OusxS9

**Gradient descent is an optimization algorithm commonly used in machine learning and deep learning to minimize a function by iteratively moving towards the minimum value of the function. In the context of training neural networks, gradient descent is used to minimize the loss function, which measures how well the model's predictions match the actual data.**
Here's a step-by-step explanation of how gradient descent works:

- Initialization: Start with an initial set of parameters (weights and biases for a neural network), often set randomly.

- Compute the Loss: Calculate the loss function, which measures the error between the model's predictions and the actual target values.

- Calculate Gradients: Compute the gradient (partial derivatives) of the loss function with respect to each parameter. This tells us the direction and rate of change of the loss with respect to each parameter.

- Update Parameters: Adjust the parameters in the direction opposite to the gradient to minimize the loss. The size of the steps taken in this direction is controlled by a parameter called the learning rate.

- Iterate: Repeat steps 2-4 for a number of iterations or until the loss converges to a minimum value.
"""

import numpy as np
import matplotlib.pyplot as plt

# Generate some synthetic data for linear regression
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Initialize parameters (weights)
theta = np.random.randn(2, 1)

def compute_cost(X_b, y, theta):
    m = len(y)
    cost = (1/2*m) * np.sum((X_b.dot(theta) - y)**2)
    return cost

def compute_gradient(X_b, y, theta):
    m = len(y)
    gradients = (1/m) * X_b.T.dot(X_b.dot(theta) - y)
    return gradients

def gradient_descent(X_b, y, theta, learning_rate, n_iterations):
    for iteration in range(n_iterations):
        gradients = compute_gradient(X_b, y, theta)
        theta = theta - learning_rate * gradients
    return theta

# Add bias term (x0 = 1) to each instance
X_b = np.c_[np.ones((100, 1)), X]

# Set hyperparameters
learning_rate = 0.1
n_iterations = 1000

# Perform gradient descent
theta = gradient_descent(X_b, y, theta, learning_rate, n_iterations)
print(f"Estimated parameters: {theta.ravel()}")

# Plot the data and the fitted line
plt.plot(X, y, "b.")
plt.plot(X, X_b.dot(theta), "r-", linewidth=2, label="Predictions")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()