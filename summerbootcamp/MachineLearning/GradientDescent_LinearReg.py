
import numpy as np
import pandas as pd

## Simple Linear Regression with Gradient Descent from Scratch


#Cost function
def cost_function(Y, b, w, X):
    m = len(Y)
    sse = 0
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) ** 2
    mse = sse / m
    return mse

#update_weights
def update_weights(Y, b, w, X, learning_rate):
    m = len(Y)
    b_deriv_sum = 0
    w_deriv_sum = 0
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat - y) * X[i]
    new_b = b - (learning_rate * 1/m * b_deriv_sum)
    new_w = w - (learning_rate * 1/m * w_deriv_sum)
    return new_b, new_w


#train func
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):
    print("Starting gradient descent at n = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                            cost_function(Y, initial_b, initial_w, X)))
    b = initial_b
    w = initial_w
    cost_history = []
    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)
        if i % 100 == 0:
            print("iter={:d} b={:.2f} w={:.4f} mse={:.4}".format(i, b, w, mse))
    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_history, b, w


df = pd.read_csv("advertising.csv")


X = df["radio"]
Y = df["sales"]

#hyperparameters
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 10000

train(Y, initial_b, initial_w, X, learning_rate, num_iters)