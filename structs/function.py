import numpy as np

# activation functions
# ReLU activation function
def relu(x):
    return np.maximum(0, x)

# ReLU activation function derivative
def relu_deriv(x):
    return (x > 0).astype(float)

# sigmoid activation function
def sigmoid(x):
    x = np.clip(x, -500, 500)  #  underflow, overflow 방지 
    return 1 / (1 + np.exp(-x))

# sigmoid activation function derivative
def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

# softmax activation function
def softmax(x):
    x = np.clip(x, -500, 500)  # underflow 방지
    x = x - np.max(x, axis=1, keepdims=True)  # 안정성 확보
    exp_x = np.exp(x)
    softmax = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return np.clip(softmax, 1e-6, 1 - 1e-6)  # log 안정성 확보

# softmax activation function derivative
def softmax_deriv(output):
    return output

# activation functions dictionary
activation_functions = {
    'ReLU': (relu, relu_deriv),
    'sigmoid': (sigmoid, sigmoid_deriv),
    'softmax': (softmax, softmax_deriv),
}

# loss functions
# Binary Cross-Entropy loss function
def BCE(y_true, y_pred):
    eps = 1e-8
    return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))

# Binary Cross-Entropy loss function derivative
def BCE_deriv(y_true, y_pred):
    return y_pred - y_true

# Cross-Entropy loss function
def CE(y_true, y_pred):
    eps = 1e-6
    y_pred = np.clip(y_pred, eps, 1 - eps) # underflow 방지
    log_y = np.log(y_pred)
    return -np.mean(np.sum(y_true * log_y, axis=1))

# Cross-Entropy loss function derivative
def CE_deriv(y_true, y_pred):
    return y_pred - y_true 

# loss functions dictionary
loss_functions = {
    'BCE': (BCE, BCE_deriv),
    'CE': (CE, CE_deriv)
}