import numpy as np

def softmax(a):
    exp_a = np.exp(a)
    y = exp_a / np.sum(exp_a)
    return y

a = np.array([0.3, 2.9, 4.0])
y = softmax(a)

print(y)
