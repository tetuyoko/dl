import numpy as np

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) #prevent overflow
    y = exp_a / np.sum(exp_a)
    return y

a = np.array([0.3, 2.9, 4.0])
a = np.array([1010, 1000, 900])
y = softmax(a)

print(y)
print(np.sum(y))
