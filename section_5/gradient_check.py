import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# データの読込
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grand_numerical = network.numerical_gradient(x_batch, t_batch)
grand_backprop = network.gradient(x_batch, t_batch)

#各重みの絶対誤差の平均を求める
for key in grand_numerical.keys():
    diff = np.average(np.abs(grand_backprop[key] - grand_numerical[key]))
    print(key + ":" + str(diff))
