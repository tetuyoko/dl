from common.functions import *

class SoftMaxWithLoss:
    def __init__(self):
        self.loss = None #損失
        self.y = None #softmaxの出力
        self.x = None #教師データ one-hot vector

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backword(self, dout=1):
        batch_size = self.t.shape(0)
        dx = (self.y - self.t) / batch_size
        return dx
