import numpy as np

class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)] #역전파 시 저장 용도
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx #추출하고자 하는 행들(index)
        out = W[idx]
        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0

        for i, word_id in enumerate(self.idx):
            dW[word_id] += dout[i]
        return None