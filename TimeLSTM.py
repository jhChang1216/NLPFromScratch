import numpy as np
from LSTM import LSTM

class TimeLSTM:
    def __init__(self, Wx, Wh, b, stateful = False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None
        self.h, self.c = None, None
        self.dh = None
        self.stateful = stateful

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')
        #생성할 hs 틀 만듦

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
            #stateful 하지 않으면 이어받을 은닉상태 h 없으므로 새로 생성
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype='f')
            # stateful 하지 않으면 이어받을 기억셀 c 없으므로 새로 생성

        for t in range(T):
            layer = LSTM(*self.params)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            #한 time의 LSTM forward하고 h, c 갱신
            hs[:, t, :] = self.h
            self.layers.append(layer)
            #시점 t에서의 LSTM 이어붙임
        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype='f')
        #역전파할 dxs 틀 생성
        dh, dc = 0, 0

        grads = [0, 0, 0]
        for t in reversed(range(T)):
            #시간 거꾸로 내려오면서
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :]+dh, dc)
            #나중 시점의 LSTM부터 역전파
            dxs[:, t, :] = dx
            for i, grad in enumerate(grads):
                grads[i] += grad
                #Wx, Wh, b별로 gradient값 누적

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad

        self.dh = dh
        #가장 마지막으로 전파된 dh를 저장
        return dxs

    def set_state(self, h, c=None):
        self.h, self.c = h, c

    def reset_state(self):
        self.h, self.c = None, None




