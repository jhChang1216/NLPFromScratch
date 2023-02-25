import numpy as np
from RNN import RNN

class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful =False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.dh = None, None
        self.stateful = stateful

    def set_state(self, h):
        # 이전 층에서 h를 이어 받을 경우
        self.h = h

    def reset_state(self):
        self.h = None

    def forward(self, xs): #xs를 입력받아 hs를 반환
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        #배치사이즈, time사이즈, 임베딩사이즈
        D, H = Wx.shape
        #임베딩사이즈, 히든상태사이즈

        self.layers = []
        hs = np.empty((N, T, D), dtype='f')

        if not self.stateful or self.h is None:
            #stateful이 false거나 h가 None일 경우(truncated)
            self.h = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = RNN(*self.params)
            # 파라미터를 주입한 RNN 생성
            self.h = layer.forward(xs[:, t, :], self.h)
            # t 시점의 x를 입력받아서(배치처리) 새로운 t 시점의 h를 생성
            hs[:, t, :] = self.h
            # 그렇게 생성한 h를 hs의 t시점 요소에 저장
            self.layers.append(layer)
            # t 시점을 처리하는 RNN을 layer에 쌓는다
        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape

        dxs = np.empty((N, T, D), dtype='f')
        dh = 0
        grads = [0,0,0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            # 마지막 t 시점의 RNN부터
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            #순전파 시 h_t를 2개로 분기하므로 역전파에서는 흘러온 dh를 합쳐서 새로운 dx, dh 계산
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad
                #각각 Wx, Wh, b의 grad를 누적(T 동안의)

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
            #역전파를 거치면서 누적된 grad(Wx, Wh, b)를 저장
        self.dh = dh
        #가장 최초 시점의 dh를 저장?

        return dxs

