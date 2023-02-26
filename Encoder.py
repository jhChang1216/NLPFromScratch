import numpy as np
from github_source.common.time_layers import TimeEmbedding, TimeLSTM

class Encoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D)/100).astype('f')
        lstm_Wx = (rn(D, 4*H)/np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4*H)/np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4*H).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=False)

        self.params = self.embed.params + self.lstm.params
        self.grads = self.embed.grads + self.lstm.grads
        self.hs = None

    def forward(self, xs):
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        self.hs = hs
        #lstm이 출력한 은닉상태 h들의 집합
        return hs[:,-1,:] #Decoder에 전달할 가장 마지막 은닉상태 h를 반환

    def backward(self, dh):
        dhs = np.zeros_like(self.hs)
        dhs[:, -1, :] = dh
        #Decoder에서 전파된 dh를 Encoder의 가장 마지막 시점의 역전파로 받아들임

        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)
        return dout