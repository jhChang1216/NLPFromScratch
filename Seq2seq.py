from Encoder import Encoder
from Decoder import Decoder
from common.time_layers import TimeSoftmaxWithLoss

class Seq2seq:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = Decoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

    def forward(self, xs, ts):
        #Seq2seq 모델을 훈련시키기 위한 forward
        decoder_xs, decoder_ts = ts[:, :-1], ts[:,1:]
        #Decoder의 target은 input보다 한발짝 늦는 상태

        h = self.encoder.forward(xs)
        #Encoder에서 출력한 마지막 은닉상태 h를 Decoder에 전달
        score = self.decoder.forward(decoder_xs, h)
        loss = self.softmax.forward(score, decoder_ts)
        return loss

    def backward(self, dout = 1):
        dout = self.softmax.backward(dout)
        dh = self.decoder.backward(dout)
        dout = self.encoder.backward(dh)
        return dout

    def generate(self, xs, start_id, sample_size):
        h = self.encoder.forward(xs)
        sampled = self.decoder.generate(h, start_id, sample_size)
        return sampled


