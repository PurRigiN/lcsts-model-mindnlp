"""attention"""
import math
from mindspore import nn, ops


class bahdanau_attention(nn.Cell):

    def __init__(self, hidden_size, emb_size):
        super().__init__()
        self.linear_encoder = nn.Dense(hidden_size, hidden_size)
        self.linear_decoder = nn.Dense(hidden_size, hidden_size)
        self.linear_v = nn.Dense(hidden_size, 1)
        self.linear_r = nn.Dense(hidden_size*2+emb_size, hidden_size*2)
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.softmax = nn.Softmax(axis=1)
        self.tanh = nn.Tanh()

    def init_context(self, context):
        self.context = context.transpose(1, 0, 2)

    def construct(self, h, x):
        gamma_encoder = self.linear_encoder(self.context)           # batch * time * size
        gamma_decoder = self.linear_decoder(h).unsqueeze(1)    # batch * 1 * size
        weights = self.linear_v(self.tanh(gamma_encoder+gamma_decoder)).squeeze(2)   # batch * time
        weights = self.softmax(weights)   # batch * time
        c_t = ops.bmm(weights.unsqueeze(1), self.context).squeeze(1) # batch * size
        r_t = self.linear_r(ops.concat([c_t, h, x], axis=1))
        output = r_t.view(-1, self.hidden_size, 2).max(2)[0]

        return output, weights
    
class luong_gate_attention(nn.Cell):
    
    """
    - prob(float): probability of an element to be zeroed, opposite of mindspore
    """
    def __init__(self, hidden_size, emb_size, prob=0.1):
        super(luong_gate_attention, self).__init__()
        self.hidden_size, self.emb_size = hidden_size, emb_size
        self.linear_enc = nn.SequentialCell(nn.Dense(hidden_size, hidden_size), nn.SeLU(), nn.Dropout(1-prob), 
                                        nn.Dense(hidden_size, hidden_size), nn.SeLU(), nn.Dropout(1-prob))
        self.linear_in = nn.SequentialCell(nn.Dense(hidden_size, hidden_size), nn.SeLU(), nn.Dropout(1-prob), 
                                       nn.Dense(hidden_size, hidden_size), nn.SeLU(), nn.Dropout(1-prob))
        self.linear_out = nn.SequentialCell(nn.Dense(2*hidden_size, hidden_size), nn.SeLU(), nn.Dropout(1-prob), 
                                        nn.Dense(hidden_size, hidden_size), nn.SeLU(), nn.Dropout(1-prob))
        self.softmax = nn.Softmax(axis=-1)

    def init_context(self, context):
        self.context = context.transpose(1, 0, 2)

    def construct(self, h, selfatt=False):
        if selfatt:
            gamma_enc = self.linear_enc(self.context) # Batch_size * Length * Hidden_size
            gamma_h = gamma_enc.transpose(0, 2, 1) # Batch_size * Hidden_size * Length
            weights = ops.bmm(gamma_enc, gamma_h) # Batch_size * Length * Length
            weights = self.softmax(weights/math.sqrt(512))
            c_t = ops.bmm(weights, gamma_enc) # Batch_size * Length * Hidden_size
            output = self.linear_out(ops.concat([gamma_enc, c_t], 2)) + self.context
            output = output.transpose(1, 0, 2) # Length * Batch_size * Hidden_size
        else:
            gamma_h = self.linear_in(h).unsqueeze(2)
            weights = ops.bmm(self.context, gamma_h).squeeze(2)
            weights = self.softmax(weights)
            c_t = ops.bmm(weights.unsqueeze(1), self.context).squeeze(1)
            output = self.linear_out(ops.cat([h, c_t], 1))

        return output, weights
