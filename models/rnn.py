"""rnn"""
import numpy as np

from mindspore import nn, ops, Tensor

import models
class RnnEncoder(nn.Cell):
    """
    Rnn Encoder
    """
    def __init__(self, config, embedding=None):
        super().__init__()

        self.embedding = embedding if embedding is not None else nn.Embedding(config.src_vocab_size, config.emb_size)
        self.hidden_size = config.hidden_size
        self.config = config

        self.sw1 = nn.SequentialCell(nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=1, padding=0, pad_mode='valid'), nn.BatchNorm1d(config.hidden_size), nn.ReLU())
        self.sw3 = nn.SequentialCell(nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=1, padding=0, pad_mode='valid'), nn.ReLU(), nn.BatchNorm1d(config.hidden_size),
                                 nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=3, padding=1, pad_mode='pad'), nn.ReLU(), nn.BatchNorm1d(config.hidden_size))
        self.sw33 = nn.SequentialCell(nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=1, padding=0), nn.ReLU(), nn.BatchNorm1d(config.hidden_size),
                                  nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=3, padding=1, pad_mode='pad'), nn.ReLU(), nn.BatchNorm1d(config.hidden_size),
                                  nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size=3, padding=1, pad_mode='pad'), nn.ReLU(), nn.BatchNorm1d(config.hidden_size))
        self.linear = nn.SequentialCell(nn.Dense(2*config.hidden_size, 2*config.hidden_size), nn.GLU(), nn.Dropout(1-config.dropout))
        self.filter_linear = nn.Dense(3*config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        if config.selfatt:
            if config.attention == 'None':
                self.attention = None
            elif config.attention == 'bahdanau':
                self.attention = models.bahdanau_attention(config.hidden_size, config.emb_size)
            # elif config.attention == 'luong':
            #     self.attention = models.luong_attention(config.hidden_size, config.emb_size, config.pool_size)
            elif config.attention == 'luong_gate':
                self.attention = models.luong_gate_attention(config.hidden_size, config.emb_size)
        
        if config.cell == 'gru':
            self.rnn = nn.GRU(input_size=config.emb_size, hidden_size=config.hidden_size,
                              num_layers=config.enc_num_layers, dropout=config.dropout,
                              bidirectional=config.bidirectional)
        else:
            self.rnn = nn.LSTM(input_size=config.emb_size, hidden_size=config.hidden_size,
                               num_layers=config.enc_num_layers, dropout=config.dropout,
                               bidirectional=config.bidirectional)

    def construct(self, inputs, lengths):       # input: [time, batch] dtype = int
        embs = self.embedding(inputs)           # embs: [time, batch, emb_size] dtype = float
        outputs, state = self.rnn(embs, seq_length=lengths) # outputs: [time, batch, D*hidden_size]
        # outputs = outputs.transpose(1, 0, 2)                # outputs: [batch, time, D*hidden_size]
        if self.config.bidirectional:
            if self.config.swish:
                outputs = self.linear(outputs)
            else:
                outputs = outputs[:,:,:self.config.hidden_size] + outputs[:,:,self.config.hidden_size:]
        if self.config.swish:
            outputs = outputs.transpose(1, 2, 0)
            conv1 = self.sw1(outputs)
            conv3 = self.sw3(outputs)
            conv33 = self.sw33(outputs)
            conv = ops.concat((conv1, conv3, conv33), 1)
            conv = self.filter_linear(conv.transpose(0,2,1))
            if self.config.selfatt:
                conv = conv.transpose(1,0,2)
                outputs = outputs.transpose(2,0,1)
            else:
                gate = self.sigmoid(conv)
                outputs = outputs * gate.transpose(0,2,1)
                outputs = outputs.transpose(2,0,1)

        if self.config.selfatt:
            self.attention.init_context(context=conv)
            out_attn, weights = self.attention(conv, selfatt=True)
            gate = self.sigmoid(out_attn)
            outputs = outputs * gate

        if self.config.cell == 'gru':
            state = state[:self.config.dec_num_layers]
        else:
            state = (state[0][::2], state[1][::2])

        return outputs, state

class RnnDecoder(nn.Cell):

    def __init__(self, config, embedding=None, use_attention=True):
        super().__init__()
        
        self.embedding = embedding if embedding is not None else nn.Embedding(config.tgt_vocab_size, config.emb_size)

        input_size = config.emb_size

        if config.cell == 'gru':
            self.rnn = StackedGRU(input_size=input_size, hidden_size=config.hidden_size,
                                  num_layers=config.dec_num_layers, dropout=config.dropout)
        else:
            self.rnn = StackedLSTM(input_size=input_size, hidden_size=config.hidden_size,
                                   num_layers=config.dec_num_layers, dropout=config.dropout)

        self.linear = nn.Dense(config.hidden_size, config.tgt_vocab_size)
        self.linear_ = nn.Dense(config.hidden_size, config.hidden_size)
        self.sigmoid = nn.Sigmoid()
 
        if not use_attention or config.attention == 'None':
            self.attention = None
        elif config.attention == 'bahdanau':
            self.attention = models.bahdanau_attention(config.hidden_size, config.emb_size)
        # elif config.attention == 'luong':
        #     self.attention = models.luong_attention(config.hidden_size, config.emb_size, config.pool_size)
        elif config.attention == 'luong_gate':
            self.attention = models.luong_gate_attention(config.hidden_size, config.emb_size, prob=config.dropout)

        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(1-config.dropout)
        self.config = config

    def construct(self, input, state):
        embs = self.embedding(input)
        output, state = self.rnn(embs, state)
        if self.attention is not None:
            if self.config.attention == 'luong_gate':
                output, attn_weights = self.attention(output)
            else:
                output, attn_weights = self.attention(output, embs)
        else:
            attn_weights = None
        
        output = self.compute_score(output)

        return output, state, attn_weights

    def compute_score(self, hiddens):
        scores = self.linear(hiddens)
        return scores



class StackedLSTM(nn.Cell):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super().__init__()
        self.dropout = nn.Dropout(1-dropout)
        self.num_layers = num_layers
        self.layers = nn.CellList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size

    def construct(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = ops.stack(h_1)
        c_1 = ops.stack(c_1)

        return input, (h_1, c_1)


class StackedGRU(nn.Cell):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(1-dropout)
        self.num_layers = num_layers
        self.layers = nn.CellList()

        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, hidden_size))
            input_size = hidden_size

    def construct(self, input, hidden):
        h_0 = hidden
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, h_0[i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]

        h_1 = ops.stack(h_1)

        return input, h_1
