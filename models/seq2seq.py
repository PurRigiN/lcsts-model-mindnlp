import random
import models
import utils

from mindspore import nn, ops

class seq2seq(nn.Cell):

    def __init__(self, config, use_attention=True, encoder=None, decoder=None):
        super(seq2seq, self).__init__()

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = models.RnnEncoder(config)
        tgt_embedding = self.encoder.embedding if config.shared_vocab else None
        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = models.RnnDecoder(config, embedding=tgt_embedding, use_attention=use_attention)
        self.log_softmax = nn.LogSoftmax(axis=-1)

        self.config = config
        self.criterion = nn.CrossEntropyLoss(ignore_index=utils.PAD, reduction='none')

    def compute_loss(self, scores, targets):
        scores = scores.view(-1, scores.shape[2])
        loss = self.criterion(scores, targets.view(-1))
        return loss

    def construct(self, src, src_len, dec, targets, teacher_ratio=1.0):
        src = src.T       # 这里推测，转换后是一个tensor，三个都是
        dec = dec.T
        targets = targets.T
        teacher = random.random() < teacher_ratio

        contexts, state = self.encoder(src, src_len)

        if self.decoder.attention is not None:
            self.decoder.attention.init_context(context=contexts)
        outputs = []
        if teacher:
            for input in dec.split(1):
                output, state, attn_weights = self.decoder(input.squeeze(0), state)
                outputs.append(output)
            outputs = ops.stack(outputs)
        else:
            inputs = [dec.split(1)[0].squeeze(0)]
            for i, _ in enumerate(dec.split(1)):
                output, state, attn_weights = self.decoder(inputs[i], state)
                predicted = output.max(1)[1]
                inputs += [predicted]
                outputs.append(output)
            outputs = ops.stack(outputs)

        loss = self.compute_loss(outputs, targets)
        return loss, outputs

    def sample(self, src, src_len):

        lengths, indices = ops.sort(src_len, axis=0, descending=True)
        _, reverse_indices = ops.sort(indices)
        src = ops.index_select(src, axis=0, index=indices)
        bos = ops.ones(src.shape[0]).long().Fill(utils.BOS)
        src = src.T

        contexts, state = self.encoder(src, lengths)

        if self.decoder.attention is not None:
            self.decoder.attention.init_context(context=contexts)
        inputs, outputs, attn_matrix = [bos], [], []
        for i in range(self.config.max_time_step):
            output, state, attn_weights = self.decoder(inputs[i], state)
            predicted = output.max(1)[1]
            inputs += [predicted]
            outputs += [predicted]
            attn_matrix += [attn_weights]

        outputs = ops.stack(outputs)
        sample_ids = ops.index_select(outputs, axis=1, index=reverse_indices).T.asnumpy().tolist()

        if self.decoder.attention is not None:
            attn_matrix = ops.stack(attn_matrix)
            alignments = attn_matrix.max(2)[1]
            alignments = ops.index_select(alignments, axis=1, index=reverse_indices).T.asnumpy().tolist()
        else:
            alignments = None

        return sample_ids, alignments

    # def beam_sample(self, src, src_len, beam_size=1, eval_=False):

    #     # (1) Run the encoder on the src.

    #     lengths, indices = torch.sort(src_len, dim=0, descending=True)
    #     _, ind = torch.sort(indices)
    #     src = torch.index_select(src, dim=0, index=indices)
    #     src = src.t()
    #     batch_size = src.size(1)
    #     contexts, encState = self.encoder(src, lengths.tolist())

    #     #  (1b) Initialize for the decoder.
    #     def var(a):
    #         return torch.tensor(a, requires_grad=False)

    #     def rvar(a):
    #         return var(a.repeat(1, beam_size, 1))

    #     def bottle(m):
    #         return m.view(batch_size * beam_size, -1)

    #     def unbottle(m):
    #         return m.view(beam_size, batch_size, -1)

    #     # Repeat everything beam_size times.
    #     # contexts = rvar(contexts.data)
    #     contexts = rvar(contexts)

    #     if self.config.cell == 'lstm':
    #         decState = (rvar(encState[0]), rvar(encState[1]))
    #     else:
    #         decState = rvar(encState)

    #     beam = [models.Beam(beam_size, n_best=1,
    #                       cuda=self.use_cuda, length_norm=self.config.length_norm)
    #             for __ in range(batch_size)]
    #     if self.decoder.attention is not None:
    #         self.decoder.attention.init_context(contexts)

    #     # (2) run the decoder to generate sentences, using beam search.

    #     for i in range(self.config.max_time_step):

    #         if all((b.done() for b in beam)):
    #             break

    #         # Construct batch x beam_size nxt words.
    #         # Get all the pending current beam words and arrange for forward.
    #         inp = var(torch.stack([b.getCurrentState() for b in beam])
    #                   .t().contiguous().view(-1))

    #         # Run one step.
    #         output, decState, attn = self.decoder(inp, decState)
    #         # decOut: beam x rnn_size

    #         # (b) Compute a vector of batch*beam word scores.
    #         output = unbottle(self.log_softmax(output))
    #         attn = unbottle(attn)
    #         # beam x tgt_vocab

    #         # (c) Advance each beam.
    #         # update state
    #         for j, b in enumerate(beam):
    #             b.advance(output[:, j], attn[:, j])
    #             if self.config.cell == 'lstm':
    #                 b.beam_update(decState, j)
    #             else:
    #                 b.beam_update_gru(decState, j)

    #     # (3) Package everything up.
    #     allHyps, allScores, allAttn = [], [], []
    #     if eval_:
    #         allWeight = []

    #     # for j in ind.data:
    #     for j in ind:
    #         b = beam[j]
    #         n_best = 1
    #         scores, ks = b.sortFinished(minimum=n_best)
    #         hyps, attn = [], []
    #         if eval_:
    #             weight = []
    #         for i, (times, k) in enumerate(ks[:n_best]):
    #             hyp, att = b.getHyp(times, k)
    #             hyps.append(hyp)
    #             attn.append(att.max(1)[1])
    #             if eval_:
    #                 weight.append(att)
    #         allHyps.append(hyps[0])
    #         allScores.append(scores[0])
    #         allAttn.append(attn[0])
    #         if eval_:
    #             allWeight.append(weight[0])
        
    #     if eval_:
    #         return allHyps, allAttn, allWeight

    #     return allHyps, allAttn
