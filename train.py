import numpy as np
from mindnlp.dataset.text_generation import LCSTS
from mindnlp.transforms import BasicTokenizer
import mindspore
from mindspore import nn, ops
from mindspore.dataset import MindDataset

from LCSTS_process import LCSTS_process
import utils
import models


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

config = {
    "hidden_size": 512,
    "selfatt": True,
    "attention": 'luong_gate',
    "cell": 'lstm',
    "emb_size": 512,
    "enc_num_layers": 3,
    "dropout": 0.0,
    "bidirectional": True,
    "swish": True,
    "dec_num_layers": 3,
    "tgt_vocab_size": 512*2,
    "shared_vocab": True,
    "max_time_step": 50,
    "src_vocab_size": 512*2,
    "max_split": 0,
    "schesamp": False
}
config = AttrDict(config)

def train_model(model, dataset, optim, epoch):

    # define forward function
    def forward_fn(src, lengths, dec, targets):
        if config.schesamp:
            if epoch > 8:
                e = epoch - 8
                loss, outputs = model(src, lengths, dec, targets, teacher_ratio=0.9**e)
            else:
                loss, outputs = model(src, lengths, dec, targets)
        else:
            loss, outputs = model(src, lengths, dec, targets)
        # pred = outputs.max(2)[1]
        targets = targets.T
        # num_correct = pred.eq(targets).masked_select(targets.ne(utils.PAD)).sum().item()
        num_total = targets.ne(utils.PAD).sum().item()
        if config.max_split == 0:
            loss = ops.sum(loss) / num_total
        return loss, outputs

    # Get gradient function
    grad_fn = mindspore.value_and_grad(forward_fn, None, optim.parameters, has_aux=True)

    # Define function of one-step training
    def train_step(src, lengths, dec, targets):
        (loss, output), grads = grad_fn(src, lengths, dec, targets)
        loss = ops.depend(loss, optim(grads))
        return loss, output

    model.set_train()

    # training steps
    # for data in dataset
    print("train starts")
    total_loss = []
    batch_num = dataset.get_batch_size
    for batch, (src, tgt, seq_length) in enumerate(dataset.create_tuple_iterator()):
        seq_length = mindspore.Tensor(seq_length, dtype=mindspore.int32)
        lengths, indices = ops.sort(seq_length, axis=0, descending=True)
        src = ops.index_select(src, axis=0, index=indices)
        tgt = ops.index_select(tgt, axis=0, index=indices)
        dec = tgt[:, :-1]
        targets = tgt[:, 1:]
        
        # call function: train_step()
        loss, output = train_step(src, lengths, dec, targets)
        total_loss.append(loss)
        print(f"--- step: {batch} / {batch_num}, loss: {loss} ---")
    
    # update lr

    return mindspore.Tensor(total_loss).sum()

if __name__ == "__main__":
    # load dataset
    
    dataset_path = "./dataset"
    split = ('train', 'dev')
    dataset_train, dataset_test = LCSTS(dataset_path, split)

    dataset_train, dataset_test, dataset_valid, vocab = LCSTS_process(dataset_train, dataset_test, tokenizer=BasicTokenizer())

    dataset_train = dataset_train.batch(8, drop_remainder=True)

    model = models.seq2seq(config, use_attention=True)
    optim = nn.Adam(model.trainable_params(), learning_rate=0.001)
    for epoch in range(2):
        print(f"----------- epoch: {epoch} -----------")
        loss = train_model(model, dataset_train, optim, epoch)
        print(f"epoch: {epoch} loss: {loss}")

