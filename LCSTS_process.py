import numpy as np
from mindnlp.dataset.text_generation import LCSTS
from mindnlp.transforms import BasicTokenizer
from mindspore.dataset import GeneratorDataset, text, transforms

import utils

def LCSTS_process(dataset_train,
                  dataset_test,
                  tokenizer,
                  max_time_step=50,
                  pad_word='<blank>',
                  unk_word='<unk>',
                  bos_word='<s>',
                  eos_word='</s>'
                  ):

    dataset_train, dataset_valid = dataset_train.split([dataset_train.get_dataset_size() - 1106, 1106])
    print(dataset_train.get_dataset_size())
    print(dataset_test.get_dataset_size())
    print(dataset_valid.get_dataset_size())

    # 分词
    # 截断
    truncate_src_op = text.Truncate(max_time_step)
    truncate_tgt_op = text.Truncate(max_time_step-2)
    dataset_train = dataset_train.map(input_columns=["source"], operations=[tokenizer, truncate_src_op])
    dataset_train = dataset_train.map(input_columns=["target"], operations=[tokenizer, truncate_tgt_op])
    dataset_test = dataset_test.map(input_columns=["source"], operations=[tokenizer, truncate_src_op])
    dataset_test = dataset_test.map(input_columns=["target"], operations=[tokenizer, truncate_tgt_op])
    dataset_valid = dataset_valid.map(input_columns=["source"], operations=[tokenizer, truncate_src_op])
    dataset_valid = dataset_valid.map(input_columns=["target"], operations=[tokenizer, truncate_tgt_op])

    # 统计seq_length
    print("counting seq_length...")
    class TmpDataset:
        """ a Dataset for seq_length column """
        def __init__(self, dataset):
            self._dataset = dataset
            self._seq_length = []
            self._load()

        def _load(self):
            for data in self._dataset.create_dict_iterator():
                self._seq_length.append(len(data["source"]))

        def __getitem__(self, index):
            return self._seq_length[index]

        def __len__(self):
            return len(self._seq_length)
    dataset_tmp = GeneratorDataset(TmpDataset(dataset_train), ["seq_length"],shuffle=False)
    dataset_train = dataset_train.zip(dataset_tmp)
    dataset_tmp = GeneratorDataset(TmpDataset(dataset_test), ["seq_length"],shuffle=False)
    dataset_test = dataset_test.zip(dataset_tmp)
    dataset_tmp = GeneratorDataset(TmpDataset(dataset_valid), ["seq_length"],shuffle=False)
    dataset_valid = dataset_valid.zip(dataset_tmp)
    print("complete")

    # 用训练集的两列做词表
    print("building vocab...")
    vocab = text.Vocab.from_dataset(dataset_train, columns=["source", "target"], special_tokens=[pad_word,unk_word,bos_word,eos_word])
    print("complete")
    # target前后加<s>和</s>
    add_begin_op = text.AddToken(bos_word, begin=True)
    add_end_op = text.AddToken(eos_word, begin=False)

    # 用词表做编码
    lookup_op = text.Lookup(vocab, unknown_token=unk_word)

    # pad
    pad_value = vocab.tokens_to_ids(pad_word)
    pad_op = transforms.PadEnd([max_time_step], pad_value)

    dataset_train = dataset_train.map(operations=[lookup_op, pad_op], input_columns=["source"])
    dataset_train = dataset_train.map(operations=[add_begin_op, add_end_op, lookup_op, pad_op], input_columns=["target"])
    dataset_test = dataset_test.map(operations=[lookup_op, pad_op], input_columns=["source"])
    dataset_test = dataset_test.map(operations=[add_begin_op, add_end_op, lookup_op, pad_op], input_columns=["target"])
    dataset_valid = dataset_valid.map(operations=[lookup_op, pad_op], input_columns=["source"])
    dataset_valid = dataset_valid.map(operations=[add_begin_op, add_end_op, lookup_op, pad_op], input_columns=["target"])


    # 保存于path中
    # dataset_train.save(dataset_train_path)
    # dataset_test.save(dataset_test_path)
    # dataset_valid.save(dataset_valid_path)

    # vocab_dict = vocab.vocab()
    # np.save(vocab_path, dict)

    return dataset_train, dataset_test, dataset_valid, vocab

if __name__ == "__main__":
    # process dataset
    dataset_path = "./dataset"
    split = ('train', 'dev')
    dataset_train, dataset_test = LCSTS(dataset_path, split)

    LCSTS_process(dataset_train, dataset_test, tokenizer=BasicTokenizer())

