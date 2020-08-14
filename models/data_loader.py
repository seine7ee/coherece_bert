import torch
from transformers import BertTokenizer
import json
from utils.logging_utils import logger
import os
import glob
import random
import gc
import argparse


def load_dataset(args, corpus_type, shuffle):
    assert corpus_type in ['train', 'test', 'valid']

    def _lazy_load_dataset(pt_file, corpus_type):
        # with open(json_file, 'r') as file:
        #     dataset = json.load(file)
        dataset = torch.load(pt_file, map_location=torch.device('cpu'))
        logger.info(
            "Loading {:s} dataset from {:s}, number of examples: {:d}".format(corpus_type, pt_file, len(dataset)))
        return dataset

    files = sorted(glob.glob(os.path.join(args.shard_tgt_root_path, "{:s}/*.pt".format(corpus_type))))
    if files:
        if shuffle:
            random.seed(66)
            random.shuffle(files)
        for file in files:
            yield _lazy_load_dataset(file, corpus_type)
    else:
        raise Exception("Only ONE dataset, check whether it right.")



class Batch(object):
    def __init__(self, batch_data, device='cuda', is_test=False):
        self.batch_data = batch_data
        self.device = device
        self.is_test = is_test
        self.batch_size = len(batch_data)

        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        input_ids, token_type_ids, attention_mask = map(self._pad, self.process())

        encode = {"input_ids": input_ids.to(device),
                  "token_type_ids": token_type_ids.to(device),
                  "attention_mask": attention_mask.to(device)}

        self.encode = encode
        self.labels = torch.LongTensor([data[1] for data in self.batch_data]).to(device)


    def process(self):
        input_ids = [data[0]['input_ids'][0] for data in self.batch_data]
        token_type_ids = [data[0]['token_type_ids'][0] for data in self.batch_data]
        attention_mask = [data[0]['attention_mask'][0] for data in self.batch_data]


        return input_ids, token_type_ids, attention_mask

    def _pad(self, _data, pad_id=0, width=128):
        # if width == -1:
        #     width = max(d.numel() for d in _data)
        padding_data = [d.numpy().tolist() + [pad_id] * (width - len(d)) for d in _data]
        return torch.tensor(padding_data)

        # 传给CrossEntropy的input_shape和target_shape
        #  - 如果input_shape : (batch_size, Category size)， Category size上是one hot
        #  - 那么target_shape: (batch_size) batch中的每一个都是gold label
        # label = [[s[1]] for s in batch_data]        # [[label], [label]]


    # def batch_tokenize(self):
    #     sent_a, sent_b, label = self.normalize(self.batch_data)
    #     # print(sent_a)
    #     encode = self.tokenizer(sent_a, sent_b, padding='longest', return_tensors='pt', is_pretokenized=True)
    #     labels = torch.LongTensor(label)
    #
    #     return encode, labels

    def __len__(self):
        return len(self.batch_data)


def current_element_number(example, count):
    encode, label = example
    input_ids = encode['input_ids']
    global max_n_tokens, max_size
    if count == 1:
        max_n_tokens, max_size = 0, 0

    max_n_tokens = max(max_n_tokens, input_ids[0].numel())
    max_size = max(max_n_tokens, max_size)
    return count * max_size






class DataLoaderBert(object):
    def __init__(self, datasets, batch_size, device='cuda', shuffle=False, is_test=False):
        self.datasets = datasets  # a generator
        self.batch_size = batch_size

        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test

        self.cur_iter = self._next_dataset_iterator(datasets)
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)

    def _next_dataset_iterator(self, dataset_iter):
        try:
            if hasattr(self, 'cur_dataset'):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()
            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        return DataIteratorBert(self.cur_dataset, self.batch_size,
                                self.device, self.is_test, self.shuffle)


class DataIteratorBert(object):
    def __init__(self, dataset, batch_size, device=None, is_test=False, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size

        self.device = device
        self.is_test = is_test
        self.shuffle = shuffle
        self.buffer_size = 300
        self.iterations = 0

        self._iterations_this_epoch = 0

    def dataset_(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        dataset_ = self.dataset
        return dataset_

    def preprocess(self, example):
        encode = example[0]

        label = example[1]

        return encode, label

    def batch_buffer(self, dataset, batch_size, buffer_size):
        buffer, current_size = [], 0
        for ex in dataset:
            ex = self.preprocess(ex)
            if ex is None:
                continue
            buffer.append(ex)
            current_size = current_element_number(ex, len(buffer))
            if current_size == batch_size * buffer_size:
                yield buffer
                buffer, current_size = [], 0
            elif current_size > batch_size * buffer_size:
                yield buffer[:-1]
                buffer, current_size = buffer[-1:], current_element_number(ex, 1)

        if buffer:
            yield buffer


    def batch(self, dataset, batch_size):

        minibatch, current_size = [], 0
        for ex in dataset:
            ex = self.preprocess(ex)   # return a tuple (encode, label)
            if ex is None:
                continue
            minibatch.append(ex)
            # current_size = current_element_number(ex, len(minibatch))
            current_size = len(minibatch)
            if current_size == batch_size:
                yield minibatch
                minibatch = []
            elif current_size > batch_size:
                yield minibatch[:-1]
                minibatch = minibatch[-1:]
        if minibatch:
            yield minibatch

    def create_batches(self):
        dataset = self.dataset_()
        for minibatch in self.batch(dataset, self.batch_size):
            yield minibatch


        # for buffer in self.batch_buffer(dataset, self.batch_size, self.buffer_size):
        #     minibatch = self.batch(buffer, self.batch_size)
        #
        #     minibatch = list(minibatch)
        #     if self.shuffle:
        #         random.shuffle(minibatch)
        #
        #     for batch in minibatch:
        #         if len(batch) == 0:
        #             continue
        #         else:
        #             yield batch

    def __iter__(self):
        while True:
            self.batches = self.create_batches()

            for idx, minibatch in enumerate(self.batches):
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(minibatch, self.device, self.is_test)

                yield batch
            return


# ============================ test module =====================================
import torch.nn as nn
import torch.distributed as dist
from torch.multiprocessing import Process

class ToyModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256,
                 in_features=256, filter_size=128,
                 out_features=2):
        super(ToyModel, self).__init__()
        self.embed = torch.nn.Embedding(vocab_size, embedding_dim=embedding_dim)
        self.net1 = nn.Linear(in_features=in_features, out_features=filter_size)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(filter_size, out_features=out_features)

    def forward(self, x):
        # x: (batch_size, seq_len)
        x = self.net2(self.relu(self.net1(self.embed(x))))
        return x[:, 0]

def train(args, train_steps, n_gpu, device_id, gpu_rank):
    print(torch.cuda.current_device())

    datasets = load_dataset(args, 'train', shuffle=True)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_iter = DataLoaderBert(datasets, batch_size=200, device='cuda', shuffle=True)

    model = ToyModel(tokenizer.vocab_size)
    loss = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.01)

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.manual_seed(666)

    model.to('cuda')

    for step in range(train_steps):
        for i, batch in enumerate(train_iter):
            if n_gpu==0 or (i % n_gpu == gpu_rank):
                optim.zero_grad()

                input = batch.encode['input_ids']
                label = batch.label

                output = model(input)
                # print(output.dtype)
                # print(output.size())

                l = loss(output, label)
                l.backward()
                if i % 5 == 0:
                    print("Training step {:d}, No.{:d} batch, loss is {:f}".format(step, i, l))

                if n_gpu > 1:
                    size = float(dist.get_world_size())
                    for param in model.parameters():
                        dist.all_reduce(param.grad.data)
                        param.grad.data /= size
                optim.step()


def is_master(gpu_ranks, device_id):
    return gpu_ranks[device_id] == 0


def multi_init(device_id, world_size, gpu_ranks):
    print(gpu_ranks)
    dist_init_method = 'tcp://localhost:10000'
    dist_world_size = world_size
    torch.distributed.init_process_group(
        backend='nccl', init_method=dist_init_method,
        world_size=dist_world_size, rank=gpu_ranks[device_id])
    gpu_rank = torch.distributed.get_rank()
    if not is_master(gpu_ranks, device_id):
    #     print('not master')
        logger.disabled = True

    return gpu_rank


def run(device_id, gpu_ranks, world_size, args):
    gpu_rank = multi_init(device_id, world_size, gpu_ranks)
    logger.info("GPU Rank: gpu_rank {:d}".format(gpu_rank))

    if gpu_rank != gpu_ranks[device_id]:
        raise AssertionError("An Error occured in Distributed intializaiton")
    n_gpu = world_size
    train(args, 10000, n_gpu, device_id, gpu_rank)







if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-shard_tgt_root_path', default="/sdc/xli/Datasets/cnn_daily/data_nsp/shard/pts", type=str)

    args = parser.parse_args()

    datasets = load_dataset(args, 'train', shuffle=True)


    train_iter = DataLoaderBert(datasets, batch_size=4, device='cuda', shuffle=True)

    # for i, batch in enumerate(train_iter):
    #     if i < 5:
    #         print("---------------------------------------------------------")
    #         print(batch.encode['input_ids'])
    #         print(batch.encode['token_type_ids'].device)
    #         print(batch.encode['attention_mask'].device)
    #         print(batch.labels.device)
    #     else:
    #         break

    from models.model_builder import Bert
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    model = Bert(False, '/sdc/xli/py/bert/models/bert_uncased')
    model.to(0)
    for i, batch in enumerate(train_iter):
        if i < 5:
            input_ids = batch.encode['input_ids']
            print(input_ids.device)
            # token_type_ids = batch.encode['token_type_ids']
            # attention_mask = batch.encode['attention_mask']
            outputs = model(**batch.encode)
            lhs = outputs[0]
            p_out = outputs[1]
            print(lhs.size())
            print(p_out.size())










