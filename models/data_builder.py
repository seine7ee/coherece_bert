import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import json
from utils.logging_utils import logger
import os
import glob
import random
import gc
import argparse

# class cnndailyTGTDataset(Dataset):
#
#     def __init__(self, dir, mapping_file, ):
#         self.dir = dir
#         self.mapping_file = mapping_file
#
#     def __getitem__(self, item):
#         pass
#
#     def __len__(self):
#         pass

def load_dataset(args, corpus_type, shuffle):
    assert corpus_type in ['train', 'test']

    def _lazy_load_dataset(json_file, corpus_type):
        with open(json_file, 'r') as file:
            dataset = json.load(file)
        logger.info(
            "Loading {:s} dataset from {:s}, number of examples: {:d}".format(corpus_type, json_file, len(dataset)))
        return dataset

    files = sorted(glob.glob(os.path.join(args.shard_tgt_root_path, "{:s}/*.json".format(corpus_type))))
    if files:
        if shuffle:
            random.seed(66)
            random.shuffle(files)
        for file in files:
            yield _lazy_load_dataset(file, corpus_type)
    else:
        raise Exception("Only ONE dataset, check whether it right.")



class Batch(object):
    def __init__(self, batch_data, device=None, is_test=False):
        self.batch_data = batch_data
        self.device = device
        self.is_test = is_test

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        encode, labels = self.batch_tokenize()
        encode['next_sentence_label'] = labels

        setattr(self, 'encode', encode.to(device))


    def normalize(self, batch_data):
        print(len(batch_data))
        print(type(batch_data))

        sentence_pair = [s[0] for s in batch_data]
        label = [[s[1]] for s in batch_data]

        sent_a = [pair[0] for pair in sentence_pair]
        sent_b = [pair[1] for pair in sentence_pair]
        return sent_a, sent_b, label

    def batch_tokenize(self):
        sent_a, sent_b, label = self.normalize(self.batch_data)
        print(sent_a)
        encode = self.tokenizer(sent_a, sent_b, padding='longest', return_tensors='pt')
        labels = torch.LongTensor(label)

        return encode, labels



class DataLoaderBert:
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

        return DataIteratorBert(self.cur_dataset, self.batch_size, self.device,
                            self.is_test, self.shuffle)


class DataIteratorBert(object):
    def __init__(self, dataset, batch_size, device=None, is_test=False, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size

        self.device = device
        self.is_test = is_test
        self.shuffle = shuffle
        self.iterations = 0

        self._iterations_this_epoch = 0

    def dataset_(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        dataset_ = self.dataset
        return dataset_

    def preprocess(self, example):
        tgt = example['tgt']
        label = example['coherence']

        return tgt, label

    def batch(self):
        dataset = self.dataset_()
        batch_size = self.batch_size

        minibatch, current_size = [], 0
        for ex in dataset:
            example = self.preprocess(ex)

            minibatch.append(example)
            if len(minibatch) == batch_size:
                yield minibatch
                minibatch = []
            elif len(minibatch) > batch_size:
                yield minibatch[:-1]
                minibatch = minibatch[-1:]
        if minibatch:
            yield minibatch

    def __iter__(self):
        while True:
            self.batches = self.batch()
            for idx, minibatch in enumerate(self.batches):
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(minibatch, self.device, self.is_test)

                yield batch
            return




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-shard_tgt_root_path', default="/sdc/xli/Datasets/cnn_daily/tgts/shard_pairs", type=str)

    args = parser.parse_args()

    datasets = load_dataset(args, 'train', shuffle=True)


    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_iter = DataLoaderBert(datasets, batch_size=200, device='cuda', shuffle=True)
    for i, batch in enumerate(train_iter):
        if i < 5:
            print(batch.encode)

            print("---------------------------------------------------")
        else:
            break


