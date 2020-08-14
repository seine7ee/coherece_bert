import os
import argparse
import glob
import subprocess
import re
import json
import random
from sklearn.model_selection import train_test_split
from collections.abc import Iterable
import pathlib
import multiprocessing as mp
from multiprocessing import Pool
from utils.logging_utils import logger, init_logger
from transformers import BertTokenizer
import torch
import gc
import shutil


class mappingAndDivide:
    def __init__(self, args):
        self.args = args

    def mapping(self):
        raw_path  = self.args.raw_path
        mapping_path = self.args.pairs_mapping_file

        def write_in(write_path, content):
            content_normal = content
            if not isinstance(content_normal, str):
                content_normal = str(content_normal)
            with open(write_path, 'a+') as file:
                file.write(content_normal + "\n")

        i = 0
        for dir in glob.glob(os.path.join(raw_path, "*.json")):
            write_in(mapping_path, dir)
            if i % 100 == 0:
                print("{} files has processed, continue processing...".format(i))
            i += 1
        print("ALL Finished!")


    def write_in(self, filename, content):
        if isinstance(content, Iterable):
            file_path = pathlib.Path(filename)
            if file_path.exists():
                os.unlink(filename)
            with open(filename, 'a+') as file:
                for ele in content:
                    file.write(str(ele))
        elif isinstance(content, str):
            file_path = pathlib.Path(filename)
            if file_path.exists():
                os.unlink(filename)
            with open(filename, 'a+') as file:
                file.write(content)
        else:
            raise Exception("Input content neithor a Iterable object nor a string, "
                            "it is a {} type object, please check the input content.".format(type(content)))

    def replace_path(self, path):
        json_path = re.sub(r"cnn_stories/stories|dailymail_stories/stories", "all_stories_jsons", path).strip() + ".json\n"
        return json_path

    def get_divide_mapping(self, test_size, shuffle=False):
        mapping_train_file = self.args.train_mapping_file
        mapping_test_file = self.args.test_mapping_file

        global_mapping_file = self.args.global_mapping_file
        with open(global_mapping_file, 'r') as file:
            files = file.readlines()

        if shuffle:
            random.seed(66)
            random.shuffle(files)

        # 数据集划分
        train_files, test_files = train_test_split(list(map(self.replace_path, files)), test_size=test_size, shuffle=shuffle)



        assert len(train_files) + len(test_files) == len(files)
        self.write_in(mapping_train_file, train_files)
        self.write_in(mapping_test_file, test_files)
        print("All {} files has divided!".format(len(files)))

class BertData:
    def __init__(self, args):
        self.args = args


    def clean_text(self):
        '''
        文本开头存在一些
        :return:
        '''
        pass

    def convert_and_save(self, path):
        def save(input):
            init_logger()
            with open(self.args.save_file, 'a+') as file:
                json.dump(input, file)
            logger.info("")

        # vocab_path = "/sdc/xli/py/bert/bert_component/bert-base-uncased-vocab.txt"
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        with open(path) as json_file:
            data = json.load(json_file)
        sentence_pair = data['tgt']
        label = data['coherence']

    def write_in(self, mapping_file):
        with open(mapping_file, 'r') as file:
            paths = [line.strip() for line in file.readlines()]

        pool = Pool(mp.cpu_count())
        for d in pool.imap(self.convert_and_save, paths):
            pass
        pool.close()
        pool.join()


def clean(input):
    REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
             "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"', "`": "'"}

    clean_input = re.sub("-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''|`", lambda res: REMAP.get(res.group()), input)
    return clean_input.strip()


class corenlpTokenizer:
    def __init__(self, args):
        self.args = args

    def command_tokenize(self):

        command = ['java', '-cp',
                   '/sdc/xli/tools/corenlp/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar:/sdc/xli/tools/corenlp/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2-models.jar:/sdc/xli/tools/corenlp/stanford-corenlp-full-2018-10-05/xom.jar:/sdc/xli/tools/corenlp/stanford-corenlp-full-2018-10-05/joda-time.jar:/sdc/xli/tools/corenlp/stanford-corenlp-full-2018-10-05/jollyday.jar:/sdc/xli/tools/corenlp/stanford-corenlp-full-2018-10-05/ejml-0.23.jar',
                   'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
                   '-ssplit.newlineIsSentenceBreak', 'always', '-file', '/sdc/xli/tt.story',
                   '-outputFormat', 'json', '-outputDirectory', '/sdc/xli/tt_res.story.json']

        command_v2 = 'java -cp /sdc/xli/tools/corenlp/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar:/sdc/xli/tools/corenlp/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2-models.jar:/sdc/xli/tools/corenlp/stanford-corenlp-full-2018-10-05/xom.jar:/sdc/xli/tools/corenlp/stanford-corenlp-full-2018-10-05/joda-time.jar:/sdc/xli/tools/corenlp/stanford-corenlp-full-2018-10-05/jollyday.jar:/sdc/xli/tools/corenlp/stanford-corenlp-full-2018-10-05/ejml-0.23.jar edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit -ssplit.newlineIsSentenceBreak always -file /sdc/xli/tt.story -outputFormat json'

        command_tokenize = 'java -cp /sdc/xli/tools/corenlp/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar:/sdc/xli/tools/corenlp/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2-models.jar:/sdc/xli/tools/corenlp/stanford-corenlp-full-2018-10-05/xom.jar:/sdc/xli/tools/corenlp/stanford-corenlp-full-2018-10-05/joda-time.jar:/sdc/xli/tools/corenlp/stanford-corenlp-full-2018-10-05/jollyday.jar:/sdc/xli/tools/corenlp/stanford-corenlp-full-2018-10-05/ejml-0.23.jar edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit -ssplit.newlineIsSentenceBreak always -filelist /sdc/xli/Datasets/cnn_daily/supplement.txt -outputFormat json -outputDirectory /sdc/xli/Datasets/cnn_daily/all_stories_jsons'
        subprocess.call(command)


class processorCnnDailymail:
    def __init__(self, args):
        self.args = args


    def make_mapping_file(self, file_root_path, mapping_file, suffix='json'):
        for f in glob.glob(os.path.join(file_root_path, "*.{:s}".format(suffix))):
            with open(mapping_file, 'a+') as m_file:
                m_file.write(f.strip() + "\n")

    def read_mapping(self, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
        legal_files = []
        for line in lines:
            file = line.strip()
            yield file
            # with open(file, 'r') as j_file:
            #     data = json.load(j_file)
            # if len(data['tgt'][0]) >= self.args.min_n_tokens and len(data['tgt'][1]) >= self.args.min_n_tokens\
            #         and len(data['tgt'][0] + data['tgt'][1]) <= self.args.max_n_tokens:
                # legal_files.append(file)

        # return legal_files

    def load_json(self, json_file, lower=True, remove_cnn=True):
        def remove(input_str):
            output = re.subn(r"\( cnn \)", r"", input_str, 1)[0]
            assert isinstance(output, str)
            output = re.sub("^--", "", output.strip()).strip()
            return output

        source, tgt = [], []
        sum_label = False
        for sent in json.load(open(json_file))['sentences']:
            if lower:
                tokens = [t['word'].lower() for t in sent['tokens']]
            else:
                tokens = [t['word'] for t in sent['tokens']]

            if tokens[0] == "@highlight":
                sum_label = True
                continue
            if not sum_label:
                source.append(tokens)
            else:
                tgt.append(tokens)

        if remove_cnn:
            source = [remove(clean(" ".join(sent))).split() for sent in source]
            tgt = [remove(clean(" ".join(sent) + " . ")).split() for sent in tgt]
        else:
            source = [clean(" ".join(sent)).split() for sent in source]
            tgt = [clean(" ".join(sent) + " . ").split() for sent in tgt]
        return source, tgt


    def make_sample(self, params):
        """
        对原文中的句子，抽取出来构造正例、负例，因为reference中的句子并不是很连贯，所以考虑使用原文中的句子来作为连贯的文本
        接收一个json file，然后将这个json file取前5句，构造正例sentence pair
        构造负例sentence pari：
            1. 随机打乱，并删除与正例一样的负例
            2. 随机替换，将正例中的第二句替换成文章中其他部分的句子
        :param json_file:
        :return:
        """
        def save_pair(pairs, coherence, mark, file_id, save_path):
            init_logger()
            if len(pairs) > 0:
                for i, pair in enumerate(pairs):
                    pair_dict = {"pair": pair,
                                 "coherence": coherence}
                    save_file = os.path.join(save_path, "{:s}_{:s}_{:d}.json".format(file_id, mark, i))
                    with open(save_file, 'w') as file:
                        json.dump(pair_dict, file)
                    logger.info("{:s} saved".format(save_file))

        json_file, save_path = params
        source, _ = self.load_json(json_file)
        extracted = source[:5]
        extracted_ = list(extracted)
        random.seed(666)
        random.shuffle(extracted_)
        rest = source[5:]

        pos_pairs = [(extracted[i], extracted[i+1]) for i in range(len(extracted) - 1)]
        # _neg = random.sample(rest, len(pos_pairs))
        # negative samples that replace the next sentence with other sentences in the rest source
        neg_pairs_replaced = None
        # source[5:]的长度够采样长度才会产生neg_pairs_replaced
        if len(rest) >= len(pos_pairs):
            # 从剩余的文本中采样一句话重新组合成新的负例样本
            neg_pairs_replaced = [(pair[0], random.sample(rest, 1)[0]) for pair in pos_pairs]
        # negative samples that just shuffled
        neg_pairs_shuffled = [(extracted_[i], extracted_[i+1]) for i in range(len(extracted_) - 1)]

        length = len(pos_pairs)
        assert len(neg_pairs_shuffled) == length

        final_ns, final_nr = [], []
        # 统计neg_pairs_shuffled中与pos_pairs中重合的部分
        for i in range(length):
            if neg_pairs_shuffled[i] not in pos_pairs:
                final_ns.append(neg_pairs_shuffled[i])

        # 统计neg_paris_replaced中与pos_pairs中重合的部分
        if neg_pairs_replaced is not None:
            for i in range(len(neg_pairs_replaced)):
                if neg_pairs_replaced[i] not in pos_pairs:
                    final_nr.append(neg_pairs_replaced[i])

        file_id = json_file.split("/")[-1].split(".")[0]

        save_pair(pos_pairs, 0, 'p', file_id, save_path)
        save_pair(final_ns, 1, 'ns', file_id, save_path)
        save_pair(final_nr, 1, 'nr', file_id, save_path)

    def multi_make_sample(self):
        pool = Pool(mp.cpu_count())
        for type in ['train', 'test', 'valid']:
            json_files = glob.glob(os.path.join(self.args.pairs_root_path, "{:s}/*.json".format(type)))
            save_path = os.path.join(self.args.pairs_save_path, type)
            params = [(json_file, save_path) for json_file in json_files]
            for p in pool.imap(self.make_sample, params):
                pass
        pool.close()
        pool.join()



    def shard(self):
        init_logger("/sdc/xli/Datasets/cnn_daily/data_nsp/logs/shard.log")

        pairs_train_mapping, pairs_test_mapping, pairs_valid_mapping = \
            self.args.pairs_train_mapping, self.args.pairs_test_mapping, self.args.pairs_valid_mapping
        # train_files, test_files, valid_files = map(self.read_mapping, (pairs_train_mapping, pairs_test_mapping, pairs_valid_mapping))
        train_files = self.read_mapping(pairs_train_mapping)
        test_files = self.read_mapping(pairs_test_mapping)
        valid_files = self.read_mapping(pairs_valid_mapping)

        divided_corpus = {'train': train_files,
                          'test': test_files,
                          'valid': valid_files}


        pool = Pool(mp.cpu_count())
        for corpus_type in ['train', 'test', 'valid']:
            files = divided_corpus.get(corpus_type)
            dataset = []
            file_no = 0
            for d in pool.imap_unordered(self.load_pairs, files):
                if d is not None:
                    dataset.append(d)

                    if len(dataset) >= self.args.shard_size:
                        pt_file = os.path.join(self.args.save_path, "{:s}/cd_{:s}_{:d}.json".format(corpus_type, corpus_type, file_no))
                        with open(pt_file, 'w') as save:
                            save.write(json.dumps(dataset))

                        logger.info("{:s} has saved at {:s}/{:s}".format(pt_file.split("/")[-1], self.args.save_path, corpus_type))
                        file_no += 1
                        dataset = []

                else:
                    continue

            if len(dataset) > 0:
                pt_file = os.path.join(self.args.save_path, "{:s}/cd_{:s}_{:d}.json".format(corpus_type, corpus_type, file_no))
                with open(pt_file, 'w') as save:
                    save.write(json.dumps(dataset))
                logger.info(
                    "{:s} has saved at {:s}/{:s}".format(pt_file.split("/")[-1], self.args.save_path, corpus_type))
                file_no += 1
        pool.close()
        pool.join()

        logger.info("Shard task is finished!")

    def load_pairs(self, json_file):
        with open(json_file, 'r') as file:
            sentence_dict = json.load(file)
        return sentence_dict




    def tgt_samples(self, params):
        """
        construct positive tgt sample and negative tgt sample which is a random version of the positive one

        :param json_file:
        :return:
        """
        def save_json(save_path, file_id, samples):
            init_logger()
            for i, sample in enumerate(samples):
                save_ = os.path.join(save_path, "{:s}_{:d}.json".format(file_id, i))
                with open(save_, 'w') as file:
                    json.dump(sample, file)
                logger.info("{:s} saved at {:s}".format(save_, save_path))


        json_file, save_path = params
        init_logger()
        _, tgt = self.load_json(json_file)

        file_id = json_file.split("/")[-1].split(".")[0]
        if len(tgt) >= self.args.min_sents_num and len(tgt) <= self.args.max_sents_num:
            tgt_ = list(tgt)
            random.seed(66)
            random.shuffle(tgt_)

            # make sentence pair and write in a single file
            positive_sents = tgt
            positive_pairs = [(positive_sents[i], positive_sents[i+1]) for i in range(len(positive_sents)-1)]

            negative_sents = tgt_
            negative_pairs = [(negative_sents[i], negative_sents[i+1]) for i in range(len(negative_sents)-1)]

            positive_samples = [{"tgt": pair, "coherence": 0} for pair in positive_pairs]  # 0 represents coherent
            negative_samples = [{"tgt": pair, "coherence": 1} for pair in negative_pairs] # 1 represents incoherent

            save_json(save_path, file_id, positive_samples)
            save_json(save_path, file_id+"_r", negative_samples)

    def save_tgts(self, save_path):
        train_mapping_file, test_mapping_file = self.args.train_mapping_file, self.args.test_mapping_file
        all_files = map(self.read_mapping, (train_mapping_file, test_mapping_file))
        pool = Pool(mp.cpu_count())

        for i, files in enumerate(all_files):
            params = [(f, save_path[i]) for f in files]
            for d in pool.imap(self.tgt_samples, params):
                pass
        pool.close()
        pool.join()




    def _format_to_bert(self, params):
        init_logger("/sdc/xli/Datasets/cnn_daily/data_nsp/logs/_format_to_bert_one_sample.log")
        tokenizer, mapping_file, save_file = params

        logger.info("Processing {:s}".format(mapping_file))
        with open(mapping_file, 'r') as m_file:
            json_paths = (line.strip() for line in m_file.readlines())

        samples = []
        for json_file in json_paths:
            with open(json_file, 'r') as j_file:
                sample = json.load(j_file)
            pair = sample['pair']
            label = sample['coherence']

            try:
                encode = tokenizer(pair[0], pair[1], return_tensors='pt', is_pretokenized=True)

                if encode['input_ids'].numel() <= self.args.bert_max_position:
                    sample_dict = {'input_ids': encode['input_ids'].to('cuda'),
                                   'token_type_ids': encode['token_type_ids'].to('cuda'),
                                   'attention_mask': encode['attention_mask'].to('cuda')}
                    samples.append((sample_dict, label))
                else:
                    logger.info("Valid sample length: {}".format(encode['input_ids'].numel()))
            except ValueError:
                logger.warning("Value Error! And your data is {}".format(pair))

        torch.save(samples, save_file)
        logger.info("{:s} has converted and saved at {:s}".format(mapping_file, save_file))

        del(samples)
        gc.collect()

        # with open(file, 'r') as json_file:
        #     dataset = json.load(json_file)
        #
        # samples = []
        # logger.info("processing {:s}".format(file))
        # for i, sample in enumerate(dataset):
        #     # if i % 1000 == 0:
        #     #     logger.info("Now No.{:d} file".format(i))
        #     pair = sample['pair']
        #     label = sample['coherence']
        #
        #     # if not isinstance(pair, list) or len(pair) == 0\
        #     #         or not isinstance(pair[0][0], type(pair[1][0])):
        #     #     continue
        #     encode = tokenizer(pair[0], pair[1], return_tensors='pt', is_pretokenized=True)
        #
        #     if encode['input_ids'].numel() <= self.args.bert_max_position:
        #         sample_dict = {'input_ids': encode['input_ids'].to('cuda'),
        #                        'token_type_ids': encode['token_type_ids'].to('cuda'),
        #                        'attention_mask': encode['attention_mask'].to('cuda')}
        #         samples.append((sample_dict, label))
        #     else:
        #         logger.info("Valid sample length: {}".format(encode['input_ids'].numel()))
        #     if len(samples) > 10000:
        #         torch.save(samples, save_file)
        #
        # torch.save(samples, save_file)
        # logger.info("{:s} has converted and saved at {:s}".format(file, save_file))
        #
        # del(samples)
        # del(dataset)
        # gc.collect()

    def format_to_bert(self, tokenizer):

        json_root_path = self.args.json_root_path
        # /sdc/xli/Datasets/cnn_daily/data_nsp/shard/mapping/train/xxx.txt
        # /sdc/xli/Datasets/cnn_daily/data_nsp/shard/pts/train/xxx.pts
        for type in ['train', 'test', 'valid']:

            mapping_files = [f for f in glob.glob(os.path.join(self.args.mapping_root_path, "{:s}/*.txt".format(type)))]

            # j_files = [f for f in glob.glob(os.path.join(json_root_path, type+"/*.json"))]
            s_files = [re.sub(".txt", ".pt", re.sub("/mapping", "/pts", m_file)) for m_file in mapping_files]
            # s_files = (s_file for s_file in s_files if not pathlib.Path(s_file).exists())

            files = [(m_file, s_file) for m_file, s_file in zip(mapping_files, s_files) if not pathlib.Path(s_file).exists()]


            # params = ((tokenizer, j_file, re.sub(".json", ".pt", re.sub("/jsons", "/pts", j_file))) for j_file in j_files)
            pool = mp.Pool(mp.cpu_count())
            params_list = [(tokenizer, file[0], file[1]) for file in files]

            for p in pool.imap_unordered(self._format_to_bert, params_list):
                pass
            pool.close()
            pool.join()

    def _format_to_bert_one_sample(self, params):
        init_logger("/sdc/xli/Datasets/cnn_daily/data_nsp/logs/_format_to_bert_one_sample.log")
        tokenizer, file, save_file, sample_type = params
        with open(file, 'r') as json_file:
            sample = json.load(json_file)
        pair, coherence = sample['pair'], sample['coherence']

        if isinstance(pair, list) and len(pair) > 0 \
                and isinstance(pair[0][0], type(pair[1][0])):

            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
            encode = tokenizer(pair[0], pair[1], return_tensors='pt', is_pretrained=True)
            if encode['input_ids'].numel() <= self.args.bert_max_position:
                sample_dict = {'input_ids': encode['input_ids'],
                               'token_type_ids': encode['token_type_ids'],
                               'attention_mask': encode['attention_mask']}
                sample_tuple = (sample_dict, coherence)
                torch.save(sample_tuple, save_file)
                logger.info("{:s} has converted and saved at {:s}".format(file, save_file))

                file_name = file.split("/")[-1]
                dst_file = os.path.join("/sdc/xli/Datasets/cnn_daily/data_nsp/pts_and_back/processed", "{:s}/{:s}".format(sample_type, file_name))
                shutil.move(file, dst_file)
                logger.info("{:s} has moved to {:s}".format(file_name, dst_file))

            gc.collect()

    def format_to_bert_one_sample(self, tokenizer):
        pool = Pool(10)
        for type in ['train', 'test', 'valid']:
            j_files = (f for f in glob.glob(os.path.join(self.args.json_root_path, "{:s}/*.json".format(type))))

            # /sdc/xli/Datasets/cnn_daily/data_nsp/train/xxx.json
            # /sdc/xli/Datasets/cnn_daily/data_nsp/pts_and_back/pts/train
            params = ((tokenizer, j_file,
                       re.sub(".json", ".pt", re.sub("/{:s}".format(type), "/pts_and_back/{:s}".format(type), j_file)),
                       type) for j_file in j_files)
            for p in pool.imap(self._format_to_bert_one_sample, params):
                pass
        pool.close()
        pool.join()








if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', default='shard', choices=['mapping', 'tokenize', 'shard', 'test', 'bert'])
    parser.add_argument('-raw_path', default="/sdc/xli/Datasets/cnn_daily/tgts/pairs_test", type=str)
    parser.add_argument('-mapping_path', default='/sdc/xli/Datasets/cnn_daily/tgts/mapping_test_pairs.txt', type=str)
    parser.add_argument('-global_mapping_file', default='/sdc/xli/Datasets/cnn_daily/mapping_file.txt', type=str)
    parser.add_argument('-train_mapping_file', default='/sdc/xli/Datasets/cnn_daily/train_mapping_file.txt', type=str)
    parser.add_argument('-test_mapping_file', default='/sdc/xli/Datasets/cnn_daily/test_mapping_file.txt', type=str)
    parser.add_argument('-test_size', default=0.3)
    parser.add_argument('-pairs_mapping_file', default='/sdc/xli/Datasets/cnn_daily/tgts/pairs_test_mapping.txt')
    parser.add_argument('-pairs_test_mapping', default='/sdc/xli/Datasets/cnn_daily/data_nsp/test.txt')
    parser.add_argument('-pairs-train_mapping', default='/sdc/xli/Datasets/cnn_daily/data_nsp/train.txt')
    parser.add_argument('-pairs-valid-mapping', default='/sdc/xli/Datasets/cnn_daily/data_nsp/valid.txt')
    parser.add_argument('-mapping_root_path',default="/sdc/xli/Datasets/cnn_daily/data_nsp/shard/mapping", type=str)
    parser.add_argument('-min_n_tokens', default=5, type=int)
    parser.add_argument('-max_n_tokens', default=100, type=int)

    parser.add_argument('-pairs_root_path', default='/sdc/xli/Datasets/cnn_daily/dataset', type=str)
    parser.add_argument('-pairs_save_path', default='/sdc/xli/Datasets/cnn_daily/data_nsp', type=str)
    parser.add_argument('-json_root_path', default="/sdc/xli/Datasets/cnn_daily/data_nsp/shard/jsons", type=str)

    parser.add_argument('-bert_max_position', default=128, type=int, help="The max length of one sample")




    parser.add_argument('-shard_size', default=20000, type=int)
    parser.add_argument('-save_path', default='/sdc/xli/Datasets/cnn_daily/data_nsp/shard/jsons', type=str)
    parser.add_argument('-lower', default=True, type=bool)
    parser.add_argument('-remove_cnn', default=True, type=bool)
    parser.add_argument('-min_sents_num', default=3, type=int)
    parser.add_argument('-max_sents_num', default=500, type=int)

    parser.add_argument('-save_file', default='/sdc/xli/Datasets/cnn_daily/tgts/train.json')


    args = parser.parse_args()

    if args.mode == 'mapping':
        mad = mappingAndDivide(args)
        # mad.get_divide_mapping(args.test_size, shuffle=True)
        mad.mapping()
    elif args.mode == 'shard':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        processor = processorCnnDailymail(args)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        processor.format_to_bert(tokenizer)

        # processor.make_mapping_file("/sdc/xli/Datasets/cnn_daily/data_nsp/valid", "/sdc/xli/Datasets/cnn_daily/data_nsp/valid.txt")
        # processor.multi_make_sample()


        # processor.save_tgts(save_path)
        # for path in save_path:
        #     processor.check_and_delete(path)




        # args.json_root_path = "/sdc/xli/Datasets/cnn_daily/tgts/pairs_tgt/shard_pairs"
        # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # processor.format_to_bert(args, tokenizer)

    elif args.mode == 'bert':
        bd = BertData(args)
        mapping_file = "/sdc/xli/Datasets/cnn_daily/tgts/mapping_train.txt"
        bd.write_in(mapping_file)

    else:
        save_path = ["/sdc/xli/Datasets/cnn_daily/tgts/pairs_train", "/sdc/xli/Datasets/cnn_daily/tgts/pairs_test"]
        total_len = 0

        for path in save_path:
            dir_list = os.listdir(path)
            length = len(dir_list)
            total_len += length
            print("Number: {:d}".format(length))
        print("Total Number: {:d}".format(total_len))
        """
        Sentence pairs:
            Train Number: 1205700
            Test Number: 519200
        """
















