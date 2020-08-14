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
    return clean_input


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

        subprocess.call(command)


class processorCnnDailymail:
    def __init__(self, args):
        self.args = args

    def read_mapping(self, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
        return [line.strip() for line in lines]

    def load_json(self, json_file, lower=True, remove_cnn=True):
        def remove(input_str):
            output = re.subn(r"\( cnn \)", r"", input_str, 1)[0]
            assert isinstance(output, str)
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

    def shard(self):
        init_logger()
        def check_file_exists(root_path):
            for f in glob.glob(os.path.join(root_path, "*.json")):
                file_path = pathlib.Path(f)
                if file_path.exists():
                    os.unlink(file_path)

        pairs_train_mapping, pairs_test_mapping = self.args.pairs_train_mapping, self.args.pairs_test_mapping
        train_files, test_files = map(self.read_mapping, (pairs_train_mapping, pairs_test_mapping))

        divided_corpus = {'train': train_files,
                          'test': test_files}

        # delete all files under the save_path before write in
        check_file_exists(self.args.save_path)

        pool = Pool(mp.cpu_count())
        for corpus_type in ['train', 'test']:
            files = divided_corpus.get(corpus_type)
            dataset = []
            file_no = 0
            for d in pool.imap_unordered(self.load_pairs, files):
                if d is not None:
                    dataset.append(d)

                    if len(dataset) > self.args.shard_size:
                        pt_file = os.path.join(self.args.save_path, "{:s}/cd_{:s}_{:d}.json".format(corpus_type, corpus_type, file_no))
                        with open(pt_file, 'w') as save:
                            save.write(json.dumps(dataset))

                        logger.info("cd_{:s}_{:d}.json saved at {:s}/{:s}.".format(corpus_type, file_no, self.args.save_path, corpus_type))
                        file_no += 1
                        dataset = []

                else:
                    continue

            if len(dataset) > 0:
                pt_file = os.path.join(self.args.save_path, "{:s}/cd_{:s}_{:d}.json".format(corpus_type, corpus_type, file_no))
                with open(pt_file, 'w') as save:
                    save.write(json.dumps(dataset))
                file_no += 1
        pool.close()
        pool.join()

        print("Shard task is finished!")

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

    def check_and_delete(self, path):
        init_logger()
        # file_path = pathlib.Path(path)
        # if file_path.exists():
        #     os.unlink(file_path)
        #     logger.info("{:s} deleted".format(path))
        for f in glob.glob(os.path.join(path, "*.json")):
            file_path = pathlib.Path(f)
            if file_path.exists():
                os.unlink(file_path)
                logger.info("{:s} deleted from {:s}".format(f, path))


    def multi_delete(self, root_path):
        pool = Pool(mp.cpu_count())
        for p in pool.imap(self.check_and_delete, glob.glob(os.path.join(root_path, "*.json"))):
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
    parser.add_argument('-pairs_test_mapping', default='/sdc/xli/Datasets/cnn_daily/tgts/pairs_test_mapping.txt')
    parser.add_argument('-pairs-train_mapping', default='/sdc/xli/Datasets/cnn_daily/tgts/pairs_train_mapping.txt')




    parser.add_argument('-shard_size', default=6000, type=int)
    parser.add_argument('-save_path', default='/sdc/xli/Datasets/cnn_daily/tgts/shard_pairs', type=str)
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
        processor = processorCnnDailymail(args)

        save_path = ["/sdc/xli/Datasets/cnn_daily/tgts/shard_pairs/train", "/sdc/xli/Datasets/cnn_daily/tgts/shard_pairs/test"]
        # processor.save_tgts(save_path)
        # for path in save_path:
        #     processor.check_and_delete(path)
        processor.shard()

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
















