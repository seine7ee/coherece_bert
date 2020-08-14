import glob
import os
from multiprocessing import Pool
import multiprocessing as mp
from utils.logging_utils import logger, init_logger
import pathlib

def shard_mapping(root_path):
    files = list(glob.glob(os.path.join(root_path, "*.json")))
    for i in range(0, len(files), 10000):
        max = i + 10000
        yield files[i : max]

def write_mapping(params):
    init_logger("/sdc/xli/Datasets/cnn_daily/data_nsp/shard/mapping/mapping.log")
    paths, save_file = params
    with open(save_file, 'w') as file:
        for path in paths:
            file.write(path + "\n")
        logger.info("{:d} files has write in mapping file".format(len(paths)))

def multi_write_mapping(root_path, save_root_path, c_type):
    mappings = list(shard_mapping(root_path))
    pool = Pool(mp.cpu_count())
    params = [(mappings[i], os.path.join(save_root_path, "cd_{:s}_{:d}.txt".format(c_type, i))) for i in range(len(mappings))]
    for p in pool.imap(write_mapping, params):
        pass
    pool.close()
    pool.join()


def read_clean_stories():
    # 读取/sdc/xli/Datasets/cnn_daily/clean_stories目录下的stories
    path = "/sdc/xli/Datasets/cnn_daily/all_stories_jsons"
    file_list = os.listdir(path)
    print(len(file_list))

def tally_CnnDaily_dataset():
    # 计算/sdc/xli/Datasets/cnn_daily/dataset目录下的train/test/valid的json文件数
    root_path = "/sdc/xli/Datasets/cnn_daily/dataset"
    # tally the train/test/valid set size
    for type in ['train', 'test', 'valid']:
        path = os.path.join(root_path, type)
        type_list = os.listdir(path)
        print("{:s} dataset's length is: {:d}".format(type,len(type_list)))

    # check whether the mapping file is correlated to the target file
    for type in ['train', 'test', 'valid']:
        mapping_file = os.path.join(root_path, "{:s}_mapping.txt".format(type))
        with open(mapping_file, 'r') as file:
            files = [line.strip() for line in file.readlines()]
        match = True
        for file in files:
            if not pathlib.Path(file).exists():
                match = False
                print("{:s}_mapping.txt is not correlated to the target file".format(type))
                break

        if match:
            print("{:s}_mapping.txt is correlated to the target file".format(type))


def delete_tgt():
    init_logger()
    root_path = "/sdc/xli/Datasets/cnn_daily/tgts"
    for root, dirs, file_list in os.walk(root_path):
        for file in file_list:
            file_path = os.path.join(root, file)
            os.unlink(file_path)
            logger.info("{:s} deleted from {:s}".format(file, root))
    os.removedirs(root_path)
    logger.info("{:s} dir deleted.".format(root_path))




if __name__ == '__main__':

    root_path = "/sdc/xli/Datasets/cnn_daily/data_nsp/valid"
    save_root_path = "/sdc/xli/Datasets/cnn_daily/data_nsp/shard/mapping/valid"
    # multi_write_mapping(root_path, save_root_path, 'valid')
    # tally_CnnDaily_dataset()
    # read_clean_stories()
    delete_tgt()

