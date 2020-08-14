import torch
from utils.logging_utils import logger, init_logger
import utils.distributed as distributed
import random
from models.model_builder import NextSentencePrediction, build_optim
from models.trainer import build_trainer
from models.data_loader import load_dataset, DataLoaderBert
import os
import signal
import argparse
from transformers import BertTokenizer



def single_train(args, device_id):
    init_logger(args.log_file)

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        # 使用指定的gpu
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from,
                                map_location=lambda storage, loc: storage)

    else:
        checkpoint = None

    def train_iter_method():
        return DataLoaderBert(load_dataset(args, 'train', shuffle=True), args.batch_size,
                            shuffle=True, is_test=False)
    model = NextSentencePrediction(args, device, checkpoint)
    optim = build_optim(args, model, checkpoint)

    logger.info(model)

    trainer = build_trainer(args, device_id, model, optim)
    trainer.train(train_iter_method, args.train_steps)


def run(args, device_id, error_queue):
    setattr(args, 'gpu_ranks', [int(i) for i in args.gpu_ranks])

    try:
        gpu_rank = distributed.multi_init(device_id, args.world_size, args.gpu_ranks)

        logger.info("GPU Rank: gpu_rank {:d}".format(gpu_rank))
        if gpu_rank != args.gpu_ranks[device_id]:
            raise AssertionError("An Error occured in Distributed intializaiton")
        single_train(args, device_id)
    except KeyboardInterrupt:
        pass
    except Exception:
        import traceback
        error_queue.put((args.gpu_ranks[device_id], traceback.format_exc()))

def multi_train(args):
    init_logger()

    gpu_number = args.world_size
    mp = torch.multiprocessing.get_context('spawn')

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing
    procs = []
    for i in range(gpu_number):
        device_id = i
        procs.append(mp.Process(target=run,
                                args=(args, device_id, error_queue, ),
                                daemon=True))
        procs[i].start()
        logger.info("Starting process pid: {:d} ".format(procs[i].pid))
        error_handler.add_child(procs[i].pid)
    for p in procs:
        p.join()


def train(args, device_id):
    if args.world_size > 1:
        multi_train(args)
    else:
        single_train(args, device_id)


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', choices=['train', 'test', 'valid'], default='train', type=str)
    parser.add_argument('-visible_gpus', default="-1", type=str)
    parser.add_argument('-gpu_ranks', default=0, type=str)
    parser.add_argument('-log_file', default='../logs/my_log.log')
    parser.add_argument('-seed', default=666, type=int)


    parser.add_argument('-train_from', default='', type=str)
    parser.add_argument('-train_steps', default=100000, type=int)
    parser.add_argument('-batch_size', default=128, type=int)
    parser.add_argument('-shard_tgt_root_path', default='/sdc/xli/Datasets/cnn_daily/data_nsp/shard/pts', type=str)

    parser.add_argument('-large', default=False, type=bool)
    parser.add_argument('-temp_dir', default='/sdc/xli/py/bert/models/bert_uncased', type=str)
    parser.add_argument('-finetune', default=True, type=bool)

    parser.add_argument('-optim', default='adam', type=str)
    parser.add_argument('-lr', default=1, type=float)
    parser.add_argument('-beta1', default=0.9, type=float)
    parser.add_argument('-beta2', default=0.999, type=float)
    parser.add_argument('-max_grad_norm', default=0, type=float)
    parser.add_argument('-warmup_steps', default=10000, type=int)
    parser.add_argument('-dropout', default=0.1, type=float)

    parser.add_argument('-tensorboard_log_dir', default='/sdc/xli/py/coherence_bert/model_path/tf_events', type=str)


    parser.add_argument('-model_path', default="/sdc/xli/py/coherence_bert/model_path")
    parser.add_argument('-save_checkpoint_steps', default=500, type=int)
    parser.add_argument('-grad_accum_count', default=1, type=int)

    parser.add_argument('-report_every', default=1, type=int)



    args = parser.parse_args()
    # gpu_ranks is a list of gpu order
    args.gpu_ranks = [int(i) for i in range(len(args.visible_gpus.split(',')))]
    print("args.gpu_ranks: {}".format(args.gpu_ranks))
    args.world_size = len(args.gpu_ranks)
    print("args.world_size: {}".format(args.world_size))
    print("args.visible_gpus: {}".format(args.visible_gpus))
    """
    pytorch 指定gpu训练的方式
     - 1. 直接终端中使用：
        >>> CUDA_VISIBLE_DEVICES=1 python train_scripts.py
     - 2. python代码中设定(官方建议使用该种方法)
        >>> import os
        >>> os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
     - 3. use torcu.cuda.set_device()
        >>> import torch
        >>> torch.cuda.set_device(gpu_id)
    """

    os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_gpus
    print("Current id: {}".format(torch.cuda.current_device()))

    init_logger(args.log_file)
    device = 'cpu' if args.visible_gpus == "-1" else 'cuda'
    device_id = 0 if device == 'cuda' else -1

    if args.mode == 'train':
        train(args, device_id)






