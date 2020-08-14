import os
import torch
from utils.logging_utils import logger
from torch.nn import CrossEntropyLoss
import utils.distributed as distributed
from utils.reporter import Statistics, ReportMgr
from tensorboardX import SummaryWriter
import torch.distributed as dist

def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params

def build_trainer(args, device_id, model, optim):

    grad_accum_count = args.grad_accum_count
    n_gpu = args.world_size

    if device_id >= 0:
        gpu_rank = int(args.gpu_ranks[device_id])
    else:
        gpu_rank = 0
        n_gpu = 0
    print("gpu_rank %d" % gpu_rank)

    tensorboard_log_dir = args.tensorboard_log_dir
    writer = SummaryWriter(tensorboard_log_dir, comment='bert_coherence_measurement')
    report_manager = ReportMgr(args.report_every, start_time=-1, tensorboard_writer=writer)

    trainer = Trainer(args, model, optim, grad_accum_count,
                      n_gpu, gpu_rank, report_manager=report_manager)

    if model:
        n_params = _tally_parameters(model)
        logger.info("* number of parameters: {:d}".format(n_params))
    return trainer

class Trainer(object):
    def __init__(self, args, model, optim, grad_accum_count=1,
                 n_gpu=1, gpu_rank=1, report_manager=None):
        self.args = args
        self.save_checkpoint_step = args.save_checkpoint_steps
        self.model = model
        self.optim = optim
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager

        self.loss = CrossEntropyLoss()
        assert grad_accum_count > 0
        if model:
            self.model.train()

    def train(self, train_iter_method, train_steps):
        logger.info("Start training...")

        step = self.optim._step + 1
        true_batches  = []
        accum, normalization = 0, 0
        train_iter = train_iter_method()

        total_stats = Statistics()   # 初始化，loss=0, n_docs=0, start_time=time.time()
        report_stats = Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        while step <= train_steps:
            reduce_number = 0
            for i, batch in enumerate(train_iter):
                if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):
                    true_batches.append(batch)
                    # normalization += batch.batch_size
                    normalization = len(true_batches)
                    accum += 1

                    if accum == self.grad_accum_count:
                        reduce_number += 1
                        # if self.n_gpu > 1:
                        #     normalization = sum(distributed.all_gather_list(normalization))

                        # print("Normalization: {}".format(normalization))
                        self._gradient_accumulation(true_batches, normalization,
                                                    total_stats, report_stats)

                        report_stats = self._maybe_report_training(
                            step, train_steps, self.optim.learning_rate,
                            report_stats
                        )


                        true_batches = []
                        accum, normalization = 0, 0

                        if (step % self.save_checkpoint_step == 0 and self.gpu_rank == 0):
                            self._save(step)
                        step += 1
                        if step > train_steps:
                            break
            train_iter = train_iter_method()

        return total_stats





    def _gradient_accumulation(self, true_batches, normalization,
                               total_stats, report_stats):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batches:
            if self.grad_accum_count == 1:
                self.model.zero_grad()

            encode = batch.encode
            mask = encode['attention_mask']
            label = batch.labels                 # batch_size, 1
            outputs = self.model(**encode)
            seq_relationship_score = outputs[0] # batch_size, 2

            # loss on one batch
            loss = self.loss(seq_relationship_score.view(-1, 2), label)
            # print("****** loss: {:f} ******".format(loss))

            batch_stats = Statistics(float(loss.cpu().data.numpy()),
                                     1, self.loss.reduction)

            (loss / self.grad_accum_count).backward()

            total_stats.update(batch_stats)  # true_batches中batch个loss相加，以及batch的文档数
            report_stats.update(batch_stats)

            if self.grad_accum_count == 1:
                if self.n_gpu > 1:
                    # multi GPU gradient gather
                    # grads = [p.grad.data for p in self.model.parameters()
                    #          if p.requires_grad and p.grad is not None]
                    # distributed.all_reduce_and_rescale_tensors(grads, float(1))
                    self.average_gradients()
                self.optim.step()


        if self.grad_accum_count > 1:
            if self.n_gpu > 1:
                # grads = [p.grad.data for p in self.model.parameters()
                #          if p.requires_grad and p.grad is not None]
                # distributed.all_reduce_and_rescale_tensors(grads, float(1))
                self.average_gradients()
            self.optim.step()


    def average_gradients(self):
        size = self.n_gpu
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                dist.all_reduce(param.grad.data)
                param.grad.data /= size

    def _save(self, step):
        real_model = self.model

        model_state_dict = real_model.state_dict()
        checkpoint = {'model': model_state_dict,
                      'optim': self.optim
                      }

        checkpoint_path = os.path.join(self.args.model_path, 'model_step_{:d}.pt'.format(step))
        logger.info('Saving checkpoing {:s}'.format(checkpoint_path))

        if not os.path.exists(checkpoint_path):
            torch.save(checkpoint, checkpoint_path)
            return checkpoint, checkpoint_path

    def _start_report_manager(self, start_time=None):
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        report_stats包含的信息只有batch个loss相加以及batch个文档数

        :param step:
        :param num_steps:
        :param learning_rate:
        :param report_stats:
        :return:
        """

        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu>1
            )


