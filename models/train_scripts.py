import subprocess
import re

if __name__ == '__main__':
    # subprocess.call("source ./.bashrc".split())
    # pre_command = """source activate py38transformers"""
    # subprocess.call(pre_command.split())

    command = """python train.py
    -mode train
    -visible_gpus 2,3
    -lr 0.002
    -grad_accum_count 3
    -report_every 100
    -batch_size 32
    -dropout 0.1
    -tensorboard_log_dir /sdc/xli/py/coherence_bert/model_path/tf_events_2
    -train_from /sdc/xli/py/coherence_bert/model_path/model_step_14000.pt
    """

    command = re.sub("\n", " ", command)

    subprocess.call(command.split())