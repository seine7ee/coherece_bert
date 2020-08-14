from __future__ import absolute_import
import logging

logger = logging.getLogger()

def init_logger(log_file=None, log_file_level=logging.NOTSET):
    log_formatter = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.handlers = [console_handler]

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(log_file_level)
        logger.addHandler(file_handler)
    return logger