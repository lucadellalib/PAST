import sys
import colorlog
import logging

import torch


def set_logger(logger):
    level = 'INFO'
    logger.setLevel(level)
    log_format = '[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s][%(log_color)s%(levelname)s%(reset)s] - %(message)s'
    formatter = colorlog.ColoredFormatter(log_format, datefmt="%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(level)
    sh.setFormatter(formatter)
    logger.addHandler(sh)


def get_device(process_id: int) -> torch.device:
    """Get device to use for training."""
    if torch.cuda.is_available():
        torch.cuda.set_device(process_id % torch.cuda.device_count())
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device
