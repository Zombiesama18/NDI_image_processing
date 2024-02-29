import argparse
import os
from typing import Optional, Union

import numpy as np
import torch
import logging
import random
import pandas as pd


def torch_fix_seed(seed=3407):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

class MetricMeter:
    def __init__(self, name_list) -> None:
        # assure no duplicate names
        assert len(name_list) == len(set(name_list))
        
        self.metric_names = name_list
        self.reset()
    
    def reset(self):
        self.metrics = [np.array([]) for i in range(len(self.metric_names))]
    
    def update(self, 
               name_field: Union[str, list[str], tuple[str]], 
               metric_list: Union[float, list[float], tuple[float]]):
        if isinstance(name_field, str):
            name_field = [name_field]
        if isinstance(metric_list, float):
            metric_list = [metric_list]
        assert len(name_field) == len(metric_list)
        
        for name, metric in zip(name_field, metric_list):
            self.metrics[self.metric_names.index(name)] = np.append(self.metrics[self.metric_names.index(name)], metric)
        
    def get_metric(self, name: Union[str, list, tuple, None] = None):
        if name is None or name == 'all':
            name = self.metric_names
        if isinstance(name, str):
            name = [name]
        output_max = self.get_max(name)
        output_min = self.get_min(name)
        output_avg = self.get_avg(name)
        
        return [{'name': n,
                 'length': len(self.metrics[self.metric_names.index(n)]),
                 'max': output_max[i],
                 'min': output_min[i],
                 'avg': output_avg[i], 
                 'data': self.metrics[self.metric_names.index(n)]} 
                for i, n in enumerate(name)]
    
    def get_max(self, name: Union[str, list, None] = None):
        if name is None or name == 'all':
            name = self.metric_names
        if isinstance(name, str):
            name = [name]
        return [np.max(self.metrics[self.metric_names.index(n)]) for n in name]

    def get_min(self, name: Union[str, list, None] = None):
        if name is None or name == 'all':
            name = self.metric_names
        if isinstance(name, str):
            name = [name]
        return [np.min(self.metrics[self.metric_names.index(n)]) for n in name]
    
    def get_avg(self, name: Union[str, list, None] = None):
        if name is None or name == 'all':
            name = self.metric_names
        if isinstance(name, str):
            name = [name]
        return [np.mean(self.metrics[self.metric_names.index(n)]) for n in name]
    
    def get_last(self):
        output_fmt = "{}: {} ,"
        output = ""
        for name, metric in zip(self.metric_names, self.metrics):
            output += output_fmt.format(name, metric[-1])
        return output
    
    def merge(self, other: 'MetricMeter'):
        assert self.metric_names == other.metric_names
        for i, name in enumerate(self.metric_names):
            if len(self.metrics[i]) == 0:
                self.metrics[i] = other.metrics[i]
            elif len(other.metrics[i]) == 0:
                pass
            elif len(self.metrics[i].shape) == 1:
                self.metrics[i] = np.expand_dims(self.metrics[i], axis=0)
                other.metrics[i] = np.expand_dims(other.metrics[i], axis=0)
                self.metrics[i] = np.concatenate((self.metrics[i], other.metrics[i]), axis=0)
            else:
                other.metrics[i] = np.expand_dims(other.metrics[i], axis=0)
                self.metrics[i] = np.concatenate((self.metrics[i], other.metrics[i]), axis=0)
    
    def to_csv(self, path=None, columns_fmt=None, indices_fmt=None):
        if path is None:
            path = './logs/'
        
        for i, name in enumerate(self.metric_names):
            columns = [columns_fmt.format(i) for i in range(self.metrics[i].shape[1])]
            indices = [indices_fmt.format(i) for i in range(self.metrics[i].shape[0])]
            dataframe = pd.DataFrame(self.metrics[i], columns=columns, index=indices)
            dataframe.to_csv(os.path.join(path, f'{name}.csv'))


class AverageMeter:
    def __init__(self, fmt=':f') -> None:
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.sum = 0
        self.counter = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.counter += n
    
    def __str__(self) -> str:
        avg = self.sum / self.counter
        return self.fmt.format(avg)
    
    @property
    def avg(self):
        return self.sum / self.counter
    
    @property
    def value(self):
        return self.val
    
    @property
    def total(self):
        return self.sum
    
    @property
    def count(self):
        return self.counter


def get_wandb_API_key():
    return '4d82a00b168aacde462a6aae9d468d23b71d9efa'


def set_all_seeds(seed=29):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def cal_accuracy_top_k(preds, label, top_k=(1,)):
    with torch.no_grad():
        result = []
        max_k = max(top_k)
        sample_num = preds.shape[0]
        pred_scores, pred_labels = preds.topk(max_k, 1, True, True)
        pred_labels = pred_labels.t()
        correct = pred_labels.eq(label.view(1, -1).expand_as(pred_labels))
        for k in top_k:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            result.append(correct_k.mul_(100.0 / sample_num))
    return result


def str2bool(v):
    """
    Converts string to bool type; enables command line
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_checkpoints(base_encoder, ckpt_path):
    temp = torch.load(ckpt_path)['state_dict']
    state_dict = {}
    for k, v in temp.items():
        if 'encoder_q' in k:
            state_dict['.'.join(k.split('.')[1:])] = v
    base_encoder.load_state_dict(state_dict, strict=False)
    return base_encoder
