import collections
import itertools
import time
import torch.nn.functional as f
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights

from dataset import *
from moco_model import MoCo


class HistoryRecorder:
    def __init__(self, metrics_names: (list, tuple), parameter_names: (list, tuple)):
        self.data = collections.defaultdict(list)
        self.metrics_names = metrics_names
        self.parameter_names = parameter_names

    def cal_add(self, data, target: dict = None):
        if not target:
            target = self.data
        assert isinstance(data, dict)
        for k, v in data.items():
            if k not in target:
                target[k] = v
            else:
                for i, item in enumerate(v):
                    if isinstance(item, np.ndarray):
                        target[k][i] += item
                    elif isinstance(item, dict):
                        self.cal_add(item, target[k][i])

    def cal_divide(self, number, target=None):
        if not target:
            target = self.data
        assert isinstance(target, dict)
        for k, v in target.items():
            for i, item in enumerate(v):
                if isinstance(item, np.ndarray):
                    target[k][i] = target[k][i] / number
                elif isinstance(item, dict):
                    self.cal_divide(number, target[k][i])

    def add(self, data: dict):
        # if len(self.names) < len(args):
        #     raise IndexError('Too much values to record')
        # else:
        #     for i in range(min(len(self.names), len(args))):
        #         if isinstance(self.data[self.names[i]], list):
        #             self.data[self.names[i]].append(args[i])
        #         elif isinstance(self.data[self.names[i]], dict):
        #             for k, v in args[i].items():
        #                 if k not in self.data[self.names[i]]:
        #                     self.data[self.names[i]][k] = [v]
        #                 else:
        #                     self.data[self.names[i]][k].append(v)
        self.data.update(data)

    def reset(self):
        self.data = {}

    def __getitem__(self, name):
        return self.data[name]


def cal_accuracy_top_k(preds, label, top_k=(1,)):
    result = []
    max_k = max(top_k)
    sample_num = preds.shape[0]
    pred_scores, pred_labels = preds.topk(max_k, dim=1)
    pred_labels = pred_labels.t()
    correct = pred_labels.eq(label.view(1, -1).expand_as(pred_labels))
    for k in top_k:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        result.append(correct_k.item())
    return result


def image_pair_matching(net, original_image, matching_image):
    net.eval()
    q = net.encoder_q(original_image)
    q = f.normalize(q, dim=1)
    k = net.encoder_k(matching_image)
    k = f.normalize(k, dim=1)
    logits = torch.einsum('nc,ck->nk', [q, k.T])
    return logits


def train_moco_return_metrics_top_k(net, train_iter, val_iter, criterion, optimizer, epochs, device, tested_parameter,
                                    k_candidates=(10,)):
    # train_metrics = HistoryRecorder(['Train Loss', 'Train Acc', 'Val Loss', 'Val Acc'], [list, dict, list, dict])

    to_tensor_func = torchvision.transforms.ToTensor()
    target_tensor = []
    for i in range(1, 185):
        target_tensor.append(
            to_tensor_func(Image.open(str(Path.joinpath(Path(TARGET_IMAGE), f'{i}.jpg')))).unsqueeze(0))
    target_tensor = torch.cat(target_tensor, dim=0)
    target_tensor = target_tensor.cuda(device)
    train_loss_record = []
    train_acc_record = {k: [] for k in k_candidates}
    val_loss_record = []
    val_acc_record = {k: [] for k in k_candidates}
    for epoch in range(epochs):
        net.cuda(device)
        total_loss = 0
        training_correct = collections.defaultdict(int)
        training_size = 0
        for origin, target, label in train_iter:
            net.train()
            origin, target, label = origin.cuda(device), target.cuda(device), label.cuda(device)
            output, labels = net(origin, target)
            loss = criterion(output, labels)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            net.eval()
            with torch.no_grad():
                for k, correct in zip(k_candidates,
                                      cal_accuracy_top_k(image_pair_matching(net, origin, target_tensor), label,
                                                         top_k=k_candidates)):
                    training_correct[k] += correct
                training_size += origin.shape[0]
        net.eval()
        with torch.no_grad():
            val_loss = 0
            val_correct = collections.defaultdict(int)
            for origin, target, label in val_iter:
                origin, target, label = origin.cuda(device), target.cuda(device), label.cuda(device)
                output, labels = net(origin, target, evaluate=True)
                val_loss += f.cross_entropy(output, labels).item()
                for k, correct in zip(k_candidates,
                                      cal_accuracy_top_k(image_pair_matching(net, origin, target_tensor), label,
                                                         top_k=k_candidates)):
                    val_correct[k] += correct
        val_acc = {k: correct / origin.shape[0] for k, correct in val_correct.items()}
        train_acc = {k: correct / training_size for k, correct in training_correct.items()}
        train_loss_record.append(total_loss / len(train_iter))
        for k, v in train_acc.items():
            train_acc_record[k].append(v)
        val_loss_record.append(val_loss / len(val_iter))
        for k, v in val_acc.items():
            val_acc_record[k].append(v)
        print(f'Epoch {epoch + 1}, Train_Loss {total_loss / len(train_iter)}, Val_loss {val_loss / len(val_iter)}')
        for k, acc in train_acc.items():
            print(f'Train_acc_top_{k} {round(acc, 4)}', end='\t')
        print()
        for k, acc in val_acc.items():
            print(f'Val_acc_top_{k} {round(acc, 2)}', end='\t')
        print()
    output = normalize_data_format(
        {tuple(tested_parameter): (train_loss_record, train_acc_record, val_loss_record, val_acc_record)})
    return output


def normalize_data_format(data: dict, inner=False):
    result = collections.defaultdict(list)
    for k, v in data.items():
        if isinstance(v, tuple):
            for item in v:
                if isinstance(item, list):
                    result[k].append(np.array(item))
                else:
                    result[k].append(normalize_data_format(item, inner=True))
        elif isinstance(v, list):
            result[k].append(np.array(v))
        elif isinstance(v, dict):
            result[k].append(normalize_data_format(v, inner=True))
    if inner:
        for k, v in result.items():
            if isinstance(v, list) and len(v) == 1:
                result[k] = v[0]
    return result


def fmts_gen():
    line_styles = ['-', ':', '--', '-.']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'w']
    for color in colors:
        for line_style in line_styles:
            yield color + line_style


def draw_graph(metrics, num_epochs: int, metrics_name: (list, tuple)):
    X = np.arange(1, num_epochs + 1, 1)
    fig, axes = plt.subplots((len(metrics) + 1) // 2, 2, figsize=(15, 20))
    if hasattr(axes, 'flatten'):
        axes = axes.flatten()
    else:
        axes = [axes]
    for i, (k, v) in enumerate(metrics.items()):
        fmts = fmts_gen()
        plot_data = [v[0]]
        plot_data.extend(v[1].values())
        plot_data.append(v[2])
        plot_data.extend(v[3].values())
        for y, fmt in zip(plot_data, fmts):
            axes[i].plot(X, y, fmt)
        axes[i].set_xlabel('Epochs')
        axes[i].set_xlim([0, num_epochs + 1])
        axes[i].set_ylim([0, 4])
        axes[i].legend(
            ['Train Loss'] + [f'Train Acc Top {k}' for k in v[1].keys()] + ['Val Loss'] + [f'Val Acc Top {k}' for k in
                                                                                           v[3].keys()])
        axes[i].grid()
        axes[i].set_title(
            f'Training result when ' + ', '.join([f'{name} = {value}' for name, value in zip(metrics_name, k)]))
    plt.show()


def train_sample():
    top_k_candidates = (10, 20, 30)
    k = 7
    temps = [0.7]
    momentums = [0.999, 0.99]
    # momentums = [0.99]
    k_value = 48
    parameters = {'temp': temps, 'momentum': momentums}
    train_metrics = HistoryRecorder(['Train Loss', 'Train Acc', 'Val Loss', 'Val Acc'], list(parameters.keys()))
    for images in k_fold_train_validation_split(ORIGINAL_IMAGE, TARGET_IMAGE, k):
        train_dataset = ThreeChannelNDIDatasetContrastiveLearningWithAug(images, False)
        val_dataset = ThreeChannelNDIDatasetContrastiveLearningWithAug(images, True)
        train_iter = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
        val_iter = DataLoader(val_dataset, batch_size=len(val_dataset))
        for parameter in itertools.product(*parameters.values()):
            ### custom part to get parameters
            temperature = parameter[0]
            momentum = parameter[1]
            ### END
            model = torchvision.models.resnet50
            model = MoCo(model, dim=512, K=k_value, T=temperature, m=momentum, model_type='resnet',
                         weights=ResNet50_Weights.IMAGENET1K_V2, three_channel=True)
            device = torch.device('cuda:0')
            criterion = nn.CrossEntropyLoss().cuda(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
            start_time = time.time()
            metrics = train_moco_return_metrics_top_k(model, train_iter, val_iter, criterion, optimizer, 50, device,
                                                      tested_parameter=parameter, k_candidates=top_k_candidates)
            end_time = time.time()
            train_metrics.cal_add(metrics)
    train_metrics.cal_divide(k)
