from datetime import datetime
from random import choices
import sys
import logging
import os

from typing import Any, List, Union
import torch
import torch.nn as nn

import torchvision.models

from dataclasses import dataclass, field

from models.core_model import RetrievalModel
from models.wd_model import CompositeResNet
from utils import MetricMeter, AverageMeter, set_all_seeds, cal_accuracy_top_k, load_checkpoints, HfArgumentParser
from losses.wasserstein_loss import get_wd_loss, get_wd_configuration
from dataset.dataset import SingleChannelNDIDatasetContrastiveLearningWithAug, k_fold_train_validation_split, \
    ORIGINAL_IMAGE, TARGET_IMAGE, get_CNI_tensor
from torch.utils.data import DataLoader


@dataclass
class WDArguments:
    wd_type: str = field(
        default="MSWD",
        metadata={"help": "Type of Approximation of Wasserstein distance",
                  "choices": ["None", "DSWD", "DGSWD", "MSWD", "MGSWD", "SWD", "GSWD"]}
    )

    wd_stage: int = field(
        default=2,
        metadata={"help": "Stage of applying Wasserstein distance loss"}
    )
    
    num_projections: int = field(
        default=1024,
    )
    
    wd_r: int = field(
        default=1000
    )
    
    wd_p : int = field(
        default=2
    )
    
    wd_max_iter: int = field(
        default=10
    )
    
    wd_lam: int = field(
        default=1
    )
    
    def __post_init__(self):
        assert self.wd_type is None or self.wd_type in ["DSWD", "DGSWD", "MSWD", "MGSWD", "SWD", "GSWD"], \
            "wd_type should be one of [DSWD, DGSWD, MSWD, MGSWD, SWD, GSWD]"
        
        if isinstance(self.wd_stage, int):
            self.wd_stage = [self.wd_stage]
        elif isinstance(self.wd_stage, list):
            self.wd_stage = sorted(self.wd_stage)
        else:
            raise ValueError(f"wd_stage should be int or list, but got {type(self.wd_stage)}")
        
        self = get_wd_configuration(self)


@dataclass
class CELArguments:
    cel_stage: Union[int, List, None] = field(
        default=2,
        metadata={"help": "Stage of applying Contrastive Learning"}
    )
    
    def __post_init__(self):
        self.loss_func = nn.CosineEmbeddingLoss(margin=0.5)
        
        if isinstance(self.cel_stage, int):
            self.cel_stage = [self.cel_stage]
        elif isinstance(self.cel_stage, list):
            self.cel_stage = sorted(self.cel_stage)
        else:
            raise ValueError(f"cel_stage should be int or list, but got {type(self.cel_stage)}")


@dataclass
class ModelArguments:
    seed : int = field(
        default=19981303,
        metadata={"help": "Random seed"}
    )
    
    per_device_train_batch_size: int = field(
        default=16,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training."},
    )
    
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    
    output_dir: str = field(
        default="./checkpoints",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    
    log_dir : str = field(
        default="./logs",
        metadata={"help": "The output directory where the log will be written."}
    )
    
    per_device_eval_batch_size: Union[int, None] = field(
        default=None,
        metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."},
    )

    learning_rate: float = field(
        default=0.005,
        metadata={"help": "The initial learning rate for SGD."},
    )

    weight_decay: float = field(
        default=1e-4,
        metadata={"help": "Weight decay if we apply some."},
    )

    num_train_epochs: float = field(
        default=50,
        metadata={"help": "Total number of training epochs to perform."},
    )
    
    momentum: float = field(
        default=0.9,
        metadata={"help": "Momentum of SGD optimizer."},
    )
    
    log_dir: str = field(
        default="./logs/",
        metadata={"help": "The output directory where the log will be written."}
    )
    
    log_step: int = field(
        default=10,
        metadata={"help": "The number of steps to log"}
    )
    
    pretrained_model_path: str = field(
        default='./checkpoints/ImageNet_ALL_CHECK_400_Epoch.pth',
        metadata={"help": "The path of the pretrained model to fine-tune on."}
    )


def get_model(pretrained=False):
    if pretrained:
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    else:
        model = torchvision.models.resnet50(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 512)
    return model



def train_epoch(training_args, wd_args, cel_args, model, target_tensor, metrics):

    model.train()
    train_loss = AverageMeter()


    for i, (origin, target, label) in enumerate(training_args.train_data):
        origin, target, label = origin.cuda(), target.cuda(), label.cuda()
        feature_ori, feature_tar = model(origin, target, cel_stage=cel_args.cel_stage, wd_stage=wd_args.wd_stage)
        sim_loss_1, sim_loss_2, cel_loss, wd_loss = 0, 0, 0, 0
        for j, (em_ori, em_tar) in enumerate(zip(feature_ori, feature_tar)):
            em_ori, em_tar = em_ori.view(em_ori.shape[0], -1), em_tar.view(em_tar.shape[0], -1)
            if j == 0:
                sim_mat = model.get_similarity_matrix(em_ori, em_tar)
                sim_loss_1 = training_args.criterion(sim_mat, torch.arange(0, origin.size(0)).cuda())
                sim_loss_2 = training_args.criterion(sim_mat.t(), torch.arange(0, origin.size(0)).cuda())
            elif j == 1:
                cel_loss = cel_args.loss_func(em_ori, em_tar, torch.ones((em_ori.size(0), )).cuda())
                wd_loss = get_wd_loss(wd_args, em_ori, em_tar)
        training_args.optimizer.zero_grad()
        loss = sim_loss_1 + sim_loss_2 + cel_loss + wd_loss
        loss.backward()
        train_loss.update(loss.item(), origin.size(0))
        training_args.optimizer.step()

    model.eval()
    val_acc_10 = AverageMeter()
    val_acc_20 = AverageMeter()
    val_acc_30 = AverageMeter()
    with torch.no_grad():
        for i, (origin, target, label) in enumerate(training_args.val_data):
            origin, target, label = origin.cuda(), target.cuda(), label.cuda()
            # em_ori, em_tar = model(origin, target)
            # em_ori, em_tar = em_ori[0], em_tar[0]
            # sim_mat = model.get_similarity_matrix(em_ori, em_tar)
            # loss = training_args.criterion(sim_mat, torch.arange(0, origin.size(0)).cuda()) + \
            #     training_args.criterion(sim_mat.t(), torch.arange(0, origin.size(0)).cuda())
            em_ori, em_all_tar = model(origin, target_tensor)
            em_ori, em_all_tar = em_ori[0], em_all_tar[0]
            em_ori, em_all_tar = em_ori.view(em_ori.shape[0], -1), em_all_tar.view(em_all_tar.shape[0], -1)
            sim_mat = model.get_similarity_matrix(em_ori, em_all_tar)
            acc_10, acc_20, acc_30 = cal_accuracy_top_k(sim_mat, label, top_k=(5, 10, 15))

            val_acc_10.update(acc_10.item(), origin.size(0))
            val_acc_20.update(acc_20.item(), origin.size(0))
            val_acc_30.update(acc_30.item(), origin.size(0))

    metrics.update(
        ['train_loss', 'val_acc_10', 'val_acc_20', 'val_acc_30'],
        [train_loss.avg, val_acc_10.avg, val_acc_20.avg, val_acc_30.avg]
    )


def train(training_args, wd_args, cel_args, 
          model, ):
    target_tensor = get_CNI_tensor(training_args.device, 200)
    logger = training_args.logger
    total_epochs = int(training_args.num_train_epochs)
    scheduler = training_args.scheduler
    
    fold_metrics = MetricMeter(
        ['train_loss', 'val_acc_10', 'val_acc_20', 'val_acc_30']
        )
    
    for epoch in range(total_epochs):
    
        train_epoch(
            training_args, 
            wd_args, 
            cel_args, 
            model, 
            target_tensor, 
            fold_metrics
            )

        lr_info = ''
        if scheduler:
            lr_info = f'lr {scheduler.get_last_lr()}'
            scheduler.step()

        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch: [{epoch + 1}/{total_epochs}]\n' + fold_metrics.get_last() + f'\nlr: {lr_info}\n' + f'logit_scale: {model.logit_scale}')

    return fold_metrics


logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, WDArguments, CELArguments))
    
    training_args, wd_args, cel_args = parser.parse_args_into_dataclasses()
    
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,  # if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.StreamHandler(sys.stdout), 
                  logging.FileHandler(os.path.join(training_args.log_dir, f"Pretraining_{current_time}.log"))],)

    logger.setLevel(logging.DEBUG)

    set_all_seeds(training_args.seed)
    
    training_args.metrics = MetricMeter(
        ['train_loss', 'val_acc_10', 'val_acc_20', 'val_acc_30']
        )
    training_args.base_lr = training_args.learning_rate

    for i, images in enumerate(k_fold_train_validation_split(ORIGINAL_IMAGE, TARGET_IMAGE, 7)):
        train_dataset = SingleChannelNDIDatasetContrastiveLearningWithAug(images, False, 200)
        val_dataset = SingleChannelNDIDatasetContrastiveLearningWithAug(images, True, 200)
        
        if not training_args.per_device_eval_batch_size:
            training_args.val_batch_size = len(val_dataset) 
        else:
            training_args.val_batch_size = training_args.per_device_eval_batch_size

        train_iter = DataLoader(train_dataset, training_args.per_device_train_batch_size, shuffle=True, drop_last=True)
        val_iter = DataLoader(val_dataset, batch_size=training_args.val_batch_size, shuffle=False)

        model = get_model(pretrained=True)
        model = load_checkpoints(model, './checkpoints/ImageNet_ALL_CHECK_400_Epoch.pth')  # Pre-trained NDI image model
        
        logger.info(f'Fold {i + 1} Model Loaded!')
        
        model = CompositeResNet(model, cel_args.cel_stage)
        model = RetrievalModel(model)
        model = model.cuda()

        optimizer = torch.optim.SGD(
            params=model.parameters(), 
            lr=training_args.base_lr, 
            weight_decay=training_args.weight_decay, 
            momentum=training_args.momentum
            )
        
        criterion = nn.CrossEntropyLoss()

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, 
            T_max=training_args.num_train_epochs, 
            eta_min=training_args.base_lr * 0.01
            )
        
        training_args.optimizer = optimizer
        training_args.scheduler = scheduler
        training_args.criterion = criterion
        training_args.train_data = train_iter
        training_args.val_data = val_iter
        training_args.logger = logger
        training_args.device = 'cuda'

        training_args.logger.info(f'Fold {i + 1} Training Start!')
        
        new_metrics = train(training_args, wd_args, cel_args, model)
        training_args.metrics.merge(new_metrics)
        
        training_args.logger.info(f'Fold {i + 1} Training Finished!')

    training_args.logger.info(f'All Training Finished!')
    metric_save_path = os.path.join(training_args.log_dir, 'finetuning')
    os.makedirs(metric_save_path, exist_ok=True)
    training_args.metrics.to_csv(metric_save_path, 'Epoch{}', 'Fold{}')   
    training_args.logger.info(f'All Training Metrics Saved!')
    

if __name__ == '__main__':
    main()



