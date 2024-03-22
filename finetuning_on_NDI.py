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

from models.core_model import CalculatedModel
from utils import MetricMeter, AverageMeter, set_all_seeds, cal_accuracy_top_k, load_checkpoints, HfArgumentParser
from dataset.dataset import create_train_val_dataset, RandomRotationWithAngle
from losses.focal_loss import focal_loss
import torchvision


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
    
    annotation_file_path: str = field(
        default="../datasets/NDI_images/annotation.csv",
        metadata={"help": "The directory of the annotation file of the dataset."}
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



def train_epoch(training_args, model, metrics):

    model.train()
    loss = AverageMeter()
    train_class_loss = AverageMeter()
    train_angle_loss = AverageMeter()

    for i, (img_tensor, class_label, angle_label) in enumerate(training_args.train_loader):
        img_tensor, class_label, angle_label = img_tensor.to(training_args.device), class_label.to(training_args.device), angle_label.to(training_args.device)
        img_tensor, angle_label = img_tensor.double(), angle_label.double()
        angle_label = angle_label.view(-1, 1)
        training_args.optimizer.zero_grad()
        class_logits, angle_logits = model(img_tensor)
        batch_loss, class_loss, angle_loss = model.compute_loss(
            (training_args.class_criterion, training_args.angle_criterion),
            (class_logits, angle_logits),
            (class_label, angle_label)
        )
        batch_loss.backward()
        training_args.optimizer.step()
        loss.update(batch_loss.item(), img_tensor.shape[0])
        train_class_loss.update(class_loss.item(), img_tensor.shape[0])
        train_angle_loss.update(angle_loss.item(), img_tensor.shape[0])

    model.eval()
    val_class_acc = AverageMeter()
    val_angle_loss = AverageMeter()
    with torch.no_grad():
        for img_tensor, class_label, angle_label in training_args.val_loader:
            img_tensor, class_label, angle_label = img_tensor.to(training_args.device), class_label.to(training_args.device), angle_label.to(training_args.device)
            img_tensor, angle_label = img_tensor.double(), angle_label.double()
            angle_label = angle_label.view(-1, 1)
            class_logits, angle_logits = model(img_tensor)
            class_acc = cal_accuracy_top_k(class_logits, class_label)[0]
            angle_loss = training_args.angle_criterion(angle_logits, angle_label)
            val_class_acc.update(class_acc.item(), img_tensor.shape[0])
            val_angle_loss.update(angle_loss.item(), img_tensor.shape[0])
    
    training_args.logger.info(f'Epoch {training_args.epoch}, Loss: {loss.avg}, Val Class Acc: {val_class_acc.avg}, Val Angle Loss: {val_angle_loss.avg}')

    metrics.update(
        ['train_loss', 'val_cls_acc', 'val_ang_loss'],
        [loss.avg, val_class_acc.avg, val_angle_loss.avg]
    )


def train(training_args, model, ):
    
    total_epochs = int(training_args.num_train_epochs)
    
    metrics = MetricMeter(
        ['train_loss', 'val_cls_acc', 'val_ang_loss']
        )
    
    for epoch in range(total_epochs):
        training_args.epoch = epoch + 1
        train_epoch(
            training_args, 
            model,  
            metrics
            )

    return metrics


logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(ModelArguments)
    
    training_args = parser.parse_args_into_dataclasses()[0]
    
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,  # if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.StreamHandler(sys.stdout), 
                  logging.FileHandler(os.path.join(training_args.log_dir, f"Pretraining_{current_time}.log"))],)

    logger.setLevel(logging.DEBUG)

    set_all_seeds(training_args.seed)
    
    training_args.base_lr = training_args.learning_rate

    train_set, val_set = create_train_val_dataset(training_args.annotation_file_path)
    
    model = get_model(pretrained=True)
    model = load_checkpoints(model, training_args.pretrained_model_path)  # Pre-trained NDI image model
    model = CalculatedModel(model, train_set.num_classes)

    training_args.device = 'cuda'
    
    class_criterion = focal_loss(None, 3.5, device=training_args.device, dtype=torch.double)
    angle_criterion = torch.nn.MSELoss()
    
    optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.005,
    momentum=0.9,
    weight_decay=1e-4
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=len(val_set)
    )

    model.to(training_args.device)
    model.double()
    
    training_args.train_loader = train_loader
    training_args.val_loader = val_loader
    training_args.optimizer = optimizer
    training_args.class_criterion = class_criterion
    training_args.angle_criterion = angle_criterion
    training_args.logger = logger
            
    training_args.metrics = train(training_args, model)

    training_args.logger.info(f'All Training Finished!')
    metric_save_path = os.path.join(training_args.log_dir, 'finetuning')
    os.makedirs(metric_save_path, exist_ok=True)
    training_args.metrics.to_csv(metric_save_path, 'Epoch{}')   
    training_args.logger.info(f'All Training Metrics Saved!')
    

if __name__ == '__main__':
    main()



