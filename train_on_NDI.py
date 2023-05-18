import time
from config import Config
import timm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as f
import argparse

import torchvision.models

from models.vit_model import vit_tiny, vit_base, vit_large, vit_small
from models.core_model import RetrievalModel
from timm.models import create_model
from timm.models.swin_transformer_v2 import PatchEmbed
from utils import get_logger, AverageMeter, ProgressMeter, set_all_seeds, get_wandb_API_key, cal_accuracy_top_k, load_checkpoints
import os
import datetime
import wandb
from dataset import SingleChannelNDIDatasetContrastiveLearningWithAug, k_fold_train_validation_split, \
    ORIGINAL_IMAGE, TARGET_IMAGE, get_CNI_tensor
from torch.utils.data import DataLoader


def get_args_parser():
    parser = argparse.ArgumentParser('Fine-tuning on NDI images', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=50, type=int)

    # Model parameters
    parser.add_argument('--model', default='swin_base', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float,
                        help="Layer scale initial values")

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=5e-3, metavar='LR',
                        help='learning rate (absolute lr)')
    
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum of SGD')

    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cosine schedulers that hit 0')

    # Dataset Parameters
    parser.add_argument('--output_dir', default='./checkpoints/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./logs/',
                        help='path where to save the log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=19981303, type=int)

    # Wandb Parameters
    parser.add_argument('--project', default='Test which Swin suits NDI images best', type=str,
                        help="The name of the W&B project where you're sending the new run.")

    return parser


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


def init_wandb(config=None, **kwargs):
    if config is None:
        config = dict()
    wandb.login(key=get_wandb_API_key())
    wandb.init(config=config, **kwargs)


def get_model(key_word, pretrained=False):
    assert key_word in ['vit_tiny', 'vit_small', 'vit_base', 'vit_large', 'ResNet50', 'convnext_base', 'convnext_small', 'swin_tiny', 'swin_base', 'swin_tiny']
    if key_word == 'vit_tiny':
        return vit_tiny(pretrained=pretrained, num_classes=0, in_chans=1)
    if key_word == 'vit_small':
        return vit_small(pretrained=pretrained, num_classes=0, in_chans=1)
    if key_word == 'vit_base':
        return vit_base(pretrained=pretrained, num_classes=0, in_chans=1)
    if key_word == 'vit_large':
        return vit_large(pretrained=pretrained, num_classes=0, in_chans=1)
    if key_word == 'ResNet50':
        # Set ResNet50 for clarifying the effectiveness of Code
        model = torchvision.models.resnet50()
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 512)
        return model
    if key_word == 'convnext_base':
        model = timm.create_model('convnext_base_in22ft1k', pretrained=True)
        model.stem[0] = nn.Conv2d(1, 128, kernel_size=4, stride=4)
        model.head.fc = nn.Linear(1024, 512, bias=True)
        return model
    if key_word == 'convnext_small':
        model = timm.create_model('convnext_small_in22ft1k', pretrained=True)
        model.stem[0] = nn.Conv2d(1, 96, kernel_size=4, stride=4)
        model.head.fc = nn.Identity()
        return model
    if key_word == 'swin_base':
        model = timm.create_model('swinv2_base_window12_192_22k', pretrained=True)
        model.patch_embed = PatchEmbed(img_size=192, patch_size=4, in_chans=1, embed_dim=128, norm_layer=nn.LayerNorm)
        model.head = nn.Linear(1024, 512, bias=True)
        return model
    if key_word == 'swin_tiny':
        model = timm.create_model('swinv2_tiny_window16_256', pretrained=True)
        model.patch_embed = PatchEmbed(img_size=256, patch_size=4, in_chans=1, embed_dim=96, norm_layer=nn.LayerNorm)
        model.head = nn.Linear(768, 512, bias=True)
        return model


def train_epoch(train_data, val_data, model, criterion, optimizer, current_epoch, total_epoch, target_tensor):
    batch_time = AverageMeter('Batch Time', ':6.3f')
    data_time = AverageMeter('Data Time', ':6.3f')
    train_loss = AverageMeter('Train Loss', ':.4e')
    val_loss = AverageMeter('Val Loss', ':.4e')
    val_acc_10 = AverageMeter('Val Acc@10', ':6.2f')
    val_acc_20 = AverageMeter('Val Acc@20', ':6.2f')
    val_acc_30 = AverageMeter('Val Acc@30', ':6.2f')

    train_progress = ProgressMeter(len(train_data), [batch_time, data_time, train_loss],
                                   prefix=f'Training Progress\tEpoch: [{current_epoch}/{total_epoch}]')
    val_progress = ProgressMeter(len(val_data), [val_loss, val_acc_10, val_acc_20, val_acc_30],
                                 prefix=f'Validation Progress\tEpoch: [{current_epoch}/{total_epoch}]')

    model.train()
    start = time.time()

    for i, (origin, target, label) in enumerate(train_data):
        data_time.update(time.time() - start)
        origin, target, label = origin.cuda(), target.cuda(), label.cuda()
        em_ori, em_tar = model(origin, target)
        sim_mat = model.get_similarity_matrix(em_ori, em_tar)
        loss1 = criterion(sim_mat, torch.arange(0, origin.size(0)).cuda())
        loss2 = criterion(sim_mat.t(), torch.arange(0, origin.size(0)).cuda())
        loss = loss1 + loss2

        train_loss.update(loss.item(), origin.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - start)
        start = time.time()

        if i % 5 == 0:
            train_progress.display(i)

    model.eval()
    with torch.no_grad():
        for i, (origin, target, label) in enumerate(val_data):
            origin, target, label = origin.cuda(), target.cuda(), label.cuda()
            em_ori, em_tar = model(origin, target)
            sim_mat = model.get_similarity_matrix(em_ori, em_tar)
            loss = criterion(sim_mat, torch.arange(0, origin.size(0)).cuda()) + \
                   criterion(sim_mat.t(), torch.arange(0, origin.size(0)).cuda())
            em_ori, em_all_tar = model(origin, target_tensor)
            sim_mat = model.get_similarity_matrix(em_ori, em_all_tar)
            acc_10, acc_20, acc_30 = cal_accuracy_top_k(sim_mat, label, top_k=(10, 20, 30))

            val_loss.update(loss.item(), origin.size(0))
            val_acc_10.update(acc_10.item(), origin.size(0))
            val_acc_20.update(acc_20.item(), origin.size(0))
            val_acc_30.update(acc_30.item(), origin.size(0))

        val_progress.display(i)

    return train_loss.avg, val_loss.avg, val_acc_10.avg, val_acc_20.avg, val_acc_30.avg


def train(args, logger, train_data, val_data, model, criterion, optimizer, total_epochs, save_folder='./',
          scheduler=None, wandb_config=None, device=None):
    best_score = 0.

    target_tensor = get_CNI_tensor(device, 200)

    for epoch in range(total_epochs):
        train_loss, val_loss, val_acc_10, val_acc_20, val_acc_30 = \
            train_epoch(train_data, val_data, model, criterion, optimizer, epoch + 1, total_epochs, target_tensor)

        lr_info = ''
        if scheduler:
            lr_info = f'lr {scheduler.get_last_lr()}'
            scheduler.step()

        logger.info(f'Epoch: [{epoch}/{total_epochs}], train loss {train_loss}, '
                    f'val loss {val_loss}, val acc @ 10 {val_acc_10}, val acc @ 20 {val_acc_20}, '
                    f'val acc @ 30 {val_acc_30}' + lr_info)

        if wandb_config:
            wandb.log({'epoch': epoch + 1, 'train/train loss': train_loss,
                       'val/val loss': val_loss, 'val/val acc @ 10': val_acc_10, 'val/val acc @ 20': val_acc_20,
                       'val/val acc @ 30': val_acc_30})

        # if val_acc_10 > best_score:
        #     best_score = val_acc_10
        #     checkpoint_to_save = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
        #                           'optimizer': optimizer.state_dict}
        #     if scheduler:
        #         checkpoint_to_save.update(scheduler=scheduler.state_dict())
        #     if wandb_config:
        #         filename = args.project + f'epoch {epoch + 1}'
        #     else:
        #         filename = datetime.datetime.now().strftime('%Y%m%d%H%M')
        #     save_checkpoint(checkpoint_to_save, filename=save_folder + filename)
        #     logger.info(f'Save Checkpoint at {epoch + 1} with train loss {train_loss}, '
        #                 f'val loss {val_loss}, val acc @ 10 {val_acc_10}, val acc @ 20 {val_acc_20}, '
        #                 f'val acc @ 30 {val_acc_30}' + lr_info)

    logger.info('Training Finished!')


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)


def main():
    args = Config()
    # parser = argparse.ArgumentParser('Fine-tuning on NDI images', parents=[get_args_parser()])
    # args = parser.parse_args()

    device = torch.device(args.device)

    set_all_seeds(args.seed)

    logger_fname = datetime.datetime.now().strftime('%Y%m%d%H%M')
    logger = get_logger(f'./logs/{logger_fname}.log')

    logger.info(f'This training is to do: {args.message_to_log}')

    model_list = ['ResNet50']
    # model_list = ['vit_tiny', 'vit_small', 'vit_base', 'vit_large']

    for model_name in model_list:
        for i, images in enumerate(k_fold_train_validation_split(ORIGINAL_IMAGE, TARGET_IMAGE, 7)):
            wandb.init(project=args.message_to_log, group='Re_implementation', job_type=f'{model_name}_original_setting',
                       name=f'{model_name}_fold {i}', config=args.__dict__)
            train_dataset = SingleChannelNDIDatasetContrastiveLearningWithAug(images, False, 200)
            val_dataset = SingleChannelNDIDatasetContrastiveLearningWithAug(images, True, 200)
            train_iter = DataLoader(train_dataset, args.batch_size, shuffle=True, drop_last=True)
            val_iter = DataLoader(val_dataset, batch_size=len(val_dataset))

            model = get_model(model_name, pretrained=True)
            model = load_checkpoints(model, './checkpoints/ImageNet_ALL_CHECK_400_Epoch.pth')
            model = RetrievalModel(model)
            model = model.cuda()

            optimizer = torch.optim.SGD(params=model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay, momentum=args.momentum)
            criterion = nn.CrossEntropyLoss()

            # scheduler = None
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=args.base_lr * 0.01)

            train(args, logger, train_iter, val_iter, model, criterion, optimizer, args.epochs, scheduler=scheduler,
                  save_folder=args.output_dir, wandb_config=True, device=device)
            wandb.finish()

if __name__ == '__main__':
    main()
