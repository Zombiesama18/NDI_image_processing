import time
from finetuning_config import Config
import torch
import torch.nn as nn

import torchvision.models

from models.vit_model import build_model
from models.core_model import RetrievalModel
from models.wd_model import TransformNet, CompositeResNet
from utils.utils import get_logger, AverageMeter, ProgressMeter, set_all_seeds, cal_accuracy_top_k, load_checkpoints
import datetime
import wandb
from losses.wasserstein_loss import distributional_sliced_wasserstein_distance, distributional_generalized_sliced_wasserstein_distance, \
            max_generalized_sliced_wasserstein_distance, max_sliced_wasserstein_distance, sliced_wasserstein_distance, \
            generalized_sliced_wasserstein_distance, circular_function
from dataset.dataset import SingleChannelNDIDatasetContrastiveLearningWithAug, k_fold_train_validation_split, \
    ORIGINAL_IMAGE, TARGET_IMAGE, get_CNI_tensor
from torch.utils.data import DataLoader
import unicom


def init_wandb(args, config=None, **kwargs):
    if config is None:
        config = dict()
    wandb.login(key=args.wandb_key)
    wandb.init(config=config, **kwargs)


def get_model(key_word, pretrained=False):
    if key_word == 'ResNet50':
        # Set ResNet50 for clarifying the effectiveness of Code
        if pretrained:
            model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            model = torchvision.models.resnet50(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 512)
        return model
    if 'ViT' in key_word:
        if pretrained:
            model = unicom.load(key_word)[0]
        else:
            model = build_model(key_word)
        return model


def get_wd_loss(first_samples, second_samples, wd_utils, index):
    loss = 0
    if wd_utils['type'] == 'DSWD':
        loss = distributional_sliced_wasserstein_distance(
            first_samples, second_samples, wd_utils['num_projections'], wd_utils[f'tran_net'],
            wd_utils[f'op_trannet'], wd_utils['p'], wd_utils['max_iter'], wd_utils['lam'], first_samples.device
        )
    elif wd_utils['type'] == 'DGSWD':
        loss = distributional_generalized_sliced_wasserstein_distance(
            first_samples, second_samples, wd_utils['num_projections'], wd_utils[f'tran_net'],
            wd_utils[f'op_trannet'], wd_utils['g_func'], wd_utils['r'], wd_utils['p'], wd_utils['max_iter'],
            wd_utils['lam'], first_samples.device
        )
    elif wd_utils['type'] == 'MSWD':
        loss, _ = max_sliced_wasserstein_distance(first_samples, second_samples, wd_utils['p'], wd_utils['max_iter'],
                                                  first_samples.device)
    elif wd_utils['type'] == 'MGSWD':
        loss = max_generalized_sliced_wasserstein_distance(
            first_samples, second_samples, wd_utils['g_func'], wd_utils['r'], wd_utils['p'], wd_utils['max_iter'],
            first_samples.device
        )
    elif wd_utils['type'] == 'SWD':
        loss = sliced_wasserstein_distance(first_samples, second_samples, wd_utils['num_projections'], wd_utils['p'],
                                           first_samples.device)
    elif wd_utils['type'] == 'GSWD':
        loss = generalized_sliced_wasserstein_distance(
            first_samples, second_samples, wd_utils['g_func'], wd_utils['r'], wd_utils['num_projections'], wd_utils['p'],
            first_samples.device
        )
    return loss


def train_epoch(train_data, val_data, model, criterion, optimizer, current_epoch, total_epoch, target_tensor,
                wd_utils=None, cel_utils=None):
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
        feature_ori, feature_tar = model(origin, target, cel_stage=cel_utils['stage'], wd_stage=wd_utils['stage'])
        loss = 0
        for j, (em_ori, em_tar) in enumerate(zip(feature_ori, feature_tar)):
            em_ori, em_tar = em_ori.view(em_ori.shape[0], -1), em_tar.view(em_tar.shape[0], -1)
            if j == 0:
                sim_mat = model.get_similarity_matrix(em_ori, em_tar)
                loss += criterion(sim_mat, torch.arange(0, origin.size(0)).cuda())
                loss += criterion(sim_mat.t(), torch.arange(0, origin.size(0)).cuda())
            elif j == 1:
                if cel_utils:
                    loss += cel_utils['loss_func'](em_ori, em_tar, torch.ones((em_ori.size(0), )).cuda())
                elif wd_utils:
                    loss += get_wd_loss(em_ori, em_tar, wd_utils, wd_utils['stage'])
            elif j == 2:
                loss += get_wd_loss(em_ori, em_tar, wd_utils, wd_utils['stage'])
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
            em_ori, em_tar = em_ori[0], em_tar[0]
            sim_mat = model.get_similarity_matrix(em_ori, em_tar)
            loss = criterion(sim_mat, torch.arange(0, origin.size(0)).cuda()) + \
                criterion(sim_mat.t(), torch.arange(0, origin.size(0)).cuda())
            em_ori, em_all_tar = model(origin, target_tensor)
            em_ori, em_all_tar = em_ori[0], em_all_tar[0]
            sim_mat = model.get_similarity_matrix(em_ori, em_all_tar)
            acc_10, acc_20, acc_30 = cal_accuracy_top_k(sim_mat, label, top_k=(5, 10, 15))

            val_loss.update(loss.item(), origin.size(0))
            val_acc_10.update(acc_10.item(), origin.size(0))
            val_acc_20.update(acc_20.item(), origin.size(0))
            val_acc_30.update(acc_30.item(), origin.size(0))

        val_progress.display(i)

    return train_loss.avg, val_loss.avg, val_acc_10.avg, val_acc_20.avg, val_acc_30.avg


def train(args, logger, train_data, val_data, model, criterion, optimizer, total_epochs, save_folder='./',
          scheduler=None, wandb_config=None, device=None, wd_utils=None, cel_utils=None):
    target_tensor = get_CNI_tensor(device, 200)

    for epoch in range(total_epochs):
        train_loss, val_loss, val_acc_10, val_acc_20, val_acc_30 = \
            train_epoch(train_data, val_data, model, criterion, optimizer, epoch + 1, total_epochs, target_tensor,
                        wd_utils, cel_utils)

        lr_info = ''
        if scheduler:
            lr_info = f'lr {scheduler.get_last_lr()}'
            scheduler.step()

        logger.info(f'Epoch: [{epoch}/{total_epochs}], train loss {train_loss}, '
                    f'val loss {val_loss}, val acc @ 5 {val_acc_10}, val acc @ 10 {val_acc_20}, '
                    f'val acc @ 15 {val_acc_30}' + lr_info)

        if wandb_config:
            wandb.log({'epoch': epoch + 1, 'train/train loss': train_loss,
                       'val/val loss': val_loss, 'val/val acc @ 5': val_acc_10, 'val/val acc @ 10': val_acc_20,
                       'val/val acc @ 15': val_acc_30})
    logger.info('Training Finished!')


def main():
    args = Config()

    device = torch.device(args.device)

    set_all_seeds(args.seed)

    logger_fname = datetime.datetime.now().strftime('%Y%m%d%H%M')
    logger = get_logger(f'./logs/{logger_fname}.log')

    logger.info(f'This training is to do: {args.message_to_log}')

    wd_stage, cel_stage = 2, 2

    test_list = ['MSWD']

    for test_item in test_list:

        wd_name = test_item

        for i, images in enumerate(k_fold_train_validation_split(ORIGINAL_IMAGE, TARGET_IMAGE, 7)):
            if args.wandb_key:
                init_wandb(args=args, config=args.__dict__)
            train_dataset = SingleChannelNDIDatasetContrastiveLearningWithAug(images, False, 200)
            val_dataset = SingleChannelNDIDatasetContrastiveLearningWithAug(images, True, 200)
            train_iter = DataLoader(train_dataset, args.batch_size, shuffle=True, drop_last=True)
            val_iter = DataLoader(val_dataset, batch_size=len(val_dataset))

            model = get_model('ResNet50', pretrained=True)
            model = load_checkpoints(model, './checkpoints/ImageNet_ALL_CHECK_400_Epoch.pth')  # Pre-trained NDI image model
            model = CompositeResNet(model, cel_stage)
            model = RetrievalModel(model)
            model = model.cuda()

            optimizer = torch.optim.SGD(params=model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay, momentum=args.momentum)
            criterion = nn.CrossEntropyLoss()

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=args.base_lr * 0.01)
            wd_utils = {}
            if wd_name == 'DSWD':
                wd_utils = {
                    'type': wd_name,
                    'num_projections': 1024,
                    'p': 2,
                    'max_iter': 10,
                    'lam': 1
                }
                channels = [512, 2048, 1024, 512, 256, 64]
                transform_net = TransformNet(channels[wd_stage - 1]).to('cuda')
                optimizer_tran_net = torch.optim.Adam(transform_net.parameters(), lr=0.0005, betas=(0.5, 0.999))
                wd_utils[f'tran_net'] = transform_net
                wd_utils[f'op_trannet'] = optimizer_tran_net

            elif wd_name == 'DGSWD':
                wd_utils = {
                    'type': wd_name,
                    'num_projections': 1024,
                    'g_func': circular_function,
                    'r': 1000,
                    'p': 2,
                    'max_iter': 10,
                    'lam': 1
                }
                channels = [512, 2048, 1024, 512, 256, 64]
                transform_net = TransformNet(channels[wd_stage - 1]).to('cuda')
                optimizer_tran_net = torch.optim.Adam(transform_net.parameters(), lr=0.0005, betas=(0.5, 0.999))
                wd_utils[f'tran_net'] = transform_net
                wd_utils[f'op_trannet'] = optimizer_tran_net

            elif wd_name == 'MSWD':
                wd_utils = {
                    'type': wd_name,
                    'p': 2,
                    'max_iter': 10
                }

            elif wd_name == 'MGSWD':
                wd_utils = {
                    'type': wd_name,
                    'g_func': circular_function,
                    'r': 1000,
                    'p': 2,
                    'max_iter': 10
                }

            elif wd_name == 'SWD':
                wd_utils = {
                    'type': wd_name,
                    'num_projections': 1000,
                    'p': 2
                }

            elif wd_name == 'GSWD':
                wd_utils = {
                    'type': wd_name,
                    'g_func': circular_function,
                    'r': 1000,
                    'num_projections': 1000,
                    'p': 2
                }

            if wd_utils:
                wd_utils['stage'] = 2
            
            cel_utils = {}
            cel_utils['loss_func'] = nn.CosineEmbeddingLoss(margin=0.5)

            if cel_utils:
                cel_utils['stage'] = 2
            
            train(args, logger, train_iter, val_iter, model, criterion, optimizer, args.epochs, scheduler=scheduler,
                  save_folder=args.output_dir, wandb_config=True, device=device, wd_utils=wd_utils, cel_utils=cel_utils)
            wandb.finish()


if __name__ == '__main__':
    main()



