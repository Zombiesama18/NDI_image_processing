import time
from dataset.dataset import get_pretraining_image_list, GaussianBlur, GaussNoise, NDIDatasetForPretraining
from torch import nn
from torchvision.models import ResNet50_Weights
import torchvision
from models.moco_model import MoCo
import datetime
from utils.utils import get_logger, AverageMeter, ProgressMeter, str2bool, cal_accuracy_top_k
from pathlib import Path
from tqdm import tqdm
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import wandb
from pretraining_config import Config


def init_wandb(args):
    wandb.login(key=args.wandb_key)
    wandb.init(project=args.project, name='your_wandb_setting', config=args.__dict__)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(len(train_loader), [
        batch_time, data_time, losses, top1, top5], prefix=f'Epoch: [{epoch}]')

    model.train()

    start = time.time()

    for i, images in enumerate(train_loader):
        data_time.update(time.time() - start)
        x1, x2 = torch.split(images, [images.size(1) // 2, images.size(1) // 2], dim=1)
        x1 = x1.cuda()
        x2 = x2.cuda()
        output, target = model(im_q=x1, im_k=x2)
        loss = criterion(output, target)

        acc1, acc5 = cal_accuracy_top_k(output, target, top_k=(1, 5))
        losses.update(loss.item(), x1.size(0))
        top1.update(acc1.item(), x1.size(0))
        top5.update(acc5.item(), x1.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - start)
        start = time.time()

        if i % 10 == 0:
            progress.display(i)
    return losses.avg, top1.avg, top5.avg


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)


def get_pretrain_model(model_path):
    base_encoder = torchvision.models.resnet50(weights=None)
    base_encoder.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    origin_dim_mlp = base_encoder.fc.in_features
    base_encoder.fc = None
    temp = torch.load(model_path)['state_dict']
    state_dict = {}
    for k, v in temp.items():
        if 'encoder_q' in k:
            if 'fc' not in k:
                state_dict['.'.join(k.split('.')[1:])] = v
    base_encoder.load_state_dict(state_dict)
    base_encoder.fc = torch.nn.Linear(origin_dim_mlp, 512)
    return base_encoder


def get_imagenet_model():
    base_encoder = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    base_encoder.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    origin_dim_mlp = base_encoder.fc.in_features
    base_encoder.fc = torch.nn.Linear(origin_dim_mlp, 512)
    return base_encoder


def get_raw_model():
    # Get raw model
    base_encoder = torchvision.models.resnet50(weights=None)
    base_encoder.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(
        7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    origin_dim_mlp = base_encoder.fc.in_features
    base_encoder.fc = torch.nn.Linear(origin_dim_mlp, 512)
    return base_encoder


def main():
    args = Config()

    if args.wandb_key:
        init_wandb(args)

    dataset_path = args.dataset_dir

    ORIGINAL_IMAGE = os.path.join(dataset_path, 'Observed')  # '../../datasets/NDI_images/Integreted/Observed/'
    EXTRA_IMAGE = os.path.join(dataset_path, 'extra observed', 'cropped')  # '../../datasets/NDI_images/Integreted/extra observed/cropped/'
    TARGET_IMAGE = os.path.join(dataset_path, 'Calculated')  # '../../datasets/NDI_images/Integreted/Calculated/'

    input_size = args.input_size
    batch_size = args.batch_size
    Epochs = args.epochs

    logger = get_logger(args.log_dir + datetime.datetime.now().strftime('%Y%m%d%H%M') + '.log')

    all_images = get_pretraining_image_list(
        ORIGINAL_IMAGE, EXTRA_IMAGE, TARGET_IMAGE)

    device = torch.device('cuda')

    start_epoch = 0
    if not args.resume:
        if args.model == 'raw':
            model = get_raw_model()
        elif 'imagenet' in args.model:
            model = get_imagenet_model()
        else:
            raise TypeError('Unsupported model type so far')
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, Epochs, eta_min=args.min_lr)
    else:
        if not args.resume_checkpoint:
            raise FileNotFoundError('A path to the checkpoint should be specified.')
        model = get_pretrain_model(args.resume_checkpoint)
        state_dict = torch.load(args.checkpoint)
        start_epoch = state_dict['scheduler']['last_epoch']
        assert start_epoch < Epochs, "Start epoch must less than the setting total epoch"
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer.load_state_dict(state_dict['optimizer'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, Epochs, eta_min=args.min_lr,
                                                               last_epoch=start_epoch)
        scheduler.load_state_dict(state_dict['scheduler'])

    model = MoCo(model, dim=512, K=1024, m=0.999, T=0.2, mlp=False, customized_model=True)
    model.cuda()

    criterion = nn.CrossEntropyLoss().cuda(device)

    all_mean = args.image_mean
    all_std = args.image_std

    normalize = transforms.Normalize(mean=all_mean, std=all_std)
    augmentation = transforms.Compose([transforms.Grayscale(3), transforms.RandomApply([transforms.RandomRotation(180)], p=0.5),
                                       transforms.RandomResizedCrop(input_size, scale=(
                                       0.2, 1.)), transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                                   transforms.RandomApply(
                                       [GaussianBlur([.1, 2.])], p=0.5), transforms.Grayscale(1),
                                   transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.ToTensor(),
                                   GaussNoise(p=0.5), normalize])

    all_images_datasets = NDIDatasetForPretraining(all_images, augmentation, img_type='L')
    all_images_dataloader = DataLoader(
        all_images_datasets, batch_size=batch_size, shuffle=True, drop_last=True)

    logger.info('Start Training!')

    best_score = 0
    for epoch in range(start_epoch, Epochs):
        loss, acc1, acc5 = train(all_images_dataloader,
                                 model, criterion, optimizer, epoch)
        logger.info(
                f'Epoch: {epoch}, loss {loss}, Acc@1 {acc1}, Acc@5 {acc5}, lr {scheduler.get_last_lr()}')
        if args.wandb_key:
            wandb.log({'epoch': epoch + 1, 'train/loss': loss, 'train/Acc@1': acc1, 'train/Acc@5': acc5})

        scheduler.step()
        score = acc1 * 2.5 + acc5
        if score > best_score:
            best_score = score
            save_checkpoint({'epoch': epoch + 1, 'arch': 'ResNet50', 'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                             'norms': [all_mean, all_std]},
                            filename='./checkpoints/UNICOM_ViT_B_32_based.pth')
            logger.info(
                    f'Save Checkpoint at {epoch + 1} with loss {loss}, Acc@1 {acc1}, Acc@5 {acc5}, score {score}')
        if (epoch + 1) % args.save_steps == 0:
            save_checkpoint({'epoch': epoch + 1, 'arch': 'ResNet50', 'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                             'norms': [all_mean, all_std]},
                            filename=f'./checkpoints/UNICOM_ViT_B_32_{epoch + 1}_Epoch.pth')
            logger.info(
                    f'Save Checkpoint at {epoch + 1} with loss {loss}, Acc@1 {acc1}, Acc@5 {acc5}, score {score}')
    logger.info('Training Finished!')


if __name__ == '__main__':
    main()

