from collections import defaultdict
import random
from git import Union
import torch
from pathlib import Path
from torchvision import transforms
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import math


ORIGINAL_IMAGE = '../datasets/NDI_images/Integreted/Observed/'
EXTRA_IMAGE = '../datasets/NDI_images/Integreted/extra observed/cropped/'
TARGET_IMAGE = '../datasets/NDI_images/Integreted/Calculated/'


def k_fold_train_validation_split(original_path, target_path, k=5):
    original_images = list(sorted(list(map(str, list(Path(original_path).glob('*.jpg'))))))
    target_images = list(sorted(list(map(str, list(Path(target_path).glob('*.jpg'))))))
    images = list(reversed(list(zip(original_images, target_images))))
    per_k = 24
    for i in range(k):
        train_images = images[: i * per_k] + images[(i + 1) * per_k:]
        val_images = images[i * per_k: (i + 1) * per_k]
        yield train_images, val_images


def split_train_validation_randomly(original_path, target_path):
    original_images = list(sorted(list(map(str, list(Path(original_path).glob('*.jpg'))))))
    target_images = list(sorted(list(map(str, list(Path(target_path).glob('*.jpg'))))))
    images = list(zip(original_images, target_images))
    train_images, val_images = torch.utils.data.random_split(images, [160, 24])
    return train_images, val_images


class SingleChannelNDIDatasetContrastiveLearningWithAug(Dataset):
    def __init__(self, images, evaluate=False, target_size=200):
        super(SingleChannelNDIDatasetContrastiveLearningWithAug, self).__init__()
        if not evaluate:
            self.images = images[0]
        else:
            self.images = images[1]
        self.train_transforms = transforms.Compose([
            # transforms.GaussianBlur(kernel_size=3, sigma=0.7),
            # transforms.CenterCrop(150),
            # transforms.Resize(200),
            # transforms.RandomHorizontalFlip(0.5),
            # transforms.RandomRotation(30)
            transforms.Resize(target_size)
        ])
        self.eval_transforms = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor()
        ])
        self.evaluate = evaluate

    def __getitem__(self, idx):
        origin_path, target_path = self.images[idx]
        origin = Image.open(origin_path).convert('L')
        target = Image.open(target_path).convert('L')
        if not self.evaluate:
            origin, target = self.train_transforms(
                torch.cat((transforms.ToTensor()(origin).unsqueeze(0), transforms.ToTensor()(target).unsqueeze(0)),
                          dim=0))
        else:
            origin, target = self.eval_transforms(origin), self.eval_transforms(target)
        label = int(origin_path.split('/')[-1].split('.')[0]) - 1
        return origin, target, label

    def __len__(self):
        return len(self.images)


class ThreeChannelNDIDataset(Dataset):
    def __init__(self, images, evaluate=False, target_size=200):
        super(ThreeChannelNDIDataset, self).__init__()
        if not evaluate:
            self.images = images[0]
        else:
            self.images = images[1]
        self.train_transforms = transforms.Compose([
            transforms.Resize(target_size)
        ])
        self.eval_transforms = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor()
        ])
        self.evaluate = evaluate

    def __getitem__(self, idx):
        origin_path, target_path = self.images[idx]
        origin = Image.open(origin_path).convert('RGB')
        target = Image.open(target_path).convert('RGB')
        if not self.evaluate:
            origin, target = self.train_transforms(
                torch.cat((transforms.ToTensor()(origin).unsqueeze(0), transforms.ToTensor()(target).unsqueeze(0)),
                          dim=0))
        else:
            origin, target = self.eval_transforms(origin), self.eval_transforms(target)
        label = int(origin_path.split('/')[-1].split('.')[0]) - 1
        return origin, target, label

    def __len__(self):
        return len(self.images)

def get_pretraining_image_list(origin_path, extra_path, target_path):
    all_images = list(map(str, Path(origin_path).glob('*.jpg')))
    all_images.extend(list(map(str, Path(extra_path).glob('*.jpg'))))
    observed_len = len(all_images)
    target_images = list(map(str, Path(target_path).glob('*.jpg')))
    for _ in range(observed_len // len(target_images)):
        all_images.extend(target_images)
    return all_images


class NDIDatasetForPretraining(Dataset):
    def __init__(self, images, transforms, grayscale=True, weight_gamma=None, img_type='L') -> None:
        super().__init__()
        self.images = images
        if weight_gamma is not None and isinstance(images, dict):
            self.weights = self._example_weights(self.images, weight_gamma)
        else:
            self.weights = None
        self.transforms = transforms
        self.img_type = img_type
    
    def _example_weights(img_path_dict, gamma=0.3):
        counts = np.array([len(paths) for paths in img_path_dict.values()])
        
        weights = 1 / counts
        weights = weights ** gamma
        
        total_weights = weights.sum()
        weights /= total_weights
        
        example_weights = []
        for w, c in zip(weights, counts):
            example_weights.extend([w] * c)
        
        return torch.tensor(example_weights)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        file = self.images[index]
        if self.img_type == 'L':
            image = Image.open(file).convert('L')
        else:
            image = Image.open(file).convert('RGB')
        image1 = self.transforms(image)
        image2 = self.transforms(image)
        return torch.cat([image1, image2], dim=0)
            
class GaussianBlur:
    def __init__(self, sigma=[0.1, 2.]) -> None:
        self.sigma = sigma
    
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class GaussNoise:
    def __init__(self, var_limit=(1e-5, 1e-4), p=0.5) -> None:
        self.var_limit = np.log(var_limit)
        self.p = p
    
    def __call__(self, image):
        if np.random.random() < self.p:
            sigma = np.exp(np.random.uniform(*self.var_limit)) ** 0.5
            noise = np.random.normal(0, sigma, size=image.shape).astype(np.float32)
            image = image + torch.from_numpy(noise)
            image = torch.clamp(image, 0, 1)
        return image


def get_CNI_tensor(device=None, target_size=200, img_type='L'):
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])
    target_tensor = []
    for i in range(1, 185):
        if img_type == 'L':
            target_tensor.append(
                transform(Image.open(str(Path.joinpath(Path(TARGET_IMAGE), f'{i}.jpg'))).convert('L')).unsqueeze(0))
        else:
            target_tensor.append(
                transform(Image.open(str(Path.joinpath(Path(TARGET_IMAGE), f'{i}.jpg'))).convert('RGB')).unsqueeze(0))
    target_tensor = torch.cat(target_tensor, dim=0)
    if device:
        return target_tensor.to(device)
    return target_tensor

