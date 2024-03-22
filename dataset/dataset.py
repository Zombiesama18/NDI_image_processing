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

def create_train_val_dataset(annotation_file, train_transforms=None, val_transforms=None, color_mode='L', ratio=0.15):        
    
    df = pd.read_csv(annotation_file)
    base_path = Path(annotation_file).parent
    img_grouped_by_class = {}
    
    classes = list(set(df['class'].to_list()))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    class_idx_to_cal_refs = {}
    
    for idx, row in iter(df.iterrows()):
        class_name = row['class']
        angle_label = row['angle'] / 90
        if class_name not in img_grouped_by_class:
            img_grouped_by_class[class_name] = []

        if class_to_idx[class_name] not in class_idx_to_cal_refs:
            class_idx_to_cal_refs[class_to_idx[class_name]] = str(base_path.joinpath(row['ref_path']))
        
        file_path = str(base_path.joinpath(row['observed_paths']))
        if 'ref.jpg' in file_path:
            continue
        class_label = class_to_idx[class_name]
        img_grouped_by_class[class_name].append((file_path, class_label, angle_label))
        
    if isinstance(ratio, float):
        ratio = (1 - ratio, ratio) if ratio < 0.5 else (ratio, 1 - ratio)
    
    train_img_list, val_img_list, train_label_list, val_label_list = [], [], [], []
    class_idx_to_obs_refs = {}
    
    for class_name, img_list in img_grouped_by_class.items():
        class_label = class_to_idx[class_name]
        class_idx_to_obs_refs[class_label] = [(item[0], item[2]) for item in img_list]
        
        repeat_times = max(0, 15 - len(img_list))
        img_list = img_list + [(None, class_to_idx[class_name], None) for _ in range(repeat_times)]
        
        random.shuffle(img_list)
        val_length = max(1, int(len(img_list) * ratio[1]))
        val_img_list.extend([item[0] for item in img_list[:val_length]])
        val_label_list.extend([(item[1], item[2]) for item in img_list[:val_length]])
        train_img_list.extend([item[0] for item in img_list[val_length:]])
        train_label_list.extend([(item[1], item[2]) for item in img_list[val_length:]])
        
    dataset_attr = {
        'num_classes': num_classes,
        'class_to_idx': class_to_idx,
        'color_mode': color_mode,
        'class_idx_to_cal_refs': class_idx_to_cal_refs,
        'class_idx_to_obs_refs': class_idx_to_obs_refs
    }
    
    print(f'Train/Val split: {len(train_img_list)} / {len(val_img_list)}')
    print(f'split ratio: {(len(train_img_list) / (len(train_img_list) + len(val_img_list)))}' +  
          f'/ {(len(val_img_list) / (len(train_img_list) + len(val_img_list)))}')
    
    return SimpleImageDataset(train_img_list, train_label_list, train_transforms, img_type=color_mode, **dataset_attr), \
        SimpleImageDataset(val_img_list, val_label_list, val_transforms, img_type=color_mode, **dataset_attr)
    
class SimpleImageDataset(Dataset):
    def __init__(self, images, labels, transforms=None, img_type='L', **kwargs) -> None:
        super().__init__()
        self.images = images
        self.labels = labels
        self.transforms = transforms
        self.img_type = img_type
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        labels = self.labels[idx]
        class_label, angle_label = labels
        if not img_path:
            ref_path, ref_angle = random.choice(self.class_idx_to_obs_refs[class_label])
            ref_angle = int(ref_angle * 90)
            rotate_angle = random.randint(-90 - ref_angle, 89 - ref_angle)
            img = rotate_and_fill(ref_path, rotate_angle, fill_color=5, img_type=self.img_type)
            angle_label = rotate_angle + ref_angle
            labels = (class_label, angle_label / 90)
        else:
            img = Image.open(img_path).convert(self.img_type)
        if self.transforms:
            img, labels = self.transforms(img, labels)
        else:
            to_tensor = transforms.ToTensor()
            img = to_tensor(img)
        return img, *labels
        
        
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


class RandomRotationWithAngle(transforms.RandomRotation):
    def __init__(self, degrees, expand=False, center=None, fill=32):
        super(RandomRotationWithAngle, self).__init__(degrees, expand=expand, center=center, fill=fill)
        self.angle = None
    
    def forward(self, img):
        angle = self.get_params(self.degrees)
        self.angle = angle
        return super().forward(img)
    
    def get_rotate_angle(self):
        if not self.angle:
            print('Please call get_rotate_angle() after calling forward()')
        output_angle = self.angle
        self.angle = None
        return output_angle

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

def rotate_and_fill(img_path, angle, fill_color=(32, 32, 32), img_type='L'):
    img = Image.open(img_path).convert(img_type)
    rotated_img = img.rotate(angle, expand=True, fillcolor=fill_color)
    
    rotated_center = (rotated_img.width / 2, rotated_img.height / 2)
    
    left = rotated_center[0] - img.width / 2
    top = rotated_center[1] - img.height / 2
    right = rotated_center[0] + img.width / 2
    bottom = rotated_center[1] + img.height / 2
    
    cropped_img = rotated_img.crop((left, top, right, bottom))
    
    return cropped_img
