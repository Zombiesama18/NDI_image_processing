import torch
from pathlib import Path
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset


def k_fold_split(original_path, target_path, k=5):
    original_images = list(sorted(list(map(str, list(Path(original_path).glob('*.jpg'))))))
    target_images = list(sorted(list(map(str, list(Path(target_path).glob('*.jpg'))))))
    images = list(zip(original_images, target_images))
    for i in range(k):
        train = images[:i * len(images) // k] + images[(i + 1) * len(images) // k:]
        val = images[i * len(images) // k: (i + 1) * len(images) // k]
        yield train, val


def split_train_validation_randomly(original_path, target_path):
    original_images = list(sorted(list(map(str, list(Path(original_path).glob('*.jpg'))))))
    target_images = list(sorted(list(map(str, list(Path(target_path).glob('*.jpg'))))))
    images = list(zip(original_images, target_images))
    train_images, val_images = torch.utils.data.random_split(images, [160, 24])
    return train_images, val_images


class ThreeChannelNDIDatasetContrastiveLearningWithAug(Dataset):
    def __init__(self, images, evaluate=False):
        super(ThreeChannelNDIDatasetContrastiveLearningWithAug, self).__init__()
        if not evaluate:
            self.images = images[0]
        else:
            self.images = images[1]
        # 水平翻转 + 随机旋转 训练慢（100）但是效果好
        self.transforms = transforms.Compose([
            # transforms.GaussianBlur(kernel_size=3, sigma=0.7),
            # transforms.CenterCrop(150),
            # transforms.Resize(200),
            # transforms.RandomHorizontalFlip(0.5),
            # transforms.RandomRotation(30)
            ])
        self.evaluate = evaluate

    def __getitem__(self, idx):
        origin_path, target_path = self.images[idx]
        origin = Image.open(origin_path).convert('RGB')
        target = Image.open(target_path)
        if not self.evaluate:
            origin, target = self.transforms(torch.cat((transforms.ToTensor()(origin).unsqueeze(0), transforms.ToTensor()(target).unsqueeze(0)), dim=0))
        else:
            origin, target = transforms.ToTensor()(origin), transforms.ToTensor()(target)
        label = int(origin_path.split('/')[-1].split('.')[0]) - 1
        return origin, target, label

    def __len__(self):
        return len(self.images)
