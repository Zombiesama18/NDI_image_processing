import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageFilter
import random
import copy


class MoCo(nn.Module):
    def __init__(self, base_encoder, dim=512, K=1024, m=0.999, T=0.07, mlp=True, model_type='resnet', pretrained=False,
                 three_channel=False, custom_model=False):
        super(MoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T
        if custom_model:
            self.encoder_q = base_encoder
            self.encoder_k = copy.deepcopy(base_encoder)
            test_input = torch.rand((1, 3, 200, 200))
            assert self.encoder_q(test_input).shape[-1] == dim, \
                "Last dimension of custom model must be equal to param dim"
        else:
            if pretrained:
                self.encoder_q = base_encoder(pretrained=pretrained)
                self.encoder_k = base_encoder(pretrained=pretrained)
                if model_type == 'resnet':
                    self.encoder_q.fc = nn.Linear(self.encoder_q.fc.in_features, dim)
                    self.encoder_k.fc = nn.Linear(self.encoder_k.fc.in_features, dim)
            else:
                self.encoder_q = base_encoder(num_classes=dim, pretrained=pretrained)
                self.encoder_k = base_encoder(num_classes=dim, pretrained=pretrained)
            if mlp:
                if model_type == 'resnet':
                    dim_mlp = self.encoder_q.fc.weight.shape[1]
                    self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
                    self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)
            if not three_channel:
                if model_type == 'resnet':
                    self.encoder_q.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                    self.encoder_k.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                elif model_type == 'alexnet':
                    self.encoder_q.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2), bias=False)
                    self.encoder_k.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2), bias=False)
                else:
                    raise TypeError('Unknown model type')
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        self.register_buffer('queue', torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k, evaluate=False):
        q = self.encoder_q(im_q)
        q = F.normalize(q, dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)
            k = F.normalize(k, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        if not evaluate:
            self._dequeue_and_enqueue(k)
        return logits, labels


class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


