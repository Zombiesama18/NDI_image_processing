import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageFilter
import random
import copy
import deprecated
import numpy as np


class MoCo(nn.Module):
    
    def __init__(self, base_encoder, dim=512, K=1024, m=0.999, T=0.07, mlp=True, weights=None, 
                 customized_model=None, k_fold=False):
        super(MoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T
        self.k_fold = k_fold
        if customized_model:
            self.encoder_q = base_encoder
            self.encoder_k = copy.deepcopy(base_encoder)
            test_input = torch.rand((1, 1, 200, 200))
            assert self.encoder_q(test_input).shape[-1] == dim, \
                "Last dimension of custom model must be equal to param dim"
        else:
            self.encoder_q = base_encoder(weights=weights)
            self.encoder_k = base_encoder(weights=weights)
            self.encoder_q.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.encoder_k.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Linear(dim_mlp, dim)
            self.encoder_k.fc = nn.Linear(dim_mlp, dim)
        if mlp:
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        self.register_buffer('queue', torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
    
    
    # @deprecated(version='before 20230130', reason='After several experiments, we find the optimal implementation so far. So' \
    #             'we rewrite the initialization function')
    # def __init__(self, base_encoder, dim=512, K=1024, m=0.999, T=0.07, mlp=True, model_type='resnet', weights=None,
    #              three_channel=False, custom_model=False, k_fold=False):
    #     super(MoCo, self).__init__()
    #     self.K = K
    #     self.m = m
    #     self.T = T
    #     self.k_fold = k_fold
    #     if custom_model:
    #         self.encoder_q = base_encoder
    #         self.encoder_k = copy.deepcopy(base_encoder)
    #         test_input = torch.rand((1, 3, 200, 200))
    #         assert self.encoder_q(test_input).shape[-1] == dim, \
    #             "Last dimension of custom model must be equal to param dim"
    #     else:
    #         if weights:
    #             self.encoder_q = base_encoder(weights=weights)
    #             self.encoder_k = base_encoder(weights=weights)
    #             if model_type == 'resnet':
    #                 self.encoder_q.fc = nn.Linear(self.encoder_q.fc.in_features, dim)
    #                 self.encoder_k.fc = nn.Linear(self.encoder_k.fc.in_features, dim)
    #         else:
    #             self.encoder_q = base_encoder(num_classes=dim, weights=weights)
    #             self.encoder_k = base_encoder(num_classes=dim, weights=weights)
    #         if mlp:
    #             if model_type == 'resnet':
    #                 dim_mlp = self.encoder_q.fc.weight.shape[1]
    #                 self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
    #                 self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)
    #         if not three_channel:
    #             if model_type == 'resnet':
    #                 self.encoder_q.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    #                 self.encoder_k.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    #             elif model_type == 'alexnet':
    #                 self.encoder_q.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2), bias=False)
    #                 self.encoder_k.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2), bias=False)
    #             else:
    #                 raise TypeError('Unknown model type')
    #     for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
    #         param_k.data.copy_(param_q.data)
    #         param_k.requires_grad = False
    #     self.register_buffer('queue', torch.randn(dim, K))
    #     self.queue = F.normalize(self.queue, dim=0)
    #     self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if not self.k_fold:
            assert self.K % batch_size == 0
            self.queue[:, ptr: ptr + batch_size] = keys.T
        else:
            if ptr + batch_size >= self.K:
                self.queue[:, ptr:] = keys.T[:ptr]
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



class RetrievalModel(nn.Module):
    def __init__(self, encoder, T=0.07) -> None:
        super().__init__()
        self.encoder_q = encoder
        self.encoder_k = copy.deepcopy(encoder)
        self.logit_scale = nn.Parameter(torch.ones([]) * (1 / T))
    
    def forward(self, im_q, im_k, **kwargs):
        q = self.encoder_q(im_q)
        k = self.encoder_k(im_k)
        return q, k
    
    def get_similarity_matrix(self, em_q, em_k):
        em_q = F.normalize(em_q, dim=1)
        em_k = F.normalize(em_k, dim=1)
        return self.logit_scale * torch.matmul(em_q, em_k.t())
    
    @staticmethod
    def compute_loss(sim_matrix, direction='b', weights=(0.5, 0.5)):
        """
        direction means to calculate loss bidirectionally or unidirectionally

        Args:
            sim_matrix (_type_): _description_
            direction (int, optional): _description_. Defaults to 2.
        """
        logpt1 = F.log_softmax(sim_matrix, dim=-1)
        logpt1 = torch.diag(logpt1)
        loss1 = -logpt1.mean()
        if direction == 'u':
            return loss1
        logpt2 = F.log_softmax(sim_matrix.T, dim=-1)
        logpt2 = torch.diag(logpt2)
        loss2 = -logpt2.mean()
        weights = np.array(weights)
        weights = weights / weights.sum()
        return weights[0] * loss1 + weights[1] * loss2
        
