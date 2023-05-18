import copy
from torch import nn
import torch
from torch.nn import functional as F
import numpy as np


class RetrievalModel(nn.Module):
    def __init__(self, encoder, T=0.07) -> None:
        super().__init__()
        self.encoder_q = encoder
        self.encoder_k = copy.deepcopy(encoder)
        self.logit_scale = nn.Parameter(torch.ones([]) * (1 / T))

    def forward(self, im_q, im_k, **kwargs):
        q = self.encoder_q(im_q, **kwargs)
        k = self.encoder_k(im_k, **kwargs)
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
