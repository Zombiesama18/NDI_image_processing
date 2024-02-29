import copy
from typing import Optional, Union
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


class SiameseModel(nn.Module):
    def __init__(self, base_model: nn.Module, split_from: int):
        super().__init__()
        self.base_model = nn.ModuleDict()
        base_model = list(base_model.named_children())
        while base_model[0][0] != f'layer{split_from}':
            self.base_model.add_module(*base_model.pop(0))
        self.branch_model_obsv = nn.ModuleDict(
            [(name, module) for name, module in base_model])
        self.branch_model_calc = copy.deepcopy(self.branch_model_obsv)
        self.logit_scale = nn.Parameter(torch.ones([]) * (1 / 0.07))
    
    def forward(self, obsv, calc, extract_feature_layer: Union[str, list]=[]):
        if isinstance(extract_feature_layer, str):
            extract_feature_layer = [extract_feature_layer]
        extract_feature_layer = set(extract_feature_layer)
        
        obsv_output, calc_output = [], []
        
        for name, module in self.base_model.named_children():
            obsv = module(obsv)
            calc = module(calc)
            if name in extract_feature_layer:
                obsv_output.append(obsv)
                calc_output.append(calc)
            
        for name, module in self.branch_model_obsv.named_children():
            obsv = module(obsv)
            if name == 'avgpool':
                obsv = obsv.flatten(start_dim=1)
            if name in extract_feature_layer:
                obsv_output.append(obsv)
            
        
        for name, module in self.branch_model_calc.named_children():
            calc = module(calc)
            if name == 'avgpool':
                calc = calc.flatten(start_dim=1)
            if name in extract_feature_layer:
                calc_output.append(calc)
        
        return (obsv, calc), (obsv_output, calc_output)
    
    def get_available_extract_layer(self):
        return [name for name, _ in self.base_model.named_children()] + \
            [name for name, _ in self.branch_model_obsv.named_children()]
    
    def get_similarity_matrix(self, em_q, em_k):
        em_q = F.normalize(em_q, dim=1)
        em_k = F.normalize(em_k, dim=1)
        return self.logit_scale * torch.matmul(em_q, em_k.t())
    
    
class CalculatedModel(nn.Module):
    def __init__(self, base_model: nn.Module, num_classes: int):
        super().__init__()
        self.base_model = base_model
        
        base_last_layer = list(self.base_model.children())[-1]
        
        assert isinstance(base_last_layer, nn.Linear)
        
        input_dim = base_last_layer.out_features
        
        self.classification_head = nn.Linear(input_dim, num_classes)
        self.angle_head = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.base_model(x)
        return self.classification_head(x), self.angle_head(x)
    
    def compute_loss(self, 
                     criterions: Union[tuple, nn.Module],
                     logits: Union[tuple, torch.Tensor], 
                     labels: Union[tuple, torch.Tensor], 
                     weights: Union[tuple, float] = None):
        
        if isinstance(logits, torch.Tensor):
            logits = tuple(logits)
        if isinstance(labels, torch.Tensor):
            labels = tuple(labels)
        
        assert len(logits) == len(labels)
        if weights is None:
            weights = [1.0] * len(logits)
        elif isinstance(weights, float):
            weights = [weights] * len(logits)
        else:
            assert len(weights) == len(logits)
            
        if isinstance(criterions, nn.Module):
            criterions = tuple([criterions] * len(logits))
        
        loss_list = []
        total_loss = 0
        
        for criterion, logit, label, weight in zip(criterions, logits, labels, weights):
            loss = criterion(logit, label)
            loss_list.append(loss)
            total_loss += weight * loss
        
        return total_loss, *loss_list
        
        
        
        
        

