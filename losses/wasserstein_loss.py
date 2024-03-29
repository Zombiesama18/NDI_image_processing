from typing import Any
import numpy as np
import ot
import torch
from models.wd_model import TransformNet

def rand_projections(dim, num_projections=1000):
    projections = torch.randn((num_projections, dim))
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections

def cost_matrix(encoded_smaples, distribution_samples, p=2):
    n = encoded_smaples.size(0)
    m = distribution_samples.size(0)
    d = encoded_smaples.size(1)
    x = encoded_smaples.unsqueeze(1).expand(n, m, d)
    y = distribution_samples.unsqueeze(0).expand(n, m, d)
    C = torch.pow(torch.abs(x - y), p).sum(2)
    return C

def phi_d(s, d):
    return torch.log((1 + 4 * s / (2 * d - 3))) * (-1.0 / 2)

def cost_matrix_slow(x, y):
    """
    Input: x is a Nxd matrix
        y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)

def circular_function(x1, x2, theta, r, p):
    cost_matrix_1 = torch.sqrt(cost_matrix_slow(x1, theta * r))
    cost_matrix_2 = torch.sqrt(cost_matrix_slow(x2, theta * r))
    return get_wasserstein_distance_final_step(cost_matrix_1, cost_matrix_2, p)


def get_wasserstein_distance_final_step(first_projections, second_projections, p=2):
    wasserstein_distance = torch.abs(
        (torch.sort(first_projections.transpose(0, 1), dim=1)[0] -
        torch.sort(second_projections.transpose(0, 1), dim=1)[0])
    )
    wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=1), 1.0 / p)
    return torch.pow(torch.pow(wasserstein_distance, p).mean(), 1.0 / p)


def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mean(torch.abs(torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)))


def cosine_sum_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mean(torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps))


def compute_true_Wasserstein(X, Y, p=2):
    M = ot.dist(X.detach().numpy(), Y.detach().numpy())
    a = np.ones((X.shape[0],)) / X.shape[0]
    b = np.ones((Y.shape[0],)) / Y.shape[0]
    return ot.emd2(a, b, M)


def compute_Wasserstein(x, y, device, p=2):
    M = cost_matrix(x, y, p)
    pi = ot.emd([], [], M.cpu().detach().numpy())
    pi = torch.from_numpy(pi).to(device)
    return torch.sum(pi * M)


def SlicedWassersteinDistance(first_samples, 
                              second_samples, 
                              num_projections=1000, 
                              p=2, 
                              device="cuda"):
    dim = second_samples.size(1)
    projections = rand_projections(dim, num_projections).to(device)
    first_projections = first_samples.matmul(projections.transpose(0, 1))
    second_projections = second_samples.matmul(projections.transpose(0, 1))
    return get_wasserstein_distance_final_step(first_projections, second_projections, p)


def GeneralizedSlicedWassersteinLoss(first_samples, 
                                     second_samples, 
                                     g_function, 
                                     r=1, 
                                     num_projections=1000, 
                                     p=2, 
                                     device="cuda"):
        embedding_dim = first_samples.size(1)
        projections = rand_projections(embedding_dim, num_projections).to(device)
        return g_function(first_samples, second_samples, projections, r, p)
        

def MaxSlicedWassersteinDistance(first_samples, 
                                 second_samples, 
                                 p=2, 
                                 max_iter=100, 
                                 device="cuda"):
    theta = torch.randn((1, first_samples.shape[1]), device=device, requires_grad=True)
    theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1))
    opt = torch.optim.Adam([theta], lr=1e-4)
    frozen_first_samples = first_samples.detach().clone()
    frozen_second_samples = second_samples.detach().clone()
    for _ in range(max_iter):
        encoded_projections = torch.matmul(frozen_first_samples, theta.transpose(0, 1))
        distribution_projections = torch.matmul(frozen_second_samples, theta.transpose(0, 1))
        wasserstein_distance = torch.abs(
            (torch.sort(encoded_projections)[0] - torch.sort(distribution_projections)[0])
        )
        wasserstein_distance = torch.mean(torch.pow(wasserstein_distance, p))
        l = -wasserstein_distance
        opt.zero_grad()
        l.backward(retain_graph=True)
        opt.step()
        theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1))

    theta.requires_grad = False
    theta = theta.detach()
    encoded_projections = torch.matmul(first_samples, theta.transpose(0, 1))
    distribution_projections = torch.matmul(second_samples, theta.transpose(0, 1))
    wasserstein_distance = torch.abs(
            (torch.sort(encoded_projections)[0] - torch.sort(distribution_projections)[0])
        )
    wasserstein_distance = torch.mean(torch.pow(wasserstein_distance, p))
    
    return wasserstein_distance, theta


def MaxGeneralizedSlicedWassersteinDistance(first_samples, 
                                            second_samples, 
                                            g_function, 
                                            r, 
                                            p=2, 
                                            max_iter=100, 
                                            device="cuda"):
        theta = torch.randn((1, first_samples.shape[1]), device=device, requires_grad=True)
        theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1))
        opt = torch.optim.Adam([theta], lr=1e-4)
        for _ in range(max_iter):
            wasserstein_distance = g_function(first_samples, second_samples, theta, r, p)
            l = -wasserstein_distance
            opt.zero_grad()
            l.backward(retain_graph=True)
            opt.step()
            theta.data = theta.data / torch.sqrt(torch.sum(theta.data ** 2, dim=1))
        wasserstein_distance = g_function(first_samples, second_samples, theta, r, p)
        return wasserstein_distance


def DistributionalGeneralizedSlicedWassersteinDistance(first_samples, 
                                                       second_samples, 
                                                       num_projections, 
                                                       f, 
                                                       f_op, 
                                                       g_function, 
                                                       r, 
                                                       p=2, 
                                                       max_iter=10, 
                                                       lam=1, 
                                                       device="cuda"):

    embedding_dim = first_samples.size(1)
    pro = rand_projections(embedding_dim, num_projections).to(device)
    for _ in range(max_iter):
        projections = f(pro)
        reg = lam * cosine_distance_torch(projections, projections)
        wasserstein_distance = g_function(first_samples, second_samples, projections, r, p)
        loss = reg - wasserstein_distance
        f_op.zero_grad()
        loss.backward(retain_graph=True)
        f_op.step()
    projections = f(pro)
    wasserstein_distance = g_function(first_samples, second_samples, projections, r, p)
    return wasserstein_distance


def DistributionalSlicedWassersteinDistance(first_samples, 
                                            second_samples, 
                                            num_projections, 
                                            f, 
                                            f_op, 
                                            p=2, 
                                            max_iter=10, 
                                            lam=1, 
                                            device="cuda"):
    embedding_dim = first_samples.size(1)
    pro = rand_projections(embedding_dim, num_projections).to(device)
    first_samples_detach = first_samples.detach()
    second_samples_detach = second_samples.detach()
    for _ in range(max_iter):
        projections = f(pro)
        cos = cosine_distance_torch(projections, projections)
        reg = lam * cos
        encoded_projections = first_samples_detach.matmul(projections.transpose(0, 1))
        distribution_projections = second_samples_detach.matmul(projections.transpose(0, 1))
        wasserstein_distance = get_wasserstein_distance_final_step(encoded_projections, distribution_projections, p)
        loss = reg - wasserstein_distance
        f_op.zero_grad()
        loss.backward(retain_graph=True)
        f_op.step()

    projections = f(pro)
    encoded_projections = first_samples.matmul(projections.transpose(0, 1))
    distribution_projections = second_samples.matmul(projections.transpose(0, 1))
    wasserstein_distance = get_wasserstein_distance_final_step(encoded_projections, distribution_projections, p)
    return wasserstein_distance


def get_wd_loss(wd_args, first_samples, second_samples) -> Any:
    loss = 0
    if wd_args.wd_type == 'DSWD':
        loss = DistributionalSlicedWassersteinDistance(
            first_samples, 
            second_samples, 
            wd_args.num_projections, 
            wd_args.tran_net,
            wd_args.op_trannet, 
            wd_args.wd_p, 
            wd_args.wd_max_iter, 
            wd_args.wd_lam, 
            first_samples.device
        )
    elif wd_args.wd_type == 'DGSWD':
        loss = DistributionalGeneralizedSlicedWassersteinDistance(
            first_samples, 
            second_samples, 
            wd_args.num_projections, 
            wd_args.tran_net,
            wd_args.op_trannet, 
            wd_args.g_func, 
            wd_args.wd_r, 
            wd_args.wd_p, 
            wd_args.wd_max_iter,
            wd_args.wd_lam, 
            first_samples.device
        )
    elif wd_args.wd_type == 'MSWD':
        loss, _ = MaxSlicedWassersteinDistance(
            first_samples, second_samples, 
            wd_args.wd_p, 
            wd_args.wd_max_iter,
            first_samples.device
            )
    elif wd_args.wd_type == 'MGSWD':
        loss = MaxGeneralizedSlicedWassersteinDistance(
            first_samples, 
            second_samples, 
            wd_args.g_func, 
            wd_args.wd_r, 
            wd_args.wd_p,
            wd_args.wd_max_iter,
            first_samples.device
        )
    elif wd_args.wd_type == 'SWD':
        loss = SlicedWassersteinDistance(
            first_samples, 
            second_samples, 
            wd_args.num_projections, 
            wd_args.wd_p,
            first_samples.device
            )
    elif wd_args.wd_type == 'GSWD':
        loss = GeneralizedSlicedWassersteinLoss(
            first_samples, 
            second_samples, 
            wd_args.g_func, 
            wd_args.wd_r,
            wd_args.num_projections, 
            wd_args.wd_p,
            first_samples.device
        )
    return loss


def get_wd_configuration(wd_args):
    channels = [512, 2048, 1024, 512, 256, 64]  # Flexible for different models
    if wd_args.wd_type == 'DSWD':
        assert wd_args.num_projections is not None
        assert wd_args.wd_p is not None
        assert wd_args.wd_max_iter is not None
        assert wd_args.wd_lam is not None
        assert wd_args.wd_stage is not None

        transform_net = TransformNet(channels[wd_args.wd_stage - 1]).to('cuda')
        optimizer_tran_net = torch.optim.Adam(transform_net.parameters(), lr=0.0005, betas=(0.5, 0.999))
        wd_args.transform_net = transform_net
        wd_args.op_trannet = optimizer_tran_net

    elif wd_args.wd_type == 'DGSWD':
        assert wd_args.num_projections is not None
        assert wd_args.wd_r is not None
        assert wd_args.wd_p is not None
        assert wd_args.wd_max_iter is not None
        assert wd_args.wd_lam is not None
        
        wd_args.g_func = circular_function

        transform_net = TransformNet(channels[wd_args.wd_stage - 1]).to('cuda')
        optimizer_tran_net = torch.optim.Adam(transform_net.parameters(), lr=0.0005, betas=(0.5, 0.999))
        wd_args.transform_net = transform_net
        wd_args.op_trannet = optimizer_tran_net

    elif wd_args.wd_type == 'MSWD':
        assert wd_args.wd_p is not None
        assert wd_args.wd_max_iter is not None

    elif wd_args.wd_type == 'MGSWD':
        assert wd_args.wd_r is not None
        assert wd_args.wd_p is not None
        assert wd_args.wd_max_iter is not None
        
        wd_args.g_func = circular_function

    elif wd_args.wd_type == 'SWD':
        assert wd_args.num_projections is not None
        assert wd_args.wd_p is not None

    elif wd_args.wd_type == 'GSWD':
        assert wd_args.num_projections is not None
        assert wd_args.wd_r is not None
        assert wd_args.wd_p is not None
        
        wd_args.g_func = circular_function




