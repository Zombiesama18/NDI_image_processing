import torch
import torch.nn as nn
from torch.nn import functional as F


def train(train_iter, net, criterion, optimizer, epochs, device):
    net.to(device)
    for epoch in range(epochs):
        net.train()
        total_loss = 0
        for images, labels in train_iter:
            images[0], images[1] = images[0].to(device), images[1].to(device)
            labels = labels.to(device)
            output, target = net(images[0], images[1])
            loss = criterion(output, target)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return net

