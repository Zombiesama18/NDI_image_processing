from torch import nn as nn
import torch


class TransformNet(nn.Module):
    def __init__(self, size):
        super(TransformNet, self).__init__()
        self.size = size
        self.net = nn.Sequential(nn.Linear(self.size, self.size))

    def forward(self, inputs):
        out = self.net(inputs)
        return out / torch.sqrt(torch.sum(out ** 2, dim=1, keepdim=True))


class CompositeResNet(nn.Module):
    def __init__(self, existing_model, max_stage: int):
        super().__init__()
        self.model = existing_model
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, stage: int = 1):
        outputs = []
        x1 = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))
        outputs.append(x1)
        x2 = self.model.layer1(x1)
        outputs.append(x2)
        x3 = self.model.layer2(x2)
        outputs.append(x3)
        x4 = self.model.layer3(x3)
        outputs.append(x4)
        x5 = self.model.layer4(x4)
        outputs.append(x5)
        x6 = self.model.fc(self.model.avgpool(x5).flatten(1))
        outputs.append(x6)
        outputs = list(reversed(outputs))
        result = []
        for i in range(stage):
            if i == 0:
                result.append(outputs[0])
            else:
                result.append(self.avgpool(outputs[i]))
        return result


