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
    def __init__(self, existing_model, max_stage: int = 1):
        super().__init__()
        self.model = existing_model
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, stage: int = 1, cel_stage=None, wd_stage=None):
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
        # result = []
        # for i in range(stage):
        #     if i == 0:
        #         result.append(outputs[0])
        #     else:
        #         result.append(self.avgpool(outputs[i]))
        result = [outputs[0]]
        if cel_stage:
            result.append(self.avgpool(outputs[cel_stage - 1]))
        if wd_stage:
            result.append(self.avgpool(outputs[wd_stage - 1]))
        if not cel_stage and not wd_stage and stage != 1:
            result.append(self.avgpool(outputs[stage - 1]))
        return result


class CompositeViT(nn.Module):
    def __init__(self, existing_model, max_stage: int = 1):
        super().__init__()
        self.model = existing_model
        self.proj = nn.Linear(37632, 512)

    def forward(self, x, **kwargs):
        result = []
        batch_size = x.shape[0]
        x = self.model.patch_embed(x)
        x = x + self.model.pos_embed
        for i, block in enumerate(self.model.blocks):
            x = block(x)
        x = self.model.norm(x.float())
        intermediate_output = self.proj(x.reshape(batch_size, -1))
        result.append(intermediate_output)
        result.append(intermediate_output)
        x = x.reshape(batch_size, -1)
        x = self.model.feature(x)
        result.insert(0, x)
        return result
