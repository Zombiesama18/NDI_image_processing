import loralib as lora
from torch import nn
import torchvision
import torch
import torch.nn.functional as F
from train_on_NDI import *
from config import Config


cfg = Config()
device = torch.device(cfg.device)
set_all_seeds(3401)


def convert_layers_LoRA(model, convert_types=[nn.Linear, nn.Conv2d], rank_factor=10):
    named_modules = dict(model.named_modules())
    for name, module in find_modules(model=model, search_classes=convert_types):
        name_parts = name.split('.')
        if len(name_parts) > 1:
            parent_module = '.'.join(name_parts[:-1])
            parent = named_modules[parent_module]
            name = name.split('.')[-1]
        else:
            parent = model
        update_to_LoRA_layer(module, parent, name, rank_factor)
    return model


def update_to_LoRA_layer(module, parent, name, rank_factor=10):
    if isinstance(module, nn.Linear):
        setattr(parent, name, lora.Linear(module.in_features, module.out_features,
                                          r=min(module.in_features, module.out_features) // rank_factor))
                                          #  r=4))
    elif isinstance(module, nn.Conv2d):
        if type(module.kernel_size) is not int:
            kernel_size = module.kernel_size[0]
            assert all([dim == kernel_size for dim in module.kernel_size])
        else:
            kernel_size = module.kernel_size
        setattr(parent, name, lora.Conv2d(module.in_channels, module.out_channels, kernel_size,
                                          r=max(16, min(module.in_channels, module.out_channels) // rank_factor),
                                          # r=4,
                                          stride=module.stride, padding=module.padding, 
                                          dilation=module.dilation, groups=module.groups))


def find_modules(model, search_classes=[nn.Linear, nn.Conv2d], exclusion=[lora.LoRALayer]):
    layers_to_replace = []
    for name, module in model.named_modules():
        for _class in exclusion:
            if isinstance(module, _class):
                continue
        for _class in search_classes:
            if isinstance(module, _class):
                layers_to_replace.append((name, module))
    return layers_to_replace


def load_checkpoints(base_encoder, ckpt_path):
    temp = torch.load(ckpt_path)['state_dict']
    state_dict = {}
    for k, v in temp.items():
        if 'encoder_q' in k:
            if 'fc' not in k:
                state_dict['.'.join(k.split('.')[1:])] = v
    base_encoder.load_state_dict(state_dict, strict=False)
    return base_encoder


def get_modified_resnet50():
    model = torchvision.models.resnet50()
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    origin_mlp_dim = model.fc.in_features
    model.fc = nn.Linear(origin_mlp_dim, 512)
    return model


def main():
    logger_fname = datetime.datetime.now().strftime('%Y%m%d%H%M')
    logger = get_logger(f'./logs/{logger_fname}.log')

    logger.info(f'This training is to do {cfg.message_to_log}')

    for i, images in enumerate(k_fold_train_validation_split(ORIGINAL_IMAGE, TARGET_IMAGE, 7)):
        wandb.init(project=cfg.message_to_log, group='lora_implementation', job_type='Imagenet_NDI_400E_sgd_lr_5e-3',
                    name=f'fold {i}', config=cfg.__dict__)
        train_dataset = SingleChannelNDIDatasetContrastiveLearningWithAug(images, False, 200)
        val_dataset = SingleChannelNDIDatasetContrastiveLearningWithAug(images, True, 200)
        train_iter = DataLoader(train_dataset, cfg.batch_size, shuffle=True, drop_last=True)
        val_iter = DataLoader(val_dataset, batch_size=len(val_dataset))

        model = get_modified_resnet50()
        model = convert_layers_LoRA(model)
        print(type(model.layer1[0].conv1))
        model = load_checkpoints(model, './checkpoints/ImageNet_ALL_CHECK_400_Epoch.pth')
        lora.mark_only_lora_as_trainable(model)

        model = RetrievalModel(model)
        model = model.cuda()

        training_params = [param for param in model.parameters() if param.requires_grad]
        optimizer = torch.optim.SGD(params=training_params, lr=cfg.base_lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
        # optimizer = torch.optim.Adam(params=training_params, lr=cfg.base_lr, weight_decay=cfg.weight_decay)
        criterion = nn.CrossEntropyLoss()

        # scheduler = None
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=cfg.epochs, eta_min=cfg.base_lr * 0.01)

        train(cfg, logger, train_iter, val_iter, model, criterion, optimizer, cfg.epochs, scheduler=scheduler,
                save_folder=cfg.output_dir, wandb_config=True, device=device)
        wandb.finish()


if __name__ == '__main__':
    main()





