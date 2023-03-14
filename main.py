from dataset import *
from train import *
from utils import torch_fix_seed

torch_fix_seed(19981303)

def get_self_pretrain_model(index=1000):
    base_encoder = torchvision.models.resnet50(weights=None)
    base_encoder.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    origin_dim_mlp = base_encoder.fc.in_features
    base_encoder.fc = None
    temp = torch.load(f'./checkpoints/CEM_ALL_CHECK_{index}_Epoch.pth')['state_dict']
    state_dict = {}
    for k, v in temp.items():
        if 'encoder_q' in k:
            if 'fc' not in k:
                state_dict['.'.join(k.split('.')[1:])] = v
    base_encoder.load_state_dict(state_dict)
    base_encoder.fc = torch.nn.Linear(origin_dim_mlp, 512)
    return base_encoder

def get_CEM_pretrain_model():
    base_encoder = torchvision.models.resnet50(weights=None)
    base_encoder.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    origin_dim_mlp = base_encoder.fc.in_features
    base_encoder.fc = None
    base_encoder.load_state_dict(torch.load('../../GithubProject/cem-dataset/cem1.5m_swav_resnet50_200ep_balanced.pth.tar')['state_dict'])
    base_encoder.fc = torch.nn.Linear(origin_dim_mlp, 512)
    return base_encoder

def get_model(keyword):
    if keyword == 'self_pretrained':
        return get_self_pretrain_model(index=900)
    if keyword == 'CEM':
        return get_CEM_pretrain_model()
    if keyword == 'ImageNet':
        model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        origin_dim_mlp = model.fc.in_features
        model.fc = torch.nn.Linear(origin_dim_mlp, 512)
        return model
    if keyword == 'None':
        model = torchvision.models.resnet50(weights=None)
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        origin_dim_mlp = model.fc.in_features
        model.fc = torch.nn.Linear(origin_dim_mlp, 512)
        return model

def main():
    top_k_candidates = (10, 20, 30)
    k = 7
    temps = 0.7
    momentums = 0.99
    
    parameters = {'pretrain_model': ['self_pretrained', 'CEM', 'ImageNet', 'None']}
    
    train_metrics = HistoryRecorder(['Train Loss', 'Train Acc', 'Val Loss', 'Val Acc'], list(parameters.keys()))

    parameters = list(itertools.product(*parameters.values()))

    for parameter in parameters:

        ### custom part to get parameters
        pretrain_model = parameter[0]
        ### END
        
        for images in k_fold_train_validation_split(ORIGINAL_IMAGE, TARGET_IMAGE, k):
            train_dataset = SingleChannelNDIDatasetContrastiveLearningWithAug(images, False)
            val_dataset = SingleChannelNDIDatasetContrastiveLearningWithAug(images, True)
            train_iter = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
            val_iter = DataLoader(val_dataset, batch_size=len(val_dataset))

            model = get_model(pretrain_model)
            model = RetrievalModel(model)
            model = model.cuda()
            
            if pretrain_model in ['self_pretrained', 'CEM', 'ImageNet']:
                lr = 5e-3
            else:
                lr = 2e-2
                
            device = torch.device('cuda:0')
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
            start_time = time.time()
            metrics = train_moco_return_metrics_top_k(model, train_iter, val_iter, optimizer, 30, device,
                                                        tested_parameter=parameter, k_candidates=top_k_candidates, scheduler=scheduler)
            end_time = time.time()
            train_metrics.cal_add(metrics)
    train_metrics.cal_divide(k)
    
    draw_graph(train_metrics.data, 30, ('tok_ks'))
    for k, v in train_metrics.data.items():
        for k1 in v[1].keys():
            print(f'{k} {k1} mean {np.mean(v[3][k1][-10:])}')
            print(f'{k} {k1} max {np.max(v[3][k1])}')

if __name__ == '__main__':
    main()
