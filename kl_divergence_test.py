from train_on_NDI import *
import itertools
from models.wd_model import TransformNet, CompositeResNet, CompositeViT
from wasserstein_loss import distributional_sliced_wasserstein_distance, \
    distributional_generalized_sliced_wasserstein_distance, max_sliced_wasserstein_distance, \
    max_generalized_sliced_wasserstein_distance, sliced_wasserstein_distance, generalized_sliced_wasserstein_distance, \
    circular_function


def kl_divergence(p, q):
    p = f.softmax(p, dim=-1)
    q = f.softmax(q, dim=-1)
    p = p.unsqueeze(1)
    q = q.unsqueeze(0)
    kl_matrix = (p * (p / q).log()).sum(-1)
    return kl_matrix
    # return torch.exp(-kl_matrix)

def get_wd_loss(first_samples, second_samples, wd_utils, index):
    loss = 0
    if wd_utils['type'] == 'DSWD':
        loss = distributional_sliced_wasserstein_distance(
            first_samples, second_samples, wd_utils['num_projections'], wd_utils[f'tran_net'],
            wd_utils[f'op_trannet'], wd_utils['p'], wd_utils['max_iter'], wd_utils['lam'], first_samples.device
        )
    elif wd_utils['type'] == 'DGSWD':
        loss = distributional_generalized_sliced_wasserstein_distance(
            first_samples, second_samples, wd_utils['num_projections'], wd_utils[f'tran_net'],
            wd_utils[f'op_trannet'], wd_utils['g_func'], wd_utils['r'], wd_utils['p'], wd_utils['max_iter'],
            wd_utils['lam'], first_samples.device
        )
    elif wd_utils['type'] == 'MSWD':
        loss, _ = max_sliced_wasserstein_distance(first_samples, second_samples, wd_utils['p'], wd_utils['max_iter'],
                                                  first_samples.device)
    elif wd_utils['type'] == 'MGSWD':
        loss = max_generalized_sliced_wasserstein_distance(
            first_samples, second_samples, wd_utils['g_func'], wd_utils['r'], wd_utils['p'], wd_utils['max_iter'],
            first_samples.device
        )
    elif wd_utils['type'] == 'SWD':
        loss = sliced_wasserstein_distance(first_samples, second_samples, wd_utils['num_projections'], wd_utils['p'],
                                           first_samples.device)
    elif wd_utils['type'] == 'GSWD':
        loss = generalized_sliced_wasserstein_distance(
            first_samples, second_samples, wd_utils['g_func'], wd_utils['r'], wd_utils['num_projections'], wd_utils['p'],
            first_samples.device
        )
    return loss

def train_epoch(train_data, val_data, model, criterion, optimizer, current_epoch, total_epoch, target_tensor,
                cel_utils=None, wd_utils=None, adp_weights=None):
    batch_time = AverageMeter('Batch Time', ':6.3f')
    data_time = AverageMeter('Data Time', ':6.3f')
    train_loss = AverageMeter('Train Loss', ':.4e')
    val_loss = AverageMeter('Val Loss', ':.4e')
    val_acc_10 = AverageMeter('Val Acc@10', ':6.2f')
    val_acc_20 = AverageMeter('Val Acc@20', ':6.2f')
    val_acc_30 = AverageMeter('Val Acc@30', ':6.2f')

    train_progress = ProgressMeter(len(train_data), [batch_time, data_time, train_loss],
                                   prefix=f'Training Progress\tEpoch: [{current_epoch}/{total_epoch}]')
    val_progress = ProgressMeter(len(val_data), [val_loss, val_acc_10, val_acc_20, val_acc_30],
                                 prefix=f'Validation Progress\tEpoch: [{current_epoch}/{total_epoch}]')

    model.train()
    start = time.time()

    for i, (origin, target, label) in enumerate(train_data):
        data_time.update(time.time() - start)
        origin, target, label = origin.cuda(), target.cuda(), label.cuda()
        feature_ori, feature_tar = model(origin, target, wd_stage=wd_utils['stage'], cel_stage=cel_utils['stage'])
        loss = 0
        if adp_weights:
            # loss_weights = nn.functional.softplus(adp_weights['data'])
            loss_weights = adp_weights['data'] / adp_weights['data'].sum()
            # loss_weights = loss_weights / loss_weights.sum()
            for j, (em_ori, em_tar) in enumerate(zip(feature_ori, feature_tar)):
                em_ori, em_tar = em_ori.view(em_ori.shape[0], -1), em_tar.view(em_tar.shape[0], -1)
                if j == 0:
                    sim_mat = model.get_similarity_matrix(em_ori, em_tar)
                    loss += loss_weights[0] * criterion(sim_mat, torch.arange(0, origin.size(0)).cuda())
                    loss += loss_weights[0] * criterion(sim_mat.t(), torch.arange(0, origin.size(0)).cuda())
                    
                    kl_matrix = kl_divergence(em_ori, em_tar)
                    loss += loss_weights[1] * criterion(kl_matrix, torch.arange(0, em_ori.size(0)).cuda())
                    kl_matrix = kl_divergence(em_tar, em_ori)
                    loss += loss_weights[1] * criterion(kl_matrix, torch.arange(0, em_ori.size(0)).cuda())
                elif j == 1:
                    if cel_utils:
                        loss += loss_weights[2] * cel_utils['loss_func'](em_ori, em_tar, torch.ones((em_ori.size(0), )).cuda())
                    elif wd_utils:
                        loss += loss_weights[2] * get_wd_loss(em_ori, em_tar, wd_utils, wd_utils['stage'])
                elif j == 2:
                    loss += loss_weights[3] * get_wd_loss(em_ori, em_tar, wd_utils, wd_utils['stage'])
        else:
            for j, (em_ori, em_tar) in enumerate(zip(feature_ori, feature_tar)):
                em_ori, em_tar = em_ori.view(em_ori.shape[0], -1), em_tar.view(em_tar.shape[0], -1)
                if j == 0:
                    sim_mat = model.get_similarity_matrix(em_ori, em_tar)
                    loss += criterion(sim_mat, torch.arange(0, origin.size(0)).cuda())
                    loss += criterion(sim_mat.t(), torch.arange(0, origin.size(0)).cuda())
                    # print('CrossEntropyLoss: ', loss - 0)
                    
                    kl_matrix = kl_divergence(em_ori, em_tar)
                    kl_loss = criterion(kl_matrix, torch.arange(0, em_ori.size(0)).cuda())
                    kl_matrix = kl_divergence(em_tar, em_ori)
                    kl_loss += criterion(kl_matrix, torch.arange(0, em_ori.size(0)).cuda())
                    loss += kl_loss
                    
                    # print('KL_Divergence: ', loss)
                elif j == 1:
                    if cel_utils:
                        cel_loss = cel_utils['loss_func'](em_ori, em_tar, torch.ones((em_ori.size(0), )).cuda())
                        loss += cel_loss
                        # print('CEL: ', cel_loss)
                    elif wd_utils:
                        wd_loss = get_wd_loss(em_ori, em_tar, wd_utils, wd_utils['stage'])
                        loss += wd_loss
                        # print('WD loss: ', wd_loss) 
                elif j == 2:
                    wd_loss = get_wd_loss(em_ori, em_tar, wd_utils, wd_utils['stage'])
                    loss += wd_loss
                    # print('WD loss: ', wd_loss)
            

        train_loss.update(loss.item(), origin.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # if adp_weights:
        #     adp_weights['optim'].step()
        #     adp_weights['optim'].zero_grad()
            
        
        batch_time.update(time.time() - start)
        start = time.time()

        if i % 5 == 0:
            train_progress.display(i)

    model.eval()
    with torch.no_grad():
        for i, (origin, target, label) in enumerate(val_data):
            origin, target, label = origin.cuda(), target.cuda(), label.cuda()
            em_ori, em_tar = model(origin, target, cel_stage=cel_utils['stage'], wd_stage=wd_utils['stage'])
            em_ori, em_tar = em_ori[0], em_tar[0]
            sim_mat = model.get_similarity_matrix(em_ori, em_tar)
            loss = criterion(sim_mat, torch.arange(0, origin.size(0)).cuda()) + \
                   criterion(sim_mat.t(), torch.arange(0, origin.size(0)).cuda())
            
            loss *= loss_weights[0] / (loss_weights[0] + loss_weights[1]) if adp_weights else 0.5
            
            kl_matrix = kl_divergence(em_ori, em_tar)
            kl_loss = criterion(kl_matrix, torch.arange(0, em_ori.size(0)).cuda())
            kl_matrix = kl_divergence(em_tar, em_ori)
            kl_loss += criterion(kl_matrix, torch.arange(0, em_ori.size(0)).cuda())
            
            loss += loss_weights[1] / (loss_weights[0] + loss_weights[1]) * kl_loss if adp_weights else 0.5 * kl_loss
            
            em_ori, em_all_tar = model(origin, target_tensor)
            em_ori, em_all_tar = em_ori[0], em_all_tar[0]
            sim_mat = model.get_similarity_matrix(em_ori, em_all_tar)
            # print(model.logit_scale)
            kl_matrix = kl_divergence(em_ori, em_all_tar) * model.logit_scale
            print(torch.max(sim_mat))
            print(torch.max(kl_matrix))
            sim_mat = (cel_utils['ratio'] * sim_mat + (1 - cel_utils['ratio']) * kl_matrix)
            acc_10, acc_20, acc_30 = cal_accuracy_top_k(sim_mat, label, top_k=(5, 10, 15))

            val_loss.update(loss.item(), origin.size(0))
            val_acc_10.update(acc_10.item(), origin.size(0))
            val_acc_20.update(acc_20.item(), origin.size(0))
            val_acc_30.update(acc_30.item(), origin.size(0))

        val_progress.display(i)

    if adp_weights:
        wandb.log({'Cosine Similarity weight': loss_weights[0].item(),
                'KL divergence weight': loss_weights[1].item(),
                'CEL weight': loss_weights[2].item(),
                'WD weight': loss_weights[3].item(),
                'epoch': current_epoch})
    
    return train_loss.avg, val_loss.avg, val_acc_10.avg, val_acc_20.avg, val_acc_30.avg


def train(args, logger, train_data, val_data, model, criterion, optimizer, total_epochs, save_folder='./',
          scheduler=None, wandb_config=None, device=None, cel_utils=None, wd_utils=None, adp_weights=None):
    target_tensor = get_CNI_tensor(device, 200)
    # target_tensor = get_CNI_tensor(device, 224, img_type='RGB')

    for epoch in range(total_epochs):
        train_loss, val_loss, val_acc_10, val_acc_20, val_acc_30 = \
            train_epoch(train_data, val_data, model, criterion, optimizer, epoch + 1, total_epochs, target_tensor,
                        cel_utils, wd_utils=wd_utils, adp_weights=adp_weights)

        lr_info = ''
        if scheduler:
            lr_info = f'lr {scheduler.get_last_lr()}'
            scheduler.step()

        logger.info(f'Epoch: [{epoch}/{total_epochs}], train loss {train_loss}, '
                    f'val loss {val_loss}, val acc @ 5 {val_acc_10}, val acc @ 10 {val_acc_20}, '
                    f'val acc @ 15 {val_acc_30}' + lr_info)

        if wandb_config:
            wandb.log({'epoch': epoch + 1, 'train/train loss': train_loss,
                       'val/val loss': val_loss, 'val/val acc @ 5': val_acc_10, 'val/val acc @ 10': val_acc_20,
                       'val/val acc @ 15': val_acc_30})
    logger.info('Training Finished!')


def main():
    args = Config()
    # parser = argparse.ArgumentParser('Fine-tuning on NDI images', parents=[get_args_parser()])
    # args = parser.parse_args()

    device = torch.device(args.device)

    set_all_seeds(args.seed)

    logger_fname = datetime.datetime.now().strftime('%Y%m%d%H%M')
    logger = get_logger(f'./logs/{logger_fname}.log')

    logger.info(f'This training is to do: {args.message_to_log}')

    # test_list = itertools.product(['DSWD', 'DGSWD', 'MSWD', 'MGSWD', 'SWD', 'GSWD'])
    test_list = zip(np.arange(1, -0.1, -0.1), np.arange(0, 1.1, 0.1))
    # test_list = [[0.5, 0.5]]
    wd_stage = 2
    cel_stage = 2
    wd_name = 'MSWD'

    for test_item in test_list:
        cos_ratio, kl_ratio = round(test_item[0], 1), round(test_item[1], 1)

        for i, images in enumerate(k_fold_train_validation_split(ORIGINAL_IMAGE, TARGET_IMAGE, 7)):
            wandb.init(project=args.message_to_log, group=f'cel_kl_wd_false_KL',
                       job_type=f'{wd_name}_{wd_stage}_cel_{cel_stage}_{cos_ratio}_to_{kl_ratio}',
                       name=f'{wd_name}_{wd_stage}_cel_{cel_stage}_fold {i}', config=args.__dict__)
            train_dataset = SingleChannelNDIDatasetContrastiveLearningWithAug(images, False, 200)
            val_dataset = SingleChannelNDIDatasetContrastiveLearningWithAug(images, True, 200)
            # train_dataset = ThreeChannelNDIDataset(images, False, 224)
            # val_dataset = ThreeChannelNDIDataset(images, True, 224)
            train_iter = DataLoader(train_dataset, args.batch_size, shuffle=True, drop_last=True)
            val_iter = DataLoader(val_dataset, batch_size=len(val_dataset))

            model = get_model('ResNet50', pretrained=True)
            model = load_checkpoints(model, './checkpoints/ImageNet_ALL_CHECK_400_Epoch.pth')
            model = CompositeResNet(model)
            # model = get_model('ViT-B/32', pretrained=True)
            # model = load_checkpoints(model, './checkpoints/UNICOM_ViT_B_32_based.pth')
            # model = CompositeViT(model)
            model = RetrievalModel(model)
            model = model.cuda()
            
            ### Adaptive Parameters
            model_optimizer = torch.optim.SGD(params=model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay, momentum=args.momentum)
            # model_optimizer = torch.optim.AdamW(params=model.parameters(), betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

            # weights_optimizer = torch.optim.Adam([{'params': loss_weights}], lr=0.01)
            
            
            # optimizer = torch.optim.SGD(params=model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay, momentum=args.momentum)
            criterion = nn.CrossEntropyLoss()

            # scheduler = None
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optimizer, T_max=args.epochs, eta_min=args.base_lr * 0.01)
            cel_utils = {}
            cel_utils['loss_func'] = nn.CosineEmbeddingLoss(margin=0.5)

            if cel_utils:
                cel_utils['stage'] = cel_stage
                
                ###
                cel_utils['ratio'] = cos_ratio

            wd_utils = {}
            if wd_name == 'DSWD':
                wd_utils = {
                    'type': wd_name,
                    'num_projections': 1024,
                    'p': 2,
                    'max_iter': 10,
                    'lam': 1
                }
                channels = [512, 2048, 1024, 512, 256, 64]
                transform_net = TransformNet(channels[wd_stage - 1]).to('cuda')
                optimizer_tran_net = torch.optim.Adam(transform_net.parameters(), lr=0.0005, betas=(0.5, 0.999))
                wd_utils[f'tran_net'] = transform_net
                wd_utils[f'op_trannet'] = optimizer_tran_net

            elif wd_name == 'DGSWD':
                wd_utils = {
                    'type': wd_name,
                    'num_projections': 1024,
                    'g_func': circular_function,
                    'r': 1000,
                    'p': 2,
                    'max_iter': 10,
                    'lam': 1
                }
                channels = [512, 2048, 1024, 512, 256, 64]
                transform_net = TransformNet(channels[wd_stage - 1]).to('cuda')
                optimizer_tran_net = torch.optim.Adam(transform_net.parameters(), lr=0.0005, betas=(0.5, 0.999))
                wd_utils[f'tran_net'] = transform_net
                wd_utils[f'op_trannet'] = optimizer_tran_net

            elif wd_name == 'MSWD':
                wd_utils = {
                    'type': wd_name,
                    'p': 2,
                    'max_iter': 10
                }

            elif wd_name == 'MGSWD':
                wd_utils = {
                    'type': wd_name,
                    'g_func': circular_function,
                    'r': 1000,
                    'p': 2,
                    'max_iter': 10
                }

            elif wd_name == 'SWD':
                wd_utils = {
                    'type': wd_name,
                    'num_projections': 1000,
                    'p': 2
                }

            elif wd_name == 'GSWD':
                wd_utils = {
                    'type': wd_name,
                    'g_func': circular_function,
                    'r': 1000,
                    'num_projections': 1000,
                    'p': 2
                }

            if wd_utils:
                wd_utils['stage'] = wd_stage
            
            
            train(args, logger, train_iter, val_iter, model, criterion, model_optimizer, args.epochs, scheduler=scheduler,
                  save_folder=args.output_dir, wandb_config=True, device=device, cel_utils=cel_utils, wd_utils=wd_utils)
            
            
            wandb.finish()


if __name__ == '__main__':
    main()

