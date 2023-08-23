from train_on_NDI import *
from wasserstein_loss import distributional_sliced_wasserstein_distance, \
    distributional_generalized_sliced_wasserstein_distance, max_sliced_wasserstein_distance, \
    max_generalized_sliced_wasserstein_distance, sliced_wasserstein_distance, generalized_sliced_wasserstein_distance, \
    circular_function
from models.wd_model import TransformNet, CompositeResNet, CompositeViT
import itertools

def kl_divergence(p, q):
    kl_div = f.kl_div(f.log_softmax(p, dim=-1), f.softmax(q, dim=-1), reduction='none')
    return kl_div.sum(-1).mean()


def train_epoch(train_data, val_data, model, criterion, optimizer, current_epoch, total_epoch, target_tensor,
                kl_utils=None, cel_utils=None):
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
        feature_ori, feature_tar = model(origin, target, stage=kl_utils['stage'])
        loss = 0
        weight = 1
        for j, (em_ori, em_tar) in enumerate(zip(feature_ori, feature_tar)):
            em_ori, em_tar = em_ori.view(em_ori.shape[0], -1), em_tar.view(em_tar.shape[0], -1)
            if j == 0:
                sim_mat = model.get_similarity_matrix(em_ori, em_tar)
                loss += criterion(sim_mat, torch.arange(0, origin.size(0)).cuda())
                loss += criterion(sim_mat.t(), torch.arange(0, origin.size(0)).cuda())
            elif j == 1:
                if cel_utils:
                    loss += cel_utils['loss_func'](em_ori, em_tar, torch.ones((em_ori.size(0), )).cuda())
                elif kl_utils:
                    loss += kl_divergence(em_ori, em_tar) + kl_divergence(em_tar, em_ori)
            elif j == 2:
                loss += kl_divergence(em_ori, em_tar) + kl_divergence(em_tar, em_ori)
        train_loss.update(loss.item(), origin.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - start)
        start = time.time()

        if i % 5 == 0:
            train_progress.display(i)

    model.eval()
    with torch.no_grad():
        for i, (origin, target, label) in enumerate(val_data):
            origin, target, label = origin.cuda(), target.cuda(), label.cuda()
            em_ori, em_tar = model(origin, target)
            em_ori, em_tar = em_ori[0], em_tar[0]
            sim_mat = model.get_similarity_matrix(em_ori, em_tar)
            loss = criterion(sim_mat, torch.arange(0, origin.size(0)).cuda()) + \
                   criterion(sim_mat.t(), torch.arange(0, origin.size(0)).cuda())
            em_ori, em_all_tar = model(origin, target_tensor)
            em_ori, em_all_tar = em_ori[0], em_all_tar[0]
            sim_mat = model.get_similarity_matrix(em_ori, em_all_tar)
            acc_10, acc_20, acc_30 = cal_accuracy_top_k(sim_mat, label, top_k=(5, 10, 15))

            val_loss.update(loss.item(), origin.size(0))
            val_acc_10.update(acc_10.item(), origin.size(0))
            val_acc_20.update(acc_20.item(), origin.size(0))
            val_acc_30.update(acc_30.item(), origin.size(0))

        val_progress.display(i)

    return train_loss.avg, val_loss.avg, val_acc_10.avg, val_acc_20.avg, val_acc_30.avg


def train(args, logger, train_data, val_data, model, criterion, optimizer, total_epochs, save_folder='./',
          scheduler=None, wandb_config=None, device=None, kl_utils=None, cel_utils=None):
    global index_of_k
    index_of_k += 1
    target_tensor = get_CNI_tensor(device, 200)
    best_score = 0.0
    # target_tensor = get_CNI_tensor(device, 224, img_type='RGB')

    for epoch in range(total_epochs):
        train_loss, val_loss, val_acc_10, val_acc_20, val_acc_30 = \
            train_epoch(train_data, val_data, model, criterion, optimizer, epoch + 1, total_epochs, target_tensor,
                        kl_utils, cel_utils)

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
        if val_acc_10 > best_score:
            best_score = val_acc_10
            checkpoint_to_save = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                                  'optimizer': optimizer.state_dict}
            if scheduler:
                checkpoint_to_save.update(scheduler=scheduler.state_dict())
            if wandb_config:
                filename = f'Best {index_of_k}th Fold.pth'
            else:
                filename = datetime.datetime.now().strftime('%Y%m%d%H%M')
            save_checkpoint(checkpoint_to_save, filename=save_folder + filename)
            logger.info(f'Save Checkpoint at {epoch + 1} with train loss {train_loss}, '
                        f'val loss {val_loss}, val acc @ 10 {val_acc_10}, val acc @ 20 {val_acc_20}, '
                        f'val acc @ 30 {val_acc_30}' + lr_info)
    
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

    test_list = itertools.product([2], [2])
    # model_list = ['vit_tiny', 'vit_small', 'vit_base', 'vit_large']

    for test_item in test_list:

        kl_stage, cel_stage = test_item

        for i, images in enumerate(k_fold_train_validation_split(ORIGINAL_IMAGE, TARGET_IMAGE, 7)):
            wandb.init(project=args.message_to_log, group=f'KL_{kl_stage}_cel_{cel_stage}_Test_checkpoints',
                       job_type=f'KL_{kl_stage}_cel_{cel_stage}_save',
                       name=f'KL_{kl_stage}_cel_{cel_stage}_fold {i}', config=args.__dict__)
            train_dataset = SingleChannelNDIDatasetContrastiveLearningWithAug(images, False, 200)
            val_dataset = SingleChannelNDIDatasetContrastiveLearningWithAug(images, True, 200)
            # train_dataset = ThreeChannelNDIDataset(images, False, 224)
            # val_dataset = ThreeChannelNDIDataset(images, True, 224)
            
            train_iter = DataLoader(train_dataset, args.batch_size, shuffle=True, drop_last=True)
            val_iter = DataLoader(val_dataset, batch_size=len(val_dataset))

            model = get_model('ResNet50', pretrained=True)
            model = load_checkpoints(model, './checkpoints/ImageNet_ALL_CHECK_400_Epoch.pth')
            model = CompositeResNet(model, cel_stage)
            # model = get_model('ViT-B/32', pretrained=True)
            # model = load_checkpoints(model, './checkpoints/UNICOM_ViT_B_32_based.pth')
            # model = CompositeViT(model)
            model = RetrievalModel(model)
            model = model.cuda()

            optimizer = torch.optim.SGD(params=model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay, momentum=args.momentum)
            criterion = nn.CrossEntropyLoss()

            # scheduler = None
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs, eta_min=args.base_lr * 0.01)
            cel_utils = {}
            cel_utils['loss_func'] = nn.CosineEmbeddingLoss(margin=0.5)
            if cel_utils:
                cel_utils['stage'] = cel_stage
            
            kl_utils = {'stage': kl_stage}
            
            train(args, logger, train_iter, val_iter, model, criterion, optimizer, args.epochs, scheduler=scheduler,
                  save_folder=args.output_dir, wandb_config=True, device=device, kl_utils=kl_utils, cel_utils=cel_utils)
            wandb.finish()


if __name__ == '__main__':
    index_of_k = -1
    main()


