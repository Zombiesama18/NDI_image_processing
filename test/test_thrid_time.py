from models.core_model import CalculatedModel
import torch
from utils.utils import get_wandb_API_key, AverageMeter, cal_accuracy_top_k
from finetuning_on_NDI import get_model, load_checkpoints
from dataset.dataset import create_train_val_dataset
import wandb
from losses.focal_loss import focal_loss


current_seed = torch.initial_seed()

def init_wandb():

    wandb.login(key=get_wandb_API_key())
    wandb.init(project="Sperated-Categories-Angles", name='Test-thrid-time-gamma-3.5-no-class-weights', config={'random_seed': current_seed})



train_set, val_set = create_train_val_dataset('../datasets/NDI_images/annotation.csv')

model = get_model(pretrained=True)
model = load_checkpoints(model, './checkpoints/ImageNet_ALL_CHECK_400_Epoch.pth')  # Pre-trained NDI image model
model = CalculatedModel(model, train_set.num_classes)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class_criterion = focal_loss(None, 3.5, device=device, dtype=torch.double)
angle_criterion = torch.nn.MSELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.005,
    momentum=0.9,
    weight_decay=1e-4
)

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
val_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=len(val_set)
)

init_wandb()

model.to(device)
wandb.watch(model)
model.double()
for epoch in range(50):
    model.train()
    loss = AverageMeter()
    train_class_loss = AverageMeter()
    train_angle_loss = AverageMeter()
    for i, (img_tensor, class_label, angle_label) in enumerate(train_loader):
        img_tensor, class_label, angle_label = img_tensor.to(device), class_label.to(device), angle_label.to(device)
        img_tensor, angle_label = img_tensor.double(), angle_label.double()
        angle_label = angle_label.view(-1, 1)
        optimizer.zero_grad()
        class_logits, angle_logits = model(img_tensor)
        batch_loss, class_loss, angle_loss = model.compute_loss(
            (class_criterion, angle_criterion),
            (class_logits, angle_logits),
            (class_label, angle_label)
        )
        batch_loss.backward()
        optimizer.step()
        loss.update(batch_loss.item(), img_tensor.shape[0])
        train_class_loss.update(class_loss.item(), img_tensor.shape[0])
        train_angle_loss.update(angle_loss.item(), img_tensor.shape[0])
    print(f'Epoch {epoch}, Loss: {loss.avg}')
    wandb.log({'train/total_loss': loss.avg, 'train/class_loss': train_class_loss.avg, 'train/angle_loss': train_angle_loss.avg})
    
    model.eval()
    val_class_acc = AverageMeter()
    val_angle_loss = AverageMeter()
    with torch.no_grad():
        for img_tensor, class_label, angle_label in val_loader:
            img_tensor, class_label, angle_label = img_tensor.to(device), class_label.to(device), angle_label.to(device)
            img_tensor, angle_label = img_tensor.double(), angle_label.double()
            angle_label = angle_label.view(-1, 1)
            class_logits, angle_logits = model(img_tensor)
            class_acc = cal_accuracy_top_k(class_logits, class_label)[0]
            angle_loss = angle_criterion(angle_logits, angle_label)
            val_class_acc.update(class_acc.item(), img_tensor.shape[0])
            val_angle_loss.update(angle_loss.item(), img_tensor.shape[0])
    print(f'Epoch {epoch}, Val Class Acc: {val_class_acc.avg}, Val Angle Loss: {val_angle_loss.avg}')
    wandb.log({'val/class_acc': val_class_acc.avg, 'val/angle_loss': val_angle_loss.avg})
            
