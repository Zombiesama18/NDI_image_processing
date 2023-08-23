import itertools


class Config:
    def __init__(self):
        self.seed = 19981303
        self.epochs = 1000
        self.batch_size = 128
        self.save_steps = 100
        self.resume = False
        self.resume_checkpoint = None
        self.model = 'resnet50_imagenet21k'
        self.input_size = 200
        self.accumulate_step = 1
        
        self.lr = 5e-3
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.min_lr = 0
        self.warmup_epochs = 0
        
        self.dataset_dir = '../datasets/NDI_images/Integreted'
        self.output_dir = './checkpoints/'
        self.log_dir = './logs/'
        self.device = 'cuda'
        
        self.message_to_log = "re_implementation"
        self.wandb_key = ''
        
        self.image_mean = 0.0877
        self.image_std = 0.085


