class Config:
    def __init__(self):
        self.seed = 19981303
        self.epochs = 50
        self.batch_size = 16
        self.base_lr = 0.005
        self.warmup_epochs = 0
        self.validation_ratio = 0.2
        self.weight_decay = 1e-4
        self.accumulate_step = 1
        self.momentum = 0.9

        self.device = 'cuda'

        self.wandb_key = ''
        self.message_to_log = "your_wandb_setting"
        self.output_dir = './checkpoints/'

