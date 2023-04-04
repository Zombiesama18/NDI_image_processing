import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Fine-tuning on NDI images', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=40, type=int)

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float,
                        help="Layer scale initial values")

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=5e-3, metavar='LR',
                        help='learning rate (absolute lr)')

    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cosine schedulers that hit 0')

    # Dataset Parameters
    parser.add_argument('--output_dir', default='./checkpoints/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./logs/',
                        help='path where to save the log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=19981303, type=int)

    # Wandb Parameters
    parser.add_argument('--project', default='Test which ViT suits NDI images best', type=str,
                        help="The name of the W&B project where you're sending the new run.")

    return parser


parser = argparse.ArgumentParser('Fine-tuning on NDI images', parents=[get_args_parser()])
args = parser.parse_args()
print(args.project)
print(type(args.project))
