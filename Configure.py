import argparse

def configure():
    parser = argparse.ArgumentParser()

    parser.add_argument("--resnet_size", type=int, default=6, 
                        help='n: Number of blocks in each stack, There are 3 stacks - ResNet Size = 9n+2')
    parser.add_argument("--batch_size", type=int, default=200, help='training batch size')
    parser.add_argument("--num_classes", type=int, default=10, help='number of classes')
    parser.add_argument("--save_interval", type=int, default=10, 
                        help='save the checkpoint when epoch MOD save_interval == 0')
    parser.add_argument("--first_num_filters", type=int, default=128, help='number of filters')
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='weight decay rate')
    parser.add_argument("--checkpoints_dir", type=str, default='/content/drive/MyDrive/Colab Notebooks/Project', help='model directory')

    return parser.parse_args()