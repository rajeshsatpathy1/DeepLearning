from ImageUtils import parse_record
from DataReader import load_data, train_vaild_split
from Model import Cifar
import numpy as np

import os
import argparse

def configure():
    parser = argparse.ArgumentParser()
    ### YOUR CODE HERE
    # parser.add_argument("--resnet_version", type=int, default=1, help="the version of ResNet")
    parser.add_argument("--resnet_size", type=int, default=6, 
                        help='n: the size of ResNet-(6n+2) v1 or ResNet-(9n+2) v2')
    parser.add_argument("--batch_size", type=int, default=200, help='training batch size')
    parser.add_argument("--num_classes", type=int, default=10, help='number of classes')
    parser.add_argument("--save_interval", type=int, default=10, 
                        help='save the checkpoint when epoch MOD save_interval == 0')
    parser.add_argument("--first_num_filters", type=int, default=128, help='number of filters')
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='weight decay rate')
    parser.add_argument("--checkpoints_dir", type=str, default='/content/drive/MyDrive/Colab Notebooks/Project', help='model directory')
    ### YOUR CODE HERE
    return parser.parse_args()

def main(config):
    print("--- Preparing Data ---")

    ### YOUR CODE HERE
    data_dir = "/content/drive/MyDrive/Colab Notebooks/Project/cifar-10-batches-py/"
    ### YOUR CODE HERE
    # load_data(data_dir)
    x_train, y_train, x_test, y_test = load_data(data_dir)
    x_train_new, y_train_new, x_valid, y_valid = train_vaild_split(x_train, y_train)

    model = Cifar(config).cuda()
    # print(model)

    # First step: use the train_new set and the valid set to choose hyperparameters.
    # model.train(x_train_new, y_train_new, 160, 40)
    # model.test_or_validate(x_valid, y_valid, [50, 100, 150, 180, 190, 200])

    # Second step: with hyperparameters determined in the first run, re-train
    # your model on the original train set.
    model.train(x_train, y_train, 200, 0)

    # Third step: after re-training, test your model on the test set.
    # Report testing accuracy in your hard-copy report.
    # model.test_or_validate(x_test, y_test, [199])
    
    
    private_test_data = np.load('/content/drive/MyDrive/Colab Notebooks/Project/private_test_images_v3.npy')
    model.predict_prob(private_test_data, 199)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = configure()
    main(config)