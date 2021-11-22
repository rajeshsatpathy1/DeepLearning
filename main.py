from ImageUtils import parse_record
from DataReader import load_data, train_vaild_split
from Model import Cifar
import numpy as np

import os
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
    parser.add_argument("--checkpoints_dir", type=str, default='/content/drive/MyDrive/Colab Notebooks/Project/validation', help='model directory')

    return parser.parse_args()

def main(config):
    print("--- Preparing Data ---")

    # Set src directory for data loading
    data_dir = "/content/drive/MyDrive/Colab Notebooks/Project/cifar-10-batches-py/"

    # Load train and test data
    x_train, y_train, x_test, y_test = load_data(data_dir)

    # Split train data into train_new and valid
    x_train_new, y_train_new, x_valid, y_valid = train_vaild_split(x_train, y_train)

    model = Cifar(config).cuda()
    # Uncomment to get the network model
    # print(model)

    # Use the train_new set and the valid set to choose hyperparameters.
    # model.train takes parameters --> (train_set, train_label_set, num_epochs, ckpt_to_start(default should be 0))
    model.train(x_train_new, y_train_new, 160, 0)
    # model.test_or_validate(x_valid, y_valid, [50, 100, 150, 180, 190, 200])

    # With hyperparameters determined in the first run, re-train
    # your model on the original train set.
    # model.train(x_train, y_train, 200, 0)

    # After re-training, test your model on the test set.
    # Report testing accuracy in your hard-copy report.
    # model.test_or_validate takes parameters --> (test_set, test_label_set, list_ckpt_to_test)
    # model.test_or_validate(x_test, y_test, [199])
    
    # After checking for the accuracy on the public test set, store the predicted probability for each image in npy file
    # private_test_data = np.load('/content/drive/MyDrive/Colab Notebooks/Project/private_test_images_v3.npy')
    # model.predict_prob(private_test_data, 199)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = configure()
    main(config)