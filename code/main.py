from ImageUtils import parse_record
from DataReader import load_data, train_vaild_split
from Model import Cifar
import numpy as np

import os

import Configure

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
    model.evaluate(x_valid, y_valid, [50, 100, 150, 180, 190, 200])

    # With hyperparameters determined in the first run, re-train
    # your model on the original train set.
    model.train(x_train, y_train, 200, 0)

    # After re-training, test the model on the test set.
    # model.evaluate takes parameters --> (test_set, test_label_set, list_ckpt_to_test)
    model.evaluate(x_test, y_test, [199])
    
    # After checking for the accuracy on the public test set, store the predicted probability for each image in npy file
    private_test_data = np.load('/content/drive/MyDrive/Colab Notebooks/Project/private_test_images_v3.npy')
    model.predict_prob(private_test_data, 189)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = Configure.configure()
    main(config)