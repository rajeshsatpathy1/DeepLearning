import os
import pickle
import numpy as np

""" This script implements the functions for reading data.
"""

def load_data(data_dir):
    """ Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches are stored.
    
    Returns:
        x_train: An numpy array of shape [50000, 3072]. 
        (dtype=np.float32)
        y_train: An numpy array of shape [50000,]. 
        (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072]. 
        (dtype=np.float32)
        y_test: An numpy array of shape [10000,]. 
        (dtype=np.int32)
    """
    ### YOUR CODE HERE
    with open(data_dir + 'data_batch_1', 'rb') as f1:
        dict1 = pickle.load(f1, encoding='bytes')
    with open(data_dir + 'data_batch_2', 'rb') as f2:
        dict2 = pickle.load(f2, encoding='bytes')
    with open(data_dir + 'data_batch_3', 'rb') as f3:
        dict3 = pickle.load(f3, encoding='bytes')
    with open(data_dir + 'data_batch_4', 'rb') as f4:
        dict4 = pickle.load(f4, encoding='bytes')
    with open(data_dir + 'data_batch_5', 'rb') as f5:
        dict5 = pickle.load(f5, encoding='bytes')
    
    with open(data_dir + 'test_batch', 'rb') as ftest:
        test_batch = pickle.load(ftest, encoding='bytes')

    x_train = np.append(dict1[b'data'], dict2[b'data'], axis=0)
    x_train = np.append(x_train, dict3[b'data'], axis=0)
    x_train = np.append(x_train, dict4[b'data'], axis=0)
    x_train = np.append(x_train, dict5[b'data'], axis=0)
    
    y_train = np.append(dict1[b'labels'], dict2[b'labels'])
    y_train = np.append(y_train, dict3[b'labels'])
    y_train = np.append(y_train, dict4[b'labels'])
    y_train = np.append(y_train, dict5[b'labels'])

    x_test = test_batch[b'data']
    y_test = np.array(test_batch[b'labels'])

    print('Train data shape:',x_train.shape, '|Train data labels shape:', y_train.shape, '|Test data shape:', x_test.shape, '|Test data labels shape:', y_test.shape)
    # print(len(dict1[b'data'][0]))
    # print(dict2[b'batch_label'])    #Batch label unnecessary
    # print(len(dict[b'batch_label']), len(dict[b'labels']), len(dict[b'data']),len(dict[b'filenames']))
    # dict.keys() - [b'batch_label', b'labels', b'data', b'filenames']
    ### YOUR CODE HERE

    return x_train, y_train, x_test, y_test

def train_vaild_split(x_train, y_train, split_index=45000):
    """ Split the original training data into a new training dataset
        and a validation dataset.
    
    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        split_index: An integer.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]

    return x_train_new, y_train_new, x_valid, y_valid